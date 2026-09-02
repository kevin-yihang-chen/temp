from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .dataset import group_by_decision
from .infographicvqa_decar import DECAR_ACTION_IDS, _require_torch
from .qwen_semantic import validate_semantic_feature_dataset
from .schema import ActionRecord


ATTENTION_WHERE_SCHEMA = "infographicvqa_attention_where_feature_audit_v1"
ATTENTION_WHERE_TOP_LAYERS = 4
ATTENTION_WHERE_NORMALIZATION_ATOL = 1e-6


@dataclass(frozen=True)
class AttentionWhereFeatures:
    state_ids: tuple[str, ...]
    replicate_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    image_ids: tuple[str, ...]
    scores: Any
    image_attention_mass: Any
    selected_indices: Any
    margins: Any

    @property
    def decisions(self) -> int:
        return len(self.state_ids)


def _quantiles(values: Any) -> dict[str, float]:
    torch = _require_torch()
    if values.ndim != 1 or values.numel() == 0 or not torch.isfinite(values).all():
        raise ValueError("attention-where quantiles require finite values")
    probabilities = torch.tensor(
        [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], dtype=torch.float64
    )
    quantiles = torch.quantile(values.to(torch.float64), probabilities)
    return {
        name: float(value)
        for name, value in zip(
            ("q00", "q10", "q25", "q50", "q75", "q90", "q100"),
            quantiles.tolist(),
        )
    }


def assemble_attention_where_features(
    records: Sequence[ActionRecord],
    feature_payload: Mapping[str, Any],
    *,
    expected_code_revision: str,
    expected_model_revision: str,
    expected_source_features_sha256: str,
    expected_rollouts_sha256: str,
) -> tuple[AttentionWhereFeatures, dict[str, Any]]:
    """Audit and assemble outcome-free raw attention crop scores."""

    torch = _require_torch()
    if not all(
        (
            expected_code_revision,
            expected_model_revision,
            expected_source_features_sha256,
            expected_rollouts_sha256,
        )
    ):
        raise ValueError("attention-where expected bindings must be non-empty")
    metadata = feature_payload.get("metadata")
    decisions = feature_payload.get("decisions")
    if (
        feature_payload.get("format_version") != 1
        or not isinstance(metadata, Mapping)
        or bool(metadata.get("outcomes_included", True))
        or not isinstance(decisions, list)
    ):
        raise ValueError("attention-where requires outcome-free semantic features")
    attention_metadata = metadata.get("question_region_attention")
    if not isinstance(attention_metadata, Mapping):
        raise ValueError("attention-where metadata is missing")
    expected_metadata = {
        "source_features_sha256": expected_source_features_sha256,
        "source_rollouts_sha256": expected_rollouts_sha256,
        "model_revision": expected_model_revision,
        "attention_implementation": "eager",
        "top_layers": ATTENTION_WHERE_TOP_LAYERS,
        "head_pooling": "mean",
        "question_token_pooling": "mean",
        "candidate_pooling": "ROI mean then normalize across candidates",
        "candidate_actions_executed": False,
        "replace_question_embedding": False,
        "code_revision": expected_code_revision,
        "completed_decisions": len(decisions),
        "total_decisions": len(decisions),
    }
    for name, expected in expected_metadata.items():
        if attention_metadata.get(name) != expected:
            raise ValueError(f"attention-where metadata changed for {name}")

    validate_semantic_feature_dataset(
        feature_payload,
        records,
        require_outcomes=False,
    )
    grouped = group_by_decision(records)
    if not grouped or len(decisions) != len(grouped):
        raise ValueError("attention-where population coverage changed")

    ordered = sorted(
        decisions,
        key=lambda decision: (
            str(decision["state_id"]),
            str(decision["replicate_id"]),
        ),
    )
    state_ids: list[str] = []
    replicate_ids: list[str] = []
    source_ids: list[str] = []
    image_ids: list[str] = []
    score_rows: list[Any] = []
    masses: list[float] = []
    for decision in ordered:
        key = (str(decision["state_id"]), str(decision["replicate_id"]))
        siblings = grouped.get(key)
        if siblings is None:
            raise ValueError("attention-where decision is absent from rollouts")
        baseline = next(
            (record for record in siblings if record.action_type == "ANSWER"), None
        )
        zooms = sorted(
            (record for record in siblings if record.action_type == "ZOOM"),
            key=lambda record: record.action_id,
        )
        if (
            baseline is None
            or tuple(record.action_id for record in zooms) != DECAR_ACTION_IDS
            or tuple(str(value) for value in decision.get("action_ids", ()))
            != DECAR_ACTION_IDS
        ):
            raise ValueError("attention-where action family changed")
        scores = decision.get("question_region_attention")
        if (
            not isinstance(scores, torch.Tensor)
            or scores.shape != (len(DECAR_ACTION_IDS),)
            or not torch.isfinite(scores).all()
            or bool((scores < 0.0).any())
            or not math.isclose(
                float(scores.sum()),
                1.0,
                rel_tol=0.0,
                abs_tol=ATTENTION_WHERE_NORMALIZATION_ATOL,
            )
        ):
            raise ValueError("attention-where scores are invalid or unnormalized")
        mass = float(decision.get("question_image_attention_mass", math.nan))
        if not math.isfinite(mass) or mass <= 0.0:
            raise ValueError("attention-where image attention mass is invalid")
        if (
            str(decision.get("source_id")) != baseline.source_id
            or str(decision.get("image_id")) != baseline.image_id
        ):
            raise ValueError("attention-where identity changed")
        state_ids.append(key[0])
        replicate_ids.append(key[1])
        source_ids.append(baseline.source_id)
        image_ids.append(baseline.image_id)
        score_rows.append(scores.to(torch.float32).cpu())
        masses.append(mass)

    score_tensor = torch.stack(score_rows)
    mass_tensor = torch.tensor(masses, dtype=torch.float32)
    selected = torch.argmax(score_tensor, dim=1)
    ordered_scores = torch.sort(score_tensor, dim=1, descending=True, stable=True).values
    margins = ordered_scores[:, 0] - ordered_scores[:, 1]
    if not torch.isfinite(margins).all() or bool((margins < 0.0).any()):
        raise RuntimeError("attention-where margins are invalid")
    counts = {
        action_id: int((selected == index).sum())
        for index, action_id in enumerate(DECAR_ACTION_IDS)
    }
    features = AttentionWhereFeatures(
        state_ids=tuple(state_ids),
        replicate_ids=tuple(replicate_ids),
        source_ids=tuple(source_ids),
        image_ids=tuple(image_ids),
        scores=score_tensor,
        image_attention_mass=mass_tensor,
        selected_indices=selected,
        margins=margins,
    )
    audit = {
        "schema": ATTENTION_WHERE_SCHEMA,
        "passed": True,
        "validation_or_test_inputs_used": False,
        "outcomes_included": False,
        "candidate_actions_executed": False,
        "population": {
            "decisions": features.decisions,
            "sources": len(set(source_ids)),
            "images": len(set(image_ids)),
        },
        "action_ids": list(DECAR_ACTION_IDS),
        "selected_action_counts": counts,
        "selected_action_rates": {
            name: count / features.decisions for name, count in counts.items()
        },
        "score_sum_max_absolute_error": float(
            torch.max(torch.abs(score_tensor.sum(dim=1) - 1.0))
        ),
        "max_score_quantiles": _quantiles(score_tensor.max(dim=1).values),
        "margin_quantiles": _quantiles(margins),
        "question_image_attention_mass_quantiles": _quantiles(mass_tensor),
        "bindings": {
            "attention_code_revision": expected_code_revision,
            "model_revision": expected_model_revision,
            "source_features_sha256": expected_source_features_sha256,
            "rollouts_sha256": expected_rollouts_sha256,
        },
    }
    return features, audit
