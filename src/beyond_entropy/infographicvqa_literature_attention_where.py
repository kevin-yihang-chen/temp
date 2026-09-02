from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .attention_features import normalized_question_region_attention
from .dataset import group_by_decision
from .infographicvqa_decar import DECAR_ACTION_IDS
from .qwen_semantic import validate_semantic_feature_dataset
from .schema import ActionRecord
from .semantic import require_torch

VICROP_QWEN25_LAYER_INDEX = 22
VICROP_ANSWER_SUFFIX = " Answer the question using a single word or phrase."
VICROP_GENERIC_QUESTION = "Write a general description of the image."
LITERATURE_ATTENTION_AUDIT_SCHEMA = (
    "infographicvqa_literature_attention_where_feature_audit_v1"
)


@dataclass(frozen=True)
class LiteratureAttentionScores:
    candidate_scores: Any
    selected_layer: int | None
    layer_scores: Any | None
    zero_map_fallback: bool


@dataclass(frozen=True)
class LiteratureAttentionFeatures:
    state_ids: tuple[str, ...]
    replicate_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    image_ids: tuple[str, ...]
    vicrop_scores: Any
    laser_scores: Any
    vicrop_selected_indices: Any
    laser_selected_indices: Any
    vicrop_margins: Any
    laser_margins: Any
    laser_selected_layers: Any
    laser_zero_map_fallbacks: Any
    encore_early_entropy: Any

    @property
    def decisions(self) -> int:
        return len(self.state_ids)


def _finite_nonnegative_grid(value: Any, *, name: str) -> Any:
    require_torch()
    import torch  # type: ignore[import-not-found]

    tensor = value.float()
    if (
        tensor.ndim != 2
        or tensor.numel() == 0
        or not bool(torch.isfinite(tensor).all())
        or bool((tensor < 0.0).any())
    ):
        raise ValueError(f"{name} must be a finite nonnegative 2D grid")
    return tensor


def vicrop_relative_candidate_scores(
    query_attention: Any,
    generic_attention: Any,
    bboxes: Any,
) -> LiteratureAttentionScores:
    """Project the audited Qwen2.5 ViCrop ratio into fixed candidate boxes."""

    require_torch()
    import torch  # type: ignore[import-not-found]

    query = _finite_nonnegative_grid(query_attention, name="query attention")
    generic = _finite_nonnegative_grid(generic_attention, name="generic attention")
    if query.shape != generic.shape:
        raise ValueError("ViCrop attention grids must have identical shapes")
    if bool((generic == 0.0).any()):
        raise ValueError("ViCrop official ratio has a zero denominator")
    relative = query / generic
    if not bool(torch.isfinite(relative).all()) or bool((relative < 0.0).any()):
        raise ValueError("ViCrop relative attention is invalid")
    scores = normalized_question_region_attention(relative, bboxes)
    return LiteratureAttentionScores(
        candidate_scores=scores,
        selected_layer=VICROP_QWEN25_LAYER_INDEX,
        layer_scores=None,
        zero_map_fallback=False,
    )


def laser_all_head_candidate_scores(
    query_attention: Any,
    no_query_attention: Any,
    bboxes: Any,
) -> LiteratureAttentionScores:
    """Project an explicitly adapted all-head LASER contrast into candidates."""

    require_torch()
    import torch  # type: ignore[import-not-found]

    query = query_attention.float()
    no_query = no_query_attention.float()
    if (
        query.ndim != 4
        or query.shape != no_query.shape
        or query.shape[0] == 0
        or query.shape[1] == 0
        or query.shape[2] == 0
        or query.shape[3] == 0
        or not bool(torch.isfinite(query).all())
        or not bool(torch.isfinite(no_query).all())
        or bool((query < 0.0).any())
        or bool((no_query < 0.0).any())
    ):
        raise ValueError(
            "LASER attention must have matching finite [layers, heads, height, width] grids"
        )
    contrast = (query - no_query).clamp_min(0.0)
    layer_scores = torch.linalg.vector_norm(contrast.flatten(start_dim=2), dim=2).mean(
        dim=1
    )
    selected_layer = int(torch.argmax(layer_scores).item())
    selected_map = contrast[selected_layer].mean(dim=0)
    if float(selected_map.sum()) == 0.0:
        candidate_count = int(bboxes.shape[0])
        if candidate_count <= 0:
            raise ValueError("LASER fallback requires at least one candidate")
        scores = torch.zeros(candidate_count, dtype=torch.float32)
        scores[0] = 1.0
        zero_map = True
    else:
        scores = normalized_question_region_attention(selected_map, bboxes)
        zero_map = False
    return LiteratureAttentionScores(
        candidate_scores=scores,
        selected_layer=selected_layer,
        layer_scores=layer_scores,
        zero_map_fallback=zero_map,
    )


def image_attention_entropy(attention_grid: Any) -> float:
    """Compute normalized image-token entropy for ENCORE-style diagnostics."""

    grid = _finite_nonnegative_grid(attention_grid, name="image attention")
    total = float(grid.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("image attention entropy requires positive mass")
    probabilities = grid / total
    positive = probabilities[probabilities > 0.0]
    return float(-(positive * positive.log()).sum())


def _score_summary(scores: Any) -> dict[str, Any]:
    require_torch()
    import torch  # type: ignore[import-not-found]

    selected = torch.argmax(scores, dim=1)
    sorted_scores = torch.sort(scores, dim=1, descending=True, stable=True).values
    margins = sorted_scores[:, 0] - sorted_scores[:, 1]
    counts = {
        action_id: int((selected == index).sum())
        for index, action_id in enumerate(DECAR_ACTION_IDS)
    }
    return {
        "selected": selected,
        "margins": margins,
        "counts": counts,
        "rates": {name: count / int(scores.shape[0]) for name, count in counts.items()},
        "score_sum_max_absolute_error": float(
            torch.max(torch.abs(scores.sum(dim=1) - 1.0))
        ),
    }


def assemble_literature_attention_where_features(
    records: Sequence[ActionRecord],
    feature_payload: Mapping[str, Any],
    *,
    expected_code_revision: str,
    expected_model_revision: str,
    expected_source_features_sha256: str,
    expected_rollouts_sha256: str,
) -> tuple[LiteratureAttentionFeatures, dict[str, Any]]:
    """Audit the frozen outcome-free ViCrop/LASER-bank feature payload."""

    require_torch()
    import torch  # type: ignore[import-not-found]

    from .infographicvqa_literature_attention_extraction import (
        LITERATURE_ATTENTION_ENCORE_LAYERS,
        LITERATURE_ATTENTION_FORMAT_VERSION,
        LITERATURE_ATTENTION_METADATA_KEY,
    )

    if not all(
        (
            expected_code_revision,
            expected_model_revision,
            expected_source_features_sha256,
            expected_rollouts_sha256,
        )
    ):
        raise ValueError("literature attention expected bindings must be non-empty")
    metadata = feature_payload.get("metadata")
    decisions = feature_payload.get("decisions")
    if (
        feature_payload.get("format_version") != 1
        or not isinstance(metadata, Mapping)
        or bool(metadata.get("outcomes_included", True))
        or not isinstance(decisions, list)
    ):
        raise ValueError("literature attention requires outcome-free semantic features")
    augmentation = metadata.get(LITERATURE_ATTENTION_METADATA_KEY)
    if not isinstance(augmentation, Mapping):
        raise ValueError("literature attention metadata is missing")
    expected_metadata = {
        "format_version": LITERATURE_ATTENTION_FORMAT_VERSION,
        "source_features_sha256": expected_source_features_sha256,
        "source_rollouts_sha256": expected_rollouts_sha256,
        "model_revision": expected_model_revision,
        "attention_implementation": "eager",
        "prefill_query_position": "final assistant-prefix token",
        "query_suffix": VICROP_ANSWER_SUFFIX,
        "generic_question": VICROP_GENERIC_QUESTION,
        "no_query_text_content": False,
        "vicrop_layer_index": VICROP_QWEN25_LAYER_INDEX,
        "vicrop_head_pooling": "mean all returned heads",
        "vicrop_formula": "query_attention / generic_attention; no epsilon",
        "laser_head_pooling": "all heads",
        "laser_layer_selection": (
            "mean head L2 norm of positive query-minus-no-query contrast"
        ),
        "encore_layers": list(LITERATURE_ATTENTION_ENCORE_LAYERS),
        "encore_head_pooling": (
            "mean all returned heads before normalized Shannon entropy"
        ),
        "candidate_pooling": "ROI mean then normalize across candidates",
        "candidate_actions_executed": False,
        "outcomes_included": False,
        "validation_or_test_inputs_used": False,
        "code_revision": expected_code_revision,
        "completed_decisions": len(decisions),
        "total_decisions": len(decisions),
    }
    for name, expected in expected_metadata.items():
        if augmentation.get(name) != expected:
            raise ValueError(f"literature attention metadata changed for {name}")

    validate_semantic_feature_dataset(
        dict(feature_payload), records, require_outcomes=False
    )
    grouped = group_by_decision(records)
    if not grouped or len(decisions) != len(grouped):
        raise ValueError("literature attention population coverage changed")
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
    vicrop_rows: list[Any] = []
    laser_rows: list[Any] = []
    laser_layers: list[int] = []
    laser_fallbacks: list[bool] = []
    encore_rows: list[Any] = []
    layer_count: int | None = None
    head_count: int | None = None
    grid_shapes: Counter[tuple[int, int, int]] = Counter()
    image_masses: list[Any] = []
    for decision in ordered:
        key = (str(decision["state_id"]), str(decision["replicate_id"]))
        siblings = grouped.get(key)
        if siblings is None:
            raise ValueError("literature attention decision is absent from rollouts")
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
            raise ValueError("literature attention action family changed")
        current_layer_count = int(decision.get("literature_attention_layer_count", 0))
        current_head_count = int(decision.get("literature_attention_head_count", 0))
        if current_layer_count <= VICROP_QWEN25_LAYER_INDEX or current_head_count <= 0:
            raise ValueError("literature attention layer/head count is invalid")
        if layer_count is None:
            layer_count, head_count = current_layer_count, current_head_count
        elif (current_layer_count, current_head_count) != (layer_count, head_count):
            raise ValueError("literature attention layer/head count changed")
        vicrop = decision.get("vicrop_relative_region_attention")
        laser = decision.get("laser_contrastive_region_attention")
        encore = decision.get("encore_early_entropy")
        layer_scores = decision.get("laser_layer_scores")
        masses = decision.get("literature_attention_image_mass")
        grid = decision.get("literature_attention_grid_thw")
        if (
            not isinstance(vicrop, torch.Tensor)
            or vicrop.shape != (len(DECAR_ACTION_IDS),)
            or not isinstance(laser, torch.Tensor)
            or laser.shape != (len(DECAR_ACTION_IDS),)
            or not isinstance(encore, torch.Tensor)
            or encore.shape != (len(LITERATURE_ATTENTION_ENCORE_LAYERS),)
            or not isinstance(layer_scores, torch.Tensor)
            or layer_scores.shape != (current_layer_count,)
            or not isinstance(masses, torch.Tensor)
            or masses.shape != (3,)
            or not isinstance(grid, torch.Tensor)
            or grid.shape != (3,)
        ):
            raise ValueError("literature attention decision tensor shape changed")
        tensors = (vicrop, laser, encore, layer_scores, masses)
        if any(not bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("literature attention decision contains nonfinite values")
        if (
            bool((vicrop < 0.0).any())
            or bool((laser < 0.0).any())
            or bool((encore < 0.0).any())
            or bool((layer_scores < 0.0).any())
            or bool((masses <= 0.0).any())
            or not math.isclose(float(vicrop.sum()), 1.0, abs_tol=1e-6)
            or not math.isclose(float(laser.sum()), 1.0, abs_tol=1e-6)
        ):
            raise ValueError("literature attention scores or diagnostics are invalid")
        selected_layer = int(decision.get("laser_selected_layer", -1))
        zero_map_fallback = bool(decision.get("laser_zero_map_fallback", False))
        if (
            selected_layer < 0
            or selected_layer >= current_layer_count
            or selected_layer != int(torch.argmax(layer_scores).item())
        ):
            raise ValueError("literature attention LASER layer selection changed")
        if zero_map_fallback != bool(float(layer_scores.max()) == 0.0):
            raise ValueError("literature attention LASER fallback flag changed")
        raw_grid_values = [int(value) for value in grid.tolist()]
        if len(raw_grid_values) != 3:
            raise ValueError("literature attention visual grid rank changed")
        raw_grid = (
            raw_grid_values[0],
            raw_grid_values[1],
            raw_grid_values[2],
        )
        if raw_grid[0] != 1 or raw_grid[1] <= 0 or raw_grid[2] <= 0:
            raise ValueError("literature attention visual grid is invalid")
        if (
            str(decision.get("source_id")) != baseline.source_id
            or str(decision.get("image_id")) != baseline.image_id
        ):
            raise ValueError("literature attention identity changed")
        state_ids.append(key[0])
        replicate_ids.append(key[1])
        source_ids.append(baseline.source_id)
        image_ids.append(baseline.image_id)
        vicrop_rows.append(vicrop.to(torch.float32).cpu())
        laser_rows.append(laser.to(torch.float32).cpu())
        laser_layers.append(selected_layer)
        laser_fallbacks.append(zero_map_fallback)
        encore_rows.append(encore.to(torch.float32).cpu())
        image_masses.append(masses.to(torch.float32).cpu())
        grid_shapes[raw_grid] += 1

    vicrop_scores = torch.stack(vicrop_rows)
    laser_scores = torch.stack(laser_rows)
    vicrop_summary = _score_summary(vicrop_scores)
    laser_summary = _score_summary(laser_scores)
    selected_layers = torch.tensor(laser_layers, dtype=torch.int64)
    fallbacks = torch.tensor(laser_fallbacks, dtype=torch.bool)
    encore_tensor = torch.stack(encore_rows)
    mass_tensor = torch.stack(image_masses)
    features = LiteratureAttentionFeatures(
        state_ids=tuple(state_ids),
        replicate_ids=tuple(replicate_ids),
        source_ids=tuple(source_ids),
        image_ids=tuple(image_ids),
        vicrop_scores=vicrop_scores,
        laser_scores=laser_scores,
        vicrop_selected_indices=vicrop_summary["selected"],
        laser_selected_indices=laser_summary["selected"],
        vicrop_margins=vicrop_summary["margins"],
        laser_margins=laser_summary["margins"],
        laser_selected_layers=selected_layers,
        laser_zero_map_fallbacks=fallbacks,
        encore_early_entropy=encore_tensor,
    )
    audit = {
        "schema": LITERATURE_ATTENTION_AUDIT_SCHEMA,
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
        "language_layers": layer_count,
        "attention_heads": head_count,
        "visual_grid_shape_counts": {
            "x".join(str(value) for value in shape): count
            for shape, count in sorted(grid_shapes.items())
        },
        "vicrop_relative_bank": {
            "selected_action_counts": vicrop_summary["counts"],
            "selected_action_rates": vicrop_summary["rates"],
            "score_sum_max_absolute_error": vicrop_summary[
                "score_sum_max_absolute_error"
            ],
        },
        "laser_contrastive_all_head_bank": {
            "selected_action_counts": laser_summary["counts"],
            "selected_action_rates": laser_summary["rates"],
            "score_sum_max_absolute_error": laser_summary[
                "score_sum_max_absolute_error"
            ],
            "selected_layer_counts": {
                str(index): int((selected_layers == index).sum())
                for index in range(int(layer_count or 0))
            },
            "zero_map_fallbacks": int(fallbacks.sum()),
        },
        "encore_entropy": {
            "layers": list(LITERATURE_ATTENTION_ENCORE_LAYERS),
            "min": encore_tensor.min(dim=0).values.tolist(),
            "mean": encore_tensor.mean(dim=0).tolist(),
            "max": encore_tensor.max(dim=0).values.tolist(),
        },
        "image_attention_mass": {
            "query_layer_22_min": float(mass_tensor[:, 0].min()),
            "generic_layer_22_min": float(mass_tensor[:, 1].min()),
            "no_query_layer_22_min": float(mass_tensor[:, 2].min()),
        },
        "bindings": {
            "attention_code_revision": expected_code_revision,
            "model_revision": expected_model_revision,
            "source_features_sha256": expected_source_features_sha256,
            "rollouts_sha256": expected_rollouts_sha256,
        },
    }
    return features, audit
