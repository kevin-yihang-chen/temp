from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .dataset import group_by_decision
from .schema import ActionRecord


DECAR_SEED = 20_260_917
DECAR_PROJECTION_DIM = 128
DECAR_FUSION_DIM = 912
DECAR_WHEN_DIM = 914
DECAR_HIDDEN_DIMS = (256, 64)
DECAR_EPOCHS = 200
DECAR_LEARNING_RATE = 0.001
DECAR_WEIGHT_DECAY = 0.0001
DECAR_SMOOTH_L1_BETA = 1.0
DECAR_PAIRWISE_WEIGHT = 0.5
DECAR_TOOL_COST = 0.05
DECAR_ACTION_IDS = tuple(f"ug-grid-{index:02d}" for index in range(4))
DECAR_SCALAR_NAMES = (
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "bbox_center_x",
    "bbox_center_y",
    "ug_grid_size",
    "original_aspect_ratio",
    "log_original_pixel_count",
    "answer_generated_tokens",
    "answer_mean_normalized_token_entropy",
    "answer_max_normalized_token_entropy",
    "answer_mean_generated_token_log_probability",
)


@dataclass(frozen=True)
class DecarDataset:
    state_ids: tuple[str, ...]
    replicate_ids: tuple[str, ...]
    image_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    action_ids: tuple[tuple[str, ...], ...]
    question: Any
    global_visual: Any
    region: Any
    scalars: Any
    loss_gaps: Any
    task_deltas: Any
    correct_before: Any
    entropy_before: Any
    entropy_after: Any

    @property
    def decisions(self) -> int:
        return len(self.state_ids)

    @property
    def candidates(self) -> int:
        return len(DECAR_ACTION_IDS)


@dataclass(frozen=True)
class FusionStandardizer:
    question_mean: Any
    question_scale: Any
    global_mean: Any
    global_scale: Any
    region_mean: Any
    region_scale: Any
    scalar_mean: Any
    scalar_scale: Any

    def transform(
        self, question: Any, global_visual: Any, region: Any, scalars: Any
    ) -> tuple[Any, Any, Any, Any]:
        return (
            (question - self.question_mean) / self.question_scale,
            (global_visual - self.global_mean) / self.global_scale,
            (region - self.region_mean) / self.region_scale,
            (scalars - self.scalar_mean) / self.scalar_scale,
        )


@dataclass
class WhereFit:
    model: Any
    standardizer: FusionStandardizer
    target_mean: float
    target_scale: float
    audit: dict[str, Any]


@dataclass
class WhenFit:
    model: Any
    standardizer: FusionStandardizer
    gap_mean: Any
    gap_scale: Any
    rescue_magnitude: float
    harm_magnitude: float
    binary: bool
    audit: dict[str, Any]


def _require_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("InfographicVQA DECAR requires torch") from exc
    return torch


def _nll_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    expected_fields = {
        "action_id",
        "action_type",
        "answer_mean_nll",
        "answer_sum_nll",
        "answer_token_count",
        "config_sha256",
        "correct_after",
        "correct_before",
        "entropy_after",
        "entropy_before",
        "image_id",
        "replicate_id",
        "schema",
        "source_id",
        "state_id",
        "target_answer_count",
        "target_answer_index",
        "target_answer_sha256",
        "target_answer_votes",
        "tool_cost",
    }
    result: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        if (
            set(row) != expected_fields
            or row.get("schema") != "visual_action_answer_nll_v1"
        ):
            raise ValueError("DECAR answer-NLL row schema changed")
        key = (str(row["state_id"]), str(row["replicate_id"]), str(row["action_id"]))
        if key in result:
            raise ValueError("DECAR answer-NLL keys must be unique")
        mean_nll = float(row["answer_mean_nll"])
        sum_nll = float(row["answer_sum_nll"])
        tokens = int(row["answer_token_count"])
        if (
            not math.isfinite(mean_nll)
            or not math.isfinite(sum_nll)
            or tokens <= 0
            or not math.isclose(mean_nll * tokens, sum_nll, rel_tol=1e-5, abs_tol=1e-5)
        ):
            raise ValueError("DECAR answer-NLL values are invalid")
        result[key] = row
    return result


def _feature_index(
    feature_payload: Mapping[str, Any]
) -> dict[tuple[str, str], Mapping[str, Any]]:
    metadata = feature_payload.get("metadata")
    decisions = feature_payload.get("decisions")
    if (
        feature_payload.get("format_version") != 1
        or not isinstance(metadata, Mapping)
        or bool(metadata.get("outcomes_included", True))
        or not isinstance(decisions, Sequence)
    ):
        raise ValueError("DECAR requires a label-free semantic feature payload")
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for decision in decisions:
        if not isinstance(decision, Mapping):
            raise ValueError("DECAR semantic decision must be a mapping")
        key = (str(decision["state_id"]), str(decision["replicate_id"]))
        if key in result:
            raise ValueError("DECAR semantic decisions must be unique")
        forbidden = {
            "success_before",
            "success_after",
            "correct_before",
            "correct_after",
            "delta_success",
        }
        if forbidden & set(decision):
            raise ValueError("DECAR semantic decision contains outcome leakage")
        result[key] = decision
    return result


def _baseline_signals(record: ActionRecord) -> tuple[float, float, float, float]:
    metadata = record.metadata.get("baseline_backend")
    if not isinstance(metadata, Mapping):
        raise ValueError("DECAR baseline backend metadata is missing")
    generated_tokens = int(metadata.get("generated_tokens", 0))
    raw_entropies = metadata.get("normalized_token_entropies")
    raw_log_probabilities = metadata.get("generated_token_log_probabilities")
    if (
        generated_tokens <= 0
        or isinstance(raw_entropies, (str, bytes))
        or not isinstance(raw_entropies, Sequence)
        or len(raw_entropies) != generated_tokens
        or isinstance(raw_log_probabilities, (str, bytes))
        or not isinstance(raw_log_probabilities, Sequence)
        or len(raw_log_probabilities) != generated_tokens
    ):
        raise ValueError("DECAR generated-token statistics are incomplete")
    entropies = [float(value) for value in raw_entropies]
    log_probabilities = [float(value) for value in raw_log_probabilities]
    if not all(
        math.isfinite(value) and 0.0 <= value <= 1.0 for value in entropies
    ) or not all(math.isfinite(value) and value <= 0.0 for value in log_probabilities):
        raise ValueError("DECAR generated-token statistics are invalid")
    mean_entropy = sum(entropies) / len(entropies)
    mean_log_probability = sum(log_probabilities) / len(log_probabilities)
    stored_mean_log_probability = float(
        metadata.get("mean_generated_token_log_probability", math.nan)
    )
    if not math.isclose(
        record.entropy_before, mean_entropy, rel_tol=0.0, abs_tol=1e-7
    ) or not math.isclose(
        stored_mean_log_probability,
        mean_log_probability,
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise ValueError("DECAR generated-token summaries changed")
    return (
        float(generated_tokens),
        mean_entropy,
        max(entropies),
        mean_log_probability,
    )


def assemble_decar_dataset(
    records: Sequence[ActionRecord],
    nll_rows: Sequence[Mapping[str, Any]],
    feature_payload: Mapping[str, Any],
    image_geometry_by_id: Mapping[str, tuple[int, int]],
) -> DecarDataset:
    """Join the frozen sibling bank without using post-action fields as inputs."""

    torch = _require_torch()
    grouped = group_by_decision(records)
    nll_by_key = _nll_index(nll_rows)
    features_by_key = _feature_index(feature_payload)
    if not grouped:
        raise ValueError("DECAR sibling bank is empty")

    state_ids: list[str] = []
    replicate_ids: list[str] = []
    image_ids: list[str] = []
    source_ids: list[str] = []
    action_ids: list[tuple[str, ...]] = []
    questions: list[Any] = []
    globals_: list[Any] = []
    regions: list[Any] = []
    scalar_rows: list[list[list[float]]] = []
    loss_gaps: list[list[float]] = []
    deltas: list[list[float]] = []
    correct_before: list[float] = []
    entropy_before: list[float] = []
    entropy_after: list[list[float]] = []
    expected_nll_keys: set[tuple[str, str, str]] = set()

    for key in sorted(grouped):
        siblings = grouped[key]
        answers = [record for record in siblings if record.action_type == "ANSWER"]
        zooms = sorted(
            (record for record in siblings if record.action_type == "ZOOM"),
            key=lambda record: record.action_id,
        )
        if (
            len(answers) != 1
            or tuple(record.action_id for record in zooms) != DECAR_ACTION_IDS
        ):
            raise ValueError(
                "DECAR requires one ANSWER and the four registered UG actions"
            )
        baseline = answers[0]
        if any(
            record.source_id != baseline.source_id
            or record.image_id != baseline.image_id
            or not math.isclose(record.tool_cost, 1.0, rel_tol=0.0, abs_tol=0.0)
            for record in zooms
        ):
            raise ValueError("DECAR sibling identity or visual cost changed")
        feature = features_by_key.get(key)
        if feature is None:
            raise ValueError("DECAR semantic feature coverage is incomplete")
        if (
            str(feature.get("source_id")) != baseline.source_id
            or str(feature.get("image_id")) != baseline.image_id
            or tuple(str(value) for value in feature.get("action_ids", []))
            != DECAR_ACTION_IDS
        ):
            raise ValueError("DECAR semantic identities differ from rollouts")
        question_embedding = feature.get("question_embedding")
        global_embedding = feature.get("global_visual_embedding")
        region_embeddings = feature.get("region_embeddings")
        if (
            not isinstance(question_embedding, torch.Tensor)
            or not isinstance(global_embedding, torch.Tensor)
            or not isinstance(region_embeddings, torch.Tensor)
            or question_embedding.ndim != 1
            or global_embedding.shape != question_embedding.shape
            or region_embeddings.shape
            != (len(DECAR_ACTION_IDS), question_embedding.shape[0])
        ):
            raise ValueError("DECAR semantic tensor shapes changed")
        visual_grid_hw = feature.get("visual_grid_hw")
        if (
            not isinstance(visual_grid_hw, Sequence)
            or len(visual_grid_hw) != 2
            or min(int(value) for value in visual_grid_hw) <= 0
        ):
            raise ValueError("DECAR visual grid shape changed")
        geometry = image_geometry_by_id.get(baseline.image_id)
        if geometry is None:
            raise ValueError("DECAR original image geometry coverage is incomplete")
        width, height = (int(geometry[0]), int(geometry[1]))
        if width <= 0 or height <= 0:
            raise ValueError("DECAR original image geometry is invalid")
        answer_signals = _baseline_signals(baseline)
        baseline_nll_key = (
            baseline.state_id,
            baseline.replicate_id,
            baseline.action_id,
        )
        baseline_nll = nll_by_key.get(baseline_nll_key)
        if baseline_nll is None or baseline_nll.get("action_type") != "ANSWER":
            raise ValueError("DECAR baseline answer-NLL coverage is incomplete")
        expected_nll_keys.add(baseline_nll_key)

        candidate_scalars: list[list[float]] = []
        candidate_loss_gaps: list[float] = []
        candidate_deltas: list[float] = []
        candidate_entropy_after: list[float] = []
        for zoom in zooms:
            if zoom.candidate_bbox is None:
                raise ValueError("DECAR crop bounding box is missing")
            bbox = zoom.candidate_bbox
            grid_size = float(zoom.pre_action_features.get("ug_grid_size", math.nan))
            if not math.isfinite(grid_size) or grid_size <= 0.0:
                raise ValueError("DECAR full-grid size is invalid")
            candidate_scalars.append(
                [
                    bbox.x1,
                    bbox.y1,
                    bbox.x2,
                    bbox.y2,
                    bbox.width,
                    bbox.height,
                    bbox.area,
                    (bbox.x1 + bbox.x2) / 2.0,
                    (bbox.y1 + bbox.y2) / 2.0,
                    grid_size,
                    width / height,
                    math.log(width * height),
                    *answer_signals,
                ]
            )
            zoom_nll_key = (zoom.state_id, zoom.replicate_id, zoom.action_id)
            zoom_nll = nll_by_key.get(zoom_nll_key)
            if zoom_nll is None or zoom_nll.get("action_type") != "ZOOM":
                raise ValueError("DECAR crop answer-NLL coverage is incomplete")
            expected_nll_keys.add(zoom_nll_key)
            candidate_loss_gaps.append(
                float(baseline_nll["answer_mean_nll"])
                - float(zoom_nll["answer_mean_nll"])
            )
            candidate_deltas.append(zoom.delta_success)
            candidate_entropy_after.append(zoom.entropy_after)
        if any(len(values) != len(DECAR_SCALAR_NAMES) for values in candidate_scalars):
            raise RuntimeError("DECAR registered scalar dimension changed")

        state_ids.append(baseline.state_id)
        replicate_ids.append(baseline.replicate_id)
        image_ids.append(baseline.image_id)
        source_ids.append(baseline.source_id)
        action_ids.append(DECAR_ACTION_IDS)
        questions.append(question_embedding.to(torch.float32))
        globals_.append(global_embedding.to(torch.float32))
        regions.append(region_embeddings.to(torch.float32))
        scalar_rows.append(candidate_scalars)
        loss_gaps.append(candidate_loss_gaps)
        deltas.append(candidate_deltas)
        correct_before.append(baseline.correct_before)
        entropy_before.append(baseline.entropy_before)
        entropy_after.append(candidate_entropy_after)

    if set(features_by_key) != set(grouped) or set(nll_by_key) != expected_nll_keys:
        raise ValueError("DECAR joined feature or NLL coverage is not exact")
    tensors = {
        "question": torch.stack(questions),
        "global_visual": torch.stack(globals_),
        "region": torch.stack(regions),
        "scalars": torch.tensor(scalar_rows, dtype=torch.float32),
        "loss_gaps": torch.tensor(loss_gaps, dtype=torch.float32),
        "task_deltas": torch.tensor(deltas, dtype=torch.float32),
        "correct_before": torch.tensor(correct_before, dtype=torch.float32),
        "entropy_before": torch.tensor(entropy_before, dtype=torch.float32),
        "entropy_after": torch.tensor(entropy_after, dtype=torch.float32),
    }
    if not all(torch.isfinite(value).all() for value in tensors.values()):
        raise ValueError("DECAR joined tensors contain non-finite values")
    return DecarDataset(
        state_ids=tuple(state_ids),
        replicate_ids=tuple(replicate_ids),
        image_ids=tuple(image_ids),
        source_ids=tuple(source_ids),
        action_ids=tuple(action_ids),
        **tensors,
    )


def source_balanced_decision_weights(source_ids: Sequence[str]) -> Any:
    torch = _require_torch()
    if not source_ids or any(not source_id for source_id in source_ids):
        raise ValueError("DECAR source IDs must be non-empty")
    counts: dict[str, int] = {}
    for source_id in source_ids:
        counts[source_id] = counts.get(source_id, 0) + 1
    weights = torch.tensor(
        [1.0 / (len(counts) * counts[source_id]) for source_id in source_ids],
        dtype=torch.float32,
    )
    return weights / weights.sum()


def _weighted_mean_scale(values: Any, weights: Any) -> tuple[Any, Any]:
    torch = _require_torch()
    if values.ndim != 2 or weights.ndim != 1 or values.shape[0] != weights.shape[0]:
        raise ValueError("DECAR weighted standardizer arrays are not aligned")
    normalized = weights / weights.sum()
    mean = (values * normalized[:, None]).sum(dim=0)
    variance = ((values - mean) ** 2 * normalized[:, None]).sum(dim=0)
    scale = torch.sqrt(torch.clamp(variance, min=0.0))
    scale = torch.where(scale > 0.0, scale, torch.ones_like(scale))
    return mean, scale


def fit_fusion_standardizer(
    question: Any,
    global_visual: Any,
    region: Any,
    scalars: Any,
    decision_weights: Any,
) -> FusionStandardizer:
    torch = _require_torch()
    if (
        question.ndim != 2
        or global_visual.shape != question.shape
        or region.ndim not in (2, 3)
        or scalars.ndim not in (2, 3)
        or region.shape[0] != question.shape[0]
        or scalars.shape[0] != question.shape[0]
        or decision_weights.shape != (question.shape[0],)
    ):
        raise ValueError("DECAR fusion standardizer arrays are not aligned")
    question_mean, question_scale = _weighted_mean_scale(question, decision_weights)
    global_mean, global_scale = _weighted_mean_scale(global_visual, decision_weights)
    if region.ndim == 3:
        candidates = region.shape[1]
        candidate_weights = (
            decision_weights[:, None].expand(-1, candidates).reshape(-1) / candidates
        )
        flat_region = region.reshape(-1, region.shape[-1])
        flat_scalars = scalars.reshape(-1, scalars.shape[-1])
    else:
        candidate_weights = decision_weights
        flat_region = region
        flat_scalars = scalars
    region_mean, region_scale = _weighted_mean_scale(flat_region, candidate_weights)
    scalar_mean, scalar_scale = _weighted_mean_scale(flat_scalars, candidate_weights)
    if not all(
        torch.isfinite(value).all()
        for value in (
            question_mean,
            question_scale,
            global_mean,
            global_scale,
            region_mean,
            region_scale,
            scalar_mean,
            scalar_scale,
        )
    ):
        raise RuntimeError("DECAR standardization produced non-finite values")
    return FusionStandardizer(
        question_mean,
        question_scale,
        global_mean,
        global_scale,
        region_mean,
        region_scale,
        scalar_mean,
        scalar_scale,
    )


def _make_fusion_network(embedding_dim: int, *, output_dim: int) -> Any:
    torch = _require_torch()
    nn = torch.nn

    class FusionNetwork(nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.question_projection = nn.Linear(embedding_dim, DECAR_PROJECTION_DIM)
            self.global_projection = nn.Linear(embedding_dim, DECAR_PROJECTION_DIM)
            self.region_projection = nn.Linear(embedding_dim, DECAR_PROJECTION_DIM)
            self.mlp = nn.Sequential(
                nn.Linear(DECAR_FUSION_DIM, DECAR_HIDDEN_DIMS[0]),
                nn.GELU(),
                nn.Linear(DECAR_HIDDEN_DIMS[0], DECAR_HIDDEN_DIMS[1]),
                nn.GELU(),
                nn.Linear(DECAR_HIDDEN_DIMS[1], output_dim),
            )

        def fusion(
            self, question: Any, global_visual: Any, region: Any, scalars: Any
        ) -> Any:
            question_projected = self.question_projection(question)
            global_projected = self.global_projection(global_visual)
            if region.ndim == 3:
                question_projected = question_projected[:, None, :].expand(
                    -1, region.shape[1], -1
                )
                global_projected = global_projected[:, None, :].expand(
                    -1, region.shape[1], -1
                )
            region_projected = self.region_projection(region)
            fusion = torch.cat(
                (
                    question_projected,
                    global_projected,
                    region_projected,
                    question_projected * region_projected,
                    question_projected * global_projected,
                    global_projected * region_projected,
                    region_projected - global_projected,
                    scalars,
                ),
                dim=-1,
            )
            if fusion.shape[-1] != DECAR_FUSION_DIM:
                raise RuntimeError("DECAR fusion dimension changed")
            return fusion

        def forward(
            self, question: Any, global_visual: Any, region: Any, scalars: Any
        ) -> Any:
            return self.mlp(self.fusion(question, global_visual, region, scalars))

    return FusionNetwork()


def _set_deterministic(seed: int) -> None:
    torch = _require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _pairwise_loss(predictions: Any, targets: Any, decision_weights: Any) -> Any:
    torch = _require_torch()
    differences = targets[:, :, None] - targets[:, None, :]
    predicted_differences = predictions[:, :, None] - predictions[:, None, :]
    upper = torch.triu(
        torch.ones(
            targets.shape[1], targets.shape[1], dtype=torch.bool, device=targets.device
        ),
        diagonal=1,
    )
    mask = upper[None, :, :] & (differences != 0.0)
    pair_losses = torch.nn.functional.softplus(
        -torch.sign(differences) * predicted_differences
    )
    counts = mask.sum(dim=(1, 2))
    eligible = counts > 0
    if not bool(eligible.any()):
        return predictions.sum() * 0.0
    per_decision = (pair_losses * mask).sum(dim=(1, 2)) / counts.clamp(min=1)
    weights = decision_weights[eligible]
    return (per_decision[eligible] * weights).sum() / weights.sum()


def fit_where(
    question: Any,
    global_visual: Any,
    region: Any,
    scalars: Any,
    targets: Any,
    source_ids: Sequence[str],
    *,
    seed: int,
    device: str,
    epochs: int = DECAR_EPOCHS,
) -> WhereFit:
    torch = _require_torch()
    if epochs <= 0:
        raise ValueError("DECAR where epochs must be positive")
    _set_deterministic(seed)
    decision_weights = source_balanced_decision_weights(source_ids)
    standardizer = fit_fusion_standardizer(
        question, global_visual, region, scalars, decision_weights
    )
    transformed = standardizer.transform(question, global_visual, region, scalars)
    candidate_weights = decision_weights[:, None].expand_as(targets) / targets.shape[1]
    target_mean_tensor = (targets * candidate_weights).sum() / candidate_weights.sum()
    target_variance = (
        (targets - target_mean_tensor) ** 2 * candidate_weights
    ).sum() / candidate_weights.sum()
    target_scale_tensor = torch.sqrt(torch.clamp(target_variance, min=0.0))
    if (
        not bool(torch.isfinite(target_scale_tensor))
        or float(target_scale_tensor) <= 0.0
    ):
        raise ValueError("DECAR where target scale must be positive and finite")
    standardized_targets = (targets - target_mean_tensor) / target_scale_tensor
    model = _make_fusion_network(question.shape[-1], output_dim=1).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=DECAR_LEARNING_RATE, weight_decay=DECAR_WEIGHT_DECAY
    )
    transformed_device = tuple(value.to(device) for value in transformed)
    targets_device = standardized_targets.to(device)
    candidate_weights_device = candidate_weights.to(device)
    decision_weights_device = decision_weights.to(device)
    losses: list[float] = []
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        predictions = model(*transformed_device).squeeze(-1)
        smooth = torch.nn.functional.smooth_l1_loss(
            predictions,
            targets_device,
            beta=DECAR_SMOOTH_L1_BETA,
            reduction="none",
        )
        smooth_loss = (
            smooth * candidate_weights_device
        ).sum() / candidate_weights_device.sum()
        pair_loss = _pairwise_loss(predictions, targets_device, decision_weights_device)
        loss = smooth_loss + DECAR_PAIRWISE_WEIGHT * pair_loss
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("DECAR where optimization became non-finite")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise RuntimeError("DECAR where parameters became non-finite")
    return WhereFit(
        model=model,
        standardizer=standardizer,
        target_mean=float(target_mean_tensor),
        target_scale=float(target_scale_tensor),
        audit={
            "seed": seed,
            "epochs": epochs,
            "learning_rate": DECAR_LEARNING_RATE,
            "weight_decay": DECAR_WEIGHT_DECAY,
            "smooth_l1_beta": DECAR_SMOOTH_L1_BETA,
            "pairwise_weight": DECAR_PAIRWISE_WEIGHT,
            "decisions": len(source_ids),
            "sources": len(set(source_ids)),
            "embedding_dim": int(question.shape[-1]),
            "scalar_dim": int(scalars.shape[-1]),
            "fusion_dim": DECAR_FUSION_DIM,
            "first_loss": losses[0],
            "last_loss": losses[-1],
        },
    )


def predict_where(
    fit: WhereFit,
    question: Any,
    global_visual: Any,
    region: Any,
    scalars: Any,
    *,
    device: str,
) -> Any:
    torch = _require_torch()
    transformed = tuple(
        value.to(device)
        for value in fit.standardizer.transform(
            question, global_visual, region, scalars
        )
    )
    fit.model.eval()
    with torch.inference_mode():
        predictions = fit.model(*transformed).squeeze(-1).to(torch.float32).cpu()
    if not torch.isfinite(predictions).all():
        raise RuntimeError("DECAR where prediction became non-finite")
    return predictions


def select_where_actions(predictions: Any) -> tuple[Any, Any, Any]:
    torch = _require_torch()
    if predictions.ndim != 2 or predictions.shape[1] != len(DECAR_ACTION_IDS):
        raise ValueError("DECAR where prediction matrix shape changed")
    # torch.argmax returns the first maximum, implementing action-ID tie breaking.
    selected = torch.argmax(predictions, dim=1)
    ordered = torch.sort(predictions, dim=1, descending=True, stable=True).values
    gaps = predictions.gather(1, selected[:, None]).squeeze(1)
    margins = ordered[:, 0] - ordered[:, 1]
    return selected, gaps, margins


def _make_when_network(embedding_dim: int, *, binary: bool) -> Any:
    torch = _require_torch()
    nn = torch.nn

    class WhenNetwork(nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.question_projection = nn.Linear(embedding_dim, DECAR_PROJECTION_DIM)
            self.global_projection = nn.Linear(embedding_dim, DECAR_PROJECTION_DIM)
            self.region_projection = nn.Linear(embedding_dim, DECAR_PROJECTION_DIM)
            self.trunk = nn.Sequential(
                nn.Linear(DECAR_WHEN_DIM, DECAR_HIDDEN_DIMS[0]),
                nn.GELU(),
                nn.Linear(DECAR_HIDDEN_DIMS[0], DECAR_HIDDEN_DIMS[1]),
                nn.GELU(),
            )
            self.class_head = nn.Linear(DECAR_HIDDEN_DIMS[1], 2 if binary else 3)
            self.delta_head = nn.Linear(DECAR_HIDDEN_DIMS[1], 1)

        def forward(
            self,
            question: Any,
            global_visual: Any,
            region: Any,
            scalars: Any,
            gap_features: Any,
        ) -> tuple[Any, Any]:
            question_projected = self.question_projection(question)
            global_projected = self.global_projection(global_visual)
            region_projected = self.region_projection(region)
            fusion = torch.cat(
                (
                    question_projected,
                    global_projected,
                    region_projected,
                    question_projected * region_projected,
                    question_projected * global_projected,
                    global_projected * region_projected,
                    region_projected - global_projected,
                    scalars,
                    gap_features,
                ),
                dim=-1,
            )
            if fusion.shape[-1] != DECAR_WHEN_DIM:
                raise RuntimeError("DECAR when fusion dimension changed")
            hidden = self.trunk(fusion)
            return self.class_head(hidden), self.delta_head(hidden).squeeze(-1)

    return WhenNetwork()


def _selected_candidate(values: Any, selected: Any) -> Any:
    torch = _require_torch()
    if values.ndim < 2 or selected.shape != (values.shape[0],):
        raise ValueError("DECAR selected-candidate arrays are not aligned")
    index_shape = [values.shape[0], 1] + [1] * (values.ndim - 2)
    expanded_shape = [values.shape[0], 1, *values.shape[2:]]
    indices = selected.reshape(index_shape).expand(expanded_shape)
    return torch.gather(values, 1, indices).squeeze(1)


def fit_when(
    question: Any,
    global_visual: Any,
    selected_region: Any,
    selected_scalars: Any,
    predicted_gaps: Any,
    predicted_margins: Any,
    selected_deltas: Any,
    source_ids: Sequence[str],
    *,
    seed: int,
    device: str,
    binary: bool = False,
    epochs: int = DECAR_EPOCHS,
) -> WhenFit:
    torch = _require_torch()
    if (
        epochs <= 0
        or question.ndim != 2
        or global_visual.shape != question.shape
        or selected_region.shape != question.shape
        or selected_scalars.shape != (question.shape[0], len(DECAR_SCALAR_NAMES))
        or predicted_gaps.shape != (question.shape[0],)
        or predicted_margins.shape != (question.shape[0],)
        or selected_deltas.shape != (question.shape[0],)
        or len(source_ids) != question.shape[0]
    ):
        raise ValueError("DECAR when training arrays are not aligned")
    _set_deterministic(seed)
    decision_weights = source_balanced_decision_weights(source_ids)
    standardizer = fit_fusion_standardizer(
        question,
        global_visual,
        selected_region,
        selected_scalars,
        decision_weights,
    )
    transformed = standardizer.transform(
        question, global_visual, selected_region, selected_scalars
    )
    gap_features = torch.stack((predicted_gaps, predicted_margins), dim=1)
    gap_mean, gap_scale = _weighted_mean_scale(gap_features, decision_weights)
    standardized_gaps = (gap_features - gap_mean) / gap_scale

    if binary:
        labels = torch.where(
            selected_deltas > 0.0,
            torch.zeros_like(selected_deltas, dtype=torch.long),
            torch.ones_like(selected_deltas, dtype=torch.long),
        )
        expected_classes = {0, 1}
    else:
        labels = torch.full_like(selected_deltas, 1, dtype=torch.long)
        labels = torch.where(selected_deltas > 0.0, 0, labels)
        labels = torch.where(selected_deltas < 0.0, 2, labels)
        expected_classes = {0, 1, 2}
    if set(int(value) for value in labels.tolist()) != expected_classes:
        raise ValueError("DECAR when training requires every registered class")
    class_weights = torch.empty_like(decision_weights)
    for class_index in sorted(expected_classes):
        mask = labels == class_index
        mass = decision_weights[mask].sum()
        if not bool(torch.isfinite(mass)) or float(mass) <= 0.0:
            raise ValueError("DECAR when class mass is invalid")
        class_weights[mask] = decision_weights[mask] / mass
    class_weights = class_weights / class_weights.sum()

    rescue_mask = selected_deltas > 0.0
    harm_mask = selected_deltas < 0.0
    if not bool(rescue_mask.any()) or not bool(harm_mask.any()):
        raise ValueError("DECAR when magnitudes require rescue and harm rows")
    rescue_weights = decision_weights[rescue_mask]
    harm_weights = decision_weights[harm_mask]
    rescue_magnitude = float(
        (selected_deltas[rescue_mask] * rescue_weights).sum() / rescue_weights.sum()
    )
    harm_magnitude = float(
        (-selected_deltas[harm_mask] * harm_weights).sum() / harm_weights.sum()
    )
    if not (
        math.isfinite(rescue_magnitude)
        and rescue_magnitude > 0.0
        and math.isfinite(harm_magnitude)
        and harm_magnitude > 0.0
    ):
        raise RuntimeError("DECAR when magnitudes are invalid")

    model = _make_when_network(question.shape[-1], binary=binary).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=DECAR_LEARNING_RATE, weight_decay=DECAR_WEIGHT_DECAY
    )
    transformed_device = tuple(value.to(device) for value in transformed)
    gaps_device = standardized_gaps.to(device)
    labels_device = labels.to(device)
    deltas_device = selected_deltas.to(device)
    class_weights_device = class_weights.to(device)
    decision_weights_device = decision_weights.to(device)
    losses: list[float] = []
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits, predicted_delta = model(*transformed_device, gaps_device)
        cross_entropy = torch.nn.functional.cross_entropy(
            logits, labels_device, reduction="none"
        )
        class_loss = (cross_entropy * class_weights_device).sum()
        delta_rows = torch.nn.functional.smooth_l1_loss(
            predicted_delta,
            deltas_device,
            beta=DECAR_SMOOTH_L1_BETA,
            reduction="none",
        )
        delta_loss = (delta_rows * decision_weights_device).sum()
        loss = class_loss + delta_loss
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("DECAR when optimization became non-finite")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise RuntimeError("DECAR when parameters became non-finite")
    class_counts = {
        str(index): int((labels == index).sum()) for index in sorted(expected_classes)
    }
    return WhenFit(
        model=model,
        standardizer=standardizer,
        gap_mean=gap_mean,
        gap_scale=gap_scale,
        rescue_magnitude=rescue_magnitude,
        harm_magnitude=harm_magnitude,
        binary=binary,
        audit={
            "seed": seed,
            "epochs": epochs,
            "learning_rate": DECAR_LEARNING_RATE,
            "weight_decay": DECAR_WEIGHT_DECAY,
            "smooth_l1_beta": DECAR_SMOOTH_L1_BETA,
            "class_mass_normalized": True,
            "binary": binary,
            "decisions": len(source_ids),
            "sources": len(set(source_ids)),
            "class_counts": class_counts,
            "rescue_magnitude": rescue_magnitude,
            "harm_magnitude": harm_magnitude,
            "first_loss": losses[0],
            "last_loss": losses[-1],
        },
    )


def predict_when(
    fit: WhenFit,
    question: Any,
    global_visual: Any,
    selected_region: Any,
    selected_scalars: Any,
    predicted_gaps: Any,
    predicted_margins: Any,
    *,
    device: str,
) -> tuple[Any, Any]:
    torch = _require_torch()
    transformed = tuple(
        value.to(device)
        for value in fit.standardizer.transform(
            question, global_visual, selected_region, selected_scalars
        )
    )
    raw_gaps = torch.stack((predicted_gaps, predicted_margins), dim=1)
    gaps = ((raw_gaps - fit.gap_mean) / fit.gap_scale).to(device)
    fit.model.eval()
    with torch.inference_mode():
        logits, predicted_delta = fit.model(*transformed, gaps)
        probabilities = torch.softmax(logits, dim=1).to(torch.float32).cpu()
        predicted_delta = predicted_delta.to(torch.float32).cpu()
    if (
        not torch.isfinite(probabilities).all()
        or not torch.isfinite(predicted_delta).all()
    ):
        raise RuntimeError("DECAR when prediction became non-finite")
    return probabilities, predicted_delta


def score_when(
    fit: WhenFit, probabilities: Any, predicted_delta: Any
) -> tuple[Any, Any]:
    torch = _require_torch()
    expected_classes = 2 if fit.binary else 3
    if probabilities.ndim != 2 or probabilities.shape[1] != expected_classes:
        raise ValueError("DECAR when probability shape changed")
    if predicted_delta.shape != (probabilities.shape[0],):
        raise ValueError("DECAR when delta shape changed")
    rescue_probability = probabilities[:, 0]
    if fit.binary:
        class_delta = rescue_probability * fit.rescue_magnitude
        eligible_veto = torch.ones_like(rescue_probability, dtype=torch.bool)
    else:
        harm_probability = probabilities[:, 2]
        class_delta = (
            rescue_probability * fit.rescue_magnitude
            - harm_probability * fit.harm_magnitude
        )
        eligible_veto = harm_probability < rescue_probability
    scores = (
        0.5 * class_delta
        + 0.5 * torch.clamp(predicted_delta, min=-1.0, max=1.0)
        - DECAR_TOOL_COST
    )
    eligible = eligible_veto & (scores > 0.0)
    return scores, eligible


DECAR_VARIANT_INDEX = {
    "decar": 0,
    "task_value_only": 1,
    "loss_only": 2,
    "no_harm_head": 3,
}
DECAR_OUTER_REFIT_PSEUDO_INNER_FOLD = 4
DECAR_WHEN_PSEUDO_INNER_FOLD = 5


def decar_fit_seed(outer_fold: int, inner_fold: int, variant: str) -> int:
    if outer_fold not in range(5) or inner_fold not in range(6):
        raise ValueError("DECAR fit fold is outside the frozen range")
    if variant not in DECAR_VARIANT_INDEX:
        raise ValueError("DECAR fit variant is not registered")
    return (
        DECAR_SEED + 100 * outer_fold + 10 * inner_fold + DECAR_VARIANT_INDEX[variant]
    )


def _model_state_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().to("cpu").contiguous()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _take_rows(values: Any, indices: Sequence[int]) -> Any:
    torch = _require_torch()
    return values.index_select(0, torch.tensor(indices, dtype=torch.long))


def _take_source_ids(source_ids: Sequence[str], indices: Sequence[int]) -> list[str]:
    return [source_ids[index] for index in indices]


def _validate_nested_folds(
    dataset: DecarDataset,
    outer_fold_by_source: Mapping[str, int],
    inner_fold_by_outer_source: Mapping[tuple[int, str], int],
) -> None:
    sources = set(dataset.source_ids)
    if (
        dataset.decisions == 0
        or len(set(zip(dataset.state_ids, dataset.replicate_ids))) != dataset.decisions
        or len(dataset.replicate_ids) != dataset.decisions
        or len(dataset.image_ids) != dataset.decisions
        or set(outer_fold_by_source) != sources
        or set(outer_fold_by_source.values()) != set(range(5))
    ):
        raise ValueError("DECAR outer-fold coverage or decision identity changed")
    expected_inner_keys = {
        (outer_fold, source_id)
        for outer_fold in range(5)
        for source_id in sources
        if outer_fold_by_source[source_id] != outer_fold
    }
    if set(inner_fold_by_outer_source) != expected_inner_keys or any(
        value not in range(4) for value in inner_fold_by_outer_source.values()
    ):
        raise ValueError("DECAR inner-fold coverage changed")
    for outer_fold in range(5):
        values = {
            inner_fold_by_outer_source[(outer_fold, source_id)]
            for source_id in sources
            if outer_fold_by_source[source_id] != outer_fold
        }
        if values != set(range(4)):
            raise ValueError("DECAR inner-fold balance lost a registered fold")


def _fit_crossfitted_where(
    dataset: DecarDataset,
    targets: Any,
    outer_fold: int,
    train_indices: Sequence[int],
    test_indices: Sequence[int],
    inner_fold_by_outer_source: Mapping[tuple[int, str], int],
    *,
    variant: str,
    device: str,
    epochs: int,
) -> tuple[Any, Any, dict[str, Any]]:
    torch = _require_torch()
    crossfit_predictions = torch.full(
        (len(train_indices), len(DECAR_ACTION_IDS)), math.nan, dtype=torch.float32
    )
    train_position_by_index = {
        dataset_index: position for position, dataset_index in enumerate(train_indices)
    }
    inner_audits: list[dict[str, Any]] = []
    for inner_fold in range(4):
        inner_test_indices = [
            index
            for index in train_indices
            if inner_fold_by_outer_source[(outer_fold, dataset.source_ids[index])]
            == inner_fold
        ]
        inner_train_indices = [
            index
            for index in train_indices
            if inner_fold_by_outer_source[(outer_fold, dataset.source_ids[index])]
            != inner_fold
        ]
        fit_sources = set(_take_source_ids(dataset.source_ids, inner_train_indices))
        held_out_sources = set(_take_source_ids(dataset.source_ids, inner_test_indices))
        if (
            not inner_train_indices
            or not inner_test_indices
            or fit_sources & held_out_sources
        ):
            raise ValueError("DECAR inner where split is empty or leaks a source")
        fit = fit_where(
            _take_rows(dataset.question, inner_train_indices),
            _take_rows(dataset.global_visual, inner_train_indices),
            _take_rows(dataset.region, inner_train_indices),
            _take_rows(dataset.scalars, inner_train_indices),
            _take_rows(targets, inner_train_indices),
            _take_source_ids(dataset.source_ids, inner_train_indices),
            seed=decar_fit_seed(outer_fold, inner_fold, variant),
            device=device,
            epochs=epochs,
        )
        predictions = predict_where(
            fit,
            _take_rows(dataset.question, inner_test_indices),
            _take_rows(dataset.global_visual, inner_test_indices),
            _take_rows(dataset.region, inner_test_indices),
            _take_rows(dataset.scalars, inner_test_indices),
            device=device,
        )
        positions = [train_position_by_index[index] for index in inner_test_indices]
        crossfit_predictions[torch.tensor(positions, dtype=torch.long)] = predictions
        inner_audits.append(
            {
                "inner_fold": inner_fold,
                "fit": fit.audit,
                "model_state_sha256": _model_state_sha256(fit.model),
                "training_source_count": len(fit_sources),
                "held_out_source_count": len(held_out_sources),
                "source_overlap": 0,
            }
        )
        del fit
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    if not torch.isfinite(crossfit_predictions).all():
        raise RuntimeError("DECAR inner where predictions are incomplete")

    outer_fit_sources = set(_take_source_ids(dataset.source_ids, train_indices))
    outer_test_sources = set(_take_source_ids(dataset.source_ids, test_indices))
    if outer_fit_sources & outer_test_sources:
        raise ValueError("DECAR outer where split leaks a source")
    outer_fit = fit_where(
        _take_rows(dataset.question, train_indices),
        _take_rows(dataset.global_visual, train_indices),
        _take_rows(dataset.region, train_indices),
        _take_rows(dataset.scalars, train_indices),
        _take_rows(targets, train_indices),
        _take_source_ids(dataset.source_ids, train_indices),
        seed=decar_fit_seed(outer_fold, DECAR_OUTER_REFIT_PSEUDO_INNER_FOLD, variant),
        device=device,
        epochs=epochs,
    )
    outer_predictions = predict_where(
        outer_fit,
        _take_rows(dataset.question, test_indices),
        _take_rows(dataset.global_visual, test_indices),
        _take_rows(dataset.region, test_indices),
        _take_rows(dataset.scalars, test_indices),
        device=device,
    )
    audit = {
        "variant": variant,
        "inner_fits": inner_audits,
        "outer_refit": {
            "pseudo_inner_fold": DECAR_OUTER_REFIT_PSEUDO_INNER_FOLD,
            "fit": outer_fit.audit,
            "model_state_sha256": _model_state_sha256(outer_fit.model),
            "training_source_count": len(outer_fit_sources),
            "held_out_source_count": len(outer_test_sources),
            "source_overlap": 0,
        },
    }
    del outer_fit
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return crossfit_predictions, outer_predictions, audit


def _fit_outer_when(
    dataset: DecarDataset,
    train_indices: Sequence[int],
    test_indices: Sequence[int],
    crossfit_where_predictions: Any,
    outer_where_predictions: Any,
    outer_fold: int,
    *,
    variant: str,
    binary: bool,
    device: str,
    epochs: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = _require_torch()
    crossfit_selected, crossfit_gaps, crossfit_margins = select_where_actions(
        crossfit_where_predictions
    )
    outer_selected, outer_gaps, outer_margins = select_where_actions(
        outer_where_predictions
    )
    train_regions = _take_rows(dataset.region, train_indices)
    train_scalars = _take_rows(dataset.scalars, train_indices)
    train_deltas = _take_rows(dataset.task_deltas, train_indices)
    selected_train_regions = _selected_candidate(train_regions, crossfit_selected)
    selected_train_scalars = _selected_candidate(train_scalars, crossfit_selected)
    selected_train_deltas = _selected_candidate(train_deltas, crossfit_selected)
    fit = fit_when(
        _take_rows(dataset.question, train_indices),
        _take_rows(dataset.global_visual, train_indices),
        selected_train_regions,
        selected_train_scalars,
        crossfit_gaps,
        crossfit_margins,
        selected_train_deltas,
        _take_source_ids(dataset.source_ids, train_indices),
        seed=decar_fit_seed(outer_fold, DECAR_WHEN_PSEUDO_INNER_FOLD, variant),
        device=device,
        binary=binary,
        epochs=epochs,
    )
    test_regions = _take_rows(dataset.region, test_indices)
    test_scalars = _take_rows(dataset.scalars, test_indices)
    selected_test_regions = _selected_candidate(test_regions, outer_selected)
    selected_test_scalars = _selected_candidate(test_scalars, outer_selected)
    probabilities, predicted_delta = predict_when(
        fit,
        _take_rows(dataset.question, test_indices),
        _take_rows(dataset.global_visual, test_indices),
        selected_test_regions,
        selected_test_scalars,
        outer_gaps,
        outer_margins,
        device=device,
    )
    scores, eligible = score_when(fit, probabilities, predicted_delta)
    result = {
        "selected": outer_selected,
        "predicted_gap": outer_gaps,
        "predicted_margin": outer_margins,
        "probabilities": probabilities,
        "predicted_delta": predicted_delta,
        "score": scores,
        "eligible": eligible,
    }
    audit = {
        "variant": variant,
        "pseudo_inner_fold": DECAR_WHEN_PSEUDO_INNER_FOLD,
        "fit": fit.audit,
        "model_state_sha256": _model_state_sha256(fit.model),
        "crossfit_rows": len(train_indices),
        "crossfit_finite": bool(torch.isfinite(crossfit_where_predictions).all()),
    }
    del fit
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result, audit


def fit_nested_oof(
    dataset: DecarDataset,
    outer_fold_by_source: Mapping[str, int],
    inner_fold_by_outer_source: Mapping[tuple[int, str], int],
    *,
    device: str,
    epochs: int = DECAR_EPOCHS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit the four registered variants and return outcome-free OOF rows."""

    torch = _require_torch()
    if epochs <= 0:
        raise ValueError("DECAR nested OOF epochs must be positive")
    _validate_nested_folds(dataset, outer_fold_by_source, inner_fold_by_outer_source)
    prediction_rows: list[dict[str, Any] | None] = [None] * dataset.decisions
    fold_audits: list[dict[str, Any]] = []
    for outer_fold in range(5):
        train_indices = [
            index
            for index, source_id in enumerate(dataset.source_ids)
            if outer_fold_by_source[source_id] != outer_fold
        ]
        test_indices = [
            index
            for index, source_id in enumerate(dataset.source_ids)
            if outer_fold_by_source[source_id] == outer_fold
        ]
        if not train_indices or not test_indices:
            raise ValueError("DECAR outer split is empty")
        loss_crossfit, loss_outer, loss_where_audit = _fit_crossfitted_where(
            dataset,
            dataset.loss_gaps,
            outer_fold,
            train_indices,
            test_indices,
            inner_fold_by_outer_source,
            variant="decar",
            device=device,
            epochs=epochs,
        )
        task_crossfit, task_outer, task_where_audit = _fit_crossfitted_where(
            dataset,
            dataset.task_deltas,
            outer_fold,
            train_indices,
            test_indices,
            inner_fold_by_outer_source,
            variant="task_value_only",
            device=device,
            epochs=epochs,
        )
        decar_result, decar_when_audit = _fit_outer_when(
            dataset,
            train_indices,
            test_indices,
            loss_crossfit,
            loss_outer,
            outer_fold,
            variant="decar",
            binary=False,
            device=device,
            epochs=epochs,
        )
        no_harm_result, no_harm_when_audit = _fit_outer_when(
            dataset,
            train_indices,
            test_indices,
            loss_crossfit,
            loss_outer,
            outer_fold,
            variant="no_harm_head",
            binary=True,
            device=device,
            epochs=epochs,
        )
        task_result, task_when_audit = _fit_outer_when(
            dataset,
            train_indices,
            test_indices,
            task_crossfit,
            task_outer,
            outer_fold,
            variant="task_value_only",
            binary=False,
            device=device,
            epochs=epochs,
        )
        loss_selected, loss_gaps, loss_margins = select_where_actions(loss_outer)
        for test_position, dataset_index in enumerate(test_indices):
            variants: dict[str, dict[str, Any]] = {}
            for name, result in (
                ("decar", decar_result),
                ("task_value_only", task_result),
                ("no_harm_head", no_harm_result),
            ):
                probabilities = result["probabilities"][test_position]
                row = {
                    "selected_action_id": DECAR_ACTION_IDS[
                        int(result["selected"][test_position])
                    ],
                    "predicted_gap": float(result["predicted_gap"][test_position]),
                    "predicted_margin": float(
                        result["predicted_margin"][test_position]
                    ),
                    "rescue_probability": float(probabilities[0]),
                    "predicted_delta": float(result["predicted_delta"][test_position]),
                    "score": float(result["score"][test_position]),
                    "eligible": bool(result["eligible"][test_position]),
                }
                if name == "no_harm_head":
                    row["other_probability"] = float(probabilities[1])
                else:
                    row["neutral_probability"] = float(probabilities[1])
                    row["harm_probability"] = float(probabilities[2])
                variants[name] = row
            variants["loss_only"] = {
                "selected_action_id": DECAR_ACTION_IDS[
                    int(loss_selected[test_position])
                ],
                "predicted_gap": float(loss_gaps[test_position]),
                "predicted_margin": float(loss_margins[test_position]),
                "score": float(loss_gaps[test_position]),
                "eligible": True,
            }
            if not all(
                math.isfinite(float(value))
                for row in variants.values()
                for key, value in row.items()
                if key
                not in {
                    "selected_action_id",
                    "eligible",
                }
            ):
                raise RuntimeError("DECAR OOF prediction contains a non-finite value")
            prediction_rows[dataset_index] = {
                "schema": "infographicvqa_decar_oof_prediction_v1",
                "state_id": dataset.state_ids[dataset_index],
                "replicate_id": dataset.replicate_ids[dataset_index],
                "image_id": dataset.image_ids[dataset_index],
                "source_id": dataset.source_ids[dataset_index],
                "outer_fold": outer_fold,
                "variants": variants,
            }
        train_sources = set(_take_source_ids(dataset.source_ids, train_indices))
        test_sources = set(_take_source_ids(dataset.source_ids, test_indices))
        fold_audits.append(
            {
                "outer_fold": outer_fold,
                "training_rows": len(train_indices),
                "test_rows": len(test_indices),
                "training_sources": len(train_sources),
                "test_sources": len(test_sources),
                "source_overlap": len(train_sources & test_sources),
                "where": {
                    "decar_and_loss_only": loss_where_audit,
                    "task_value_only": task_where_audit,
                },
                "when": {
                    "decar": decar_when_audit,
                    "task_value_only": task_when_audit,
                    "no_harm_head": no_harm_when_audit,
                },
            }
        )
        del loss_crossfit, loss_outer, task_crossfit, task_outer
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    if any(row is None for row in prediction_rows):
        raise RuntimeError("DECAR OOF prediction coverage is incomplete")
    predictions = [row for row in prediction_rows if row is not None]
    payload = {
        "format_version": 1,
        "metadata": {
            "schema": "infographicvqa_decar_oof_predictions_v1",
            "outcomes_included": False,
            "decisions": dataset.decisions,
            "variants": list(DECAR_VARIANT_INDEX),
            "epochs": epochs,
            "device": device,
        },
        "predictions": predictions,
    }
    audit = {
        "schema": "infographicvqa_decar_nested_oof_audit_v1",
        "seed": DECAR_SEED,
        "seed_mapping": {
            "formula": "20260917 + 100*outer_fold + 10*inner_fold + variant_index",
            "variant_index": DECAR_VARIANT_INDEX,
            "outer_refit_pseudo_inner_fold": DECAR_OUTER_REFIT_PSEUDO_INNER_FOLD,
            "when_pseudo_inner_fold": DECAR_WHEN_PSEUDO_INNER_FOLD,
        },
        "folds": fold_audits,
        "prediction_rows": len(predictions),
        "prediction_outcomes_included": False,
    }
    return payload, audit
