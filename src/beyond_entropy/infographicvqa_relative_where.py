from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .infographicvqa_decar import (
    DECAR_ACTION_IDS,
    DecarDataset,
    FusionStandardizer,
    _require_torch,
    _take_rows,
    fit_fusion_standardizer,
    source_balanced_decision_weights,
)


RELATIVE_WHERE_SCHEMA = "infographicvqa_relative_where_oof_prediction_v1"
RELATIVE_WHERE_VARIANTS = (
    "relative_teacher_entropy",
    "absolute_teacher_entropy",
    "relative_teacher_uniform",
    "relative_task_entropy",
)
RELATIVE_WHERE_SEED = 20_260_923
RELATIVE_WHERE_PROJECTION_DIM = 64
RELATIVE_WHERE_HIDDEN_DIMS = (128, 32)
RELATIVE_WHERE_RELATIVE_DIM = 544
RELATIVE_WHERE_ABSOLUTE_DIM = 464
RELATIVE_WHERE_EPOCHS = 200
RELATIVE_WHERE_LEARNING_RATE = 0.001
RELATIVE_WHERE_WEIGHT_DECAY = 0.0001
RELATIVE_WHERE_PAIRWISE_WEIGHT = 0.5


@dataclass(frozen=True)
class RelativeWhereFit:
    model: Any
    standardizer: FusionStandardizer
    audit: Mapping[str, Any]


def relative_where_seed(outer_fold: int, variant: str) -> int:
    if outer_fold not in range(5) or variant not in RELATIVE_WHERE_VARIANTS:
        raise ValueError("relative-where seed coordinates are invalid")
    return (
        RELATIVE_WHERE_SEED + 100 * outer_fold + RELATIVE_WHERE_VARIANTS.index(variant)
    )


def _set_deterministic(seed: int) -> None:
    torch = _require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _average_rank_percentiles(values: Any) -> Any:
    torch = _require_torch()
    if values.ndim != 1 or values.numel() < 2 or not torch.isfinite(values).all():
        raise ValueError("relative-where entropy ranks are invalid")
    raw = [float(value) for value in values.detach().cpu().tolist()]
    ordered = sorted(range(len(raw)), key=lambda index: (raw[index], index))
    result = [math.nan] * len(raw)
    start = 0
    denominator = len(raw) - 1
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and raw[ordered[end]] == raw[ordered[start]]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        percentile = average_rank / denominator
        for position in range(start, end):
            result[ordered[position]] = percentile
        start = end
    tensor = torch.tensor(result, dtype=torch.float32)
    if (
        not torch.isfinite(tensor).all()
        or float(tensor.min()) < 0.0
        or float(tensor.max()) > 1.0
    ):
        raise RuntimeError("relative-where entropy percentiles are invalid")
    return tensor


def _ranking_weights(
    targets: Any,
    entropy_before: Any,
    source_ids: Sequence[str],
    *,
    entropy_weighted: bool,
) -> tuple[Any, float]:
    torch = _require_torch()
    if (
        targets.ndim != 2
        or targets.shape[1] != len(DECAR_ACTION_IDS)
        or entropy_before.shape != (targets.shape[0],)
        or len(source_ids) != targets.shape[0]
        or not torch.isfinite(targets).all()
    ):
        raise ValueError("relative-where ranking weights are not aligned")
    source_weights = source_balanced_decision_weights(source_ids)
    target_ranges = targets.max(dim=1).values - targets.min(dim=1).values
    positive = target_ranges > 0.0
    if not bool(positive.any()):
        raise ValueError("relative-where targets have no informative decisions")
    positive_source_mass = source_weights[positive].sum()
    range_scale = float(
        (target_ranges[positive] * source_weights[positive]).sum()
        / positive_source_mass
    )
    if not math.isfinite(range_scale) or range_scale <= 0.0:
        raise RuntimeError("relative-where range scale is invalid")
    informativeness = target_ranges / (target_ranges + range_scale)
    entropy_multiplier = (
        1.0 + 4.0 * _average_rank_percentiles(entropy_before)
        if entropy_weighted
        else torch.ones_like(entropy_before)
    )
    weights = source_weights * informativeness * entropy_multiplier
    if not torch.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        raise RuntimeError("relative-where final decision weights are invalid")
    return weights / weights.sum(), range_scale


def _target_distribution(targets: Any) -> Any:
    torch = _require_torch()
    maxima = targets.max(dim=1, keepdim=True).values
    mask = targets == maxima
    distribution = mask.to(torch.float32) / mask.sum(dim=1, keepdim=True)
    if not torch.isfinite(distribution).all() or not torch.allclose(
        distribution.sum(dim=1), torch.ones(targets.shape[0])
    ):
        raise RuntimeError("relative-where target distribution is invalid")
    return distribution


def _weighted_pairwise_loss(logits: Any, targets: Any, weights: Any) -> Any:
    torch = _require_torch()
    differences = targets[:, :, None] - targets[:, None, :]
    predicted = logits[:, :, None] - logits[:, None, :]
    upper = torch.triu(
        torch.ones(
            targets.shape[1], targets.shape[1], dtype=torch.bool, device=targets.device
        ),
        diagonal=1,
    )
    pair_weights = differences.abs() * upper[None, :, :]
    denominators = pair_weights.sum(dim=(1, 2))
    eligible = denominators > 0.0
    if not bool(eligible.any()):
        return logits.sum() * 0.0
    losses = torch.nn.functional.softplus(-torch.sign(differences) * predicted)
    per_decision = (losses * pair_weights).sum(dim=(1, 2)) / denominators.clamp(
        min=torch.finfo(logits.dtype).eps
    )
    eligible_weights = weights[eligible]
    return (per_decision[eligible] * eligible_weights).sum() / eligible_weights.sum()


def _make_relative_where_network(embedding_dim: int, *, relative: bool) -> Any:
    torch = _require_torch()
    nn = torch.nn

    class RelativeWhereNetwork(nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.question_projection = nn.Linear(
                embedding_dim, RELATIVE_WHERE_PROJECTION_DIM
            )
            self.global_projection = nn.Linear(
                embedding_dim, RELATIVE_WHERE_PROJECTION_DIM
            )
            self.region_projection = nn.Linear(
                embedding_dim, RELATIVE_WHERE_PROJECTION_DIM
            )
            input_dim = (
                RELATIVE_WHERE_RELATIVE_DIM if relative else RELATIVE_WHERE_ABSOLUTE_DIM
            )
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, RELATIVE_WHERE_HIDDEN_DIMS[0]),
                nn.GELU(),
                nn.Linear(RELATIVE_WHERE_HIDDEN_DIMS[0], RELATIVE_WHERE_HIDDEN_DIMS[1]),
                nn.GELU(),
                nn.Linear(RELATIVE_WHERE_HIDDEN_DIMS[1], 1),
            )

        def fusion(
            self, question: Any, global_visual: Any, region: Any, scalars: Any
        ) -> Any:
            if region.ndim != 3 or scalars.ndim != 3:
                raise ValueError("relative-where requires a candidate dimension")
            candidates = region.shape[1]
            question_projected = self.question_projection(question)[:, None, :].expand(
                -1, candidates, -1
            )
            global_projected = self.global_projection(global_visual)[:, None, :].expand(
                -1, candidates, -1
            )
            region_projected = self.region_projection(region)
            if relative:
                relative_region = region_projected - region_projected.mean(
                    dim=1, keepdim=True
                )
                relative_scalars = scalars - scalars.mean(dim=1, keepdim=True)
                fusion = torch.cat(
                    (
                        question_projected,
                        global_projected,
                        region_projected,
                        relative_region,
                        question_projected * region_projected,
                        question_projected * relative_region,
                        global_projected * region_projected,
                        global_projected * relative_region,
                        scalars,
                        relative_scalars,
                    ),
                    dim=-1,
                )
                expected = RELATIVE_WHERE_RELATIVE_DIM
            else:
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
                expected = RELATIVE_WHERE_ABSOLUTE_DIM
            if fusion.shape[-1] != expected:
                raise RuntimeError("relative-where fusion dimension changed")
            return fusion

        def forward(
            self, question: Any, global_visual: Any, region: Any, scalars: Any
        ) -> Any:
            return self.mlp(
                self.fusion(question, global_visual, region, scalars)
            ).squeeze(-1)

    return RelativeWhereNetwork()


def fit_relative_where(
    question: Any,
    global_visual: Any,
    region: Any,
    scalars: Any,
    targets: Any,
    entropy_before: Any,
    source_ids: Sequence[str],
    *,
    variant: str,
    seed: int,
    device: str,
    epochs: int = RELATIVE_WHERE_EPOCHS,
) -> RelativeWhereFit:
    torch = _require_torch()
    if variant not in RELATIVE_WHERE_VARIANTS or epochs <= 0:
        raise ValueError("relative-where fit configuration is invalid")
    _set_deterministic(seed)
    source_weights = source_balanced_decision_weights(source_ids)
    standardizer = fit_fusion_standardizer(
        question, global_visual, region, scalars, source_weights
    )
    transformed = standardizer.transform(question, global_visual, region, scalars)
    entropy_weighted = variant != "relative_teacher_uniform"
    decision_weights, range_scale = _ranking_weights(
        targets,
        entropy_before,
        source_ids,
        entropy_weighted=entropy_weighted,
    )
    target_distribution = _target_distribution(targets)
    relative = variant != "absolute_teacher_entropy"
    model = _make_relative_where_network(question.shape[-1], relative=relative).to(
        device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=RELATIVE_WHERE_LEARNING_RATE,
        weight_decay=RELATIVE_WHERE_WEIGHT_DECAY,
    )
    transformed_device = tuple(value.to(device) for value in transformed)
    targets_device = targets.to(device)
    distribution_device = target_distribution.to(device)
    weights_device = decision_weights.to(device)
    losses: list[float] = []
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(*transformed_device)
        listwise_rows = -(
            distribution_device * torch.nn.functional.log_softmax(logits, dim=1)
        ).sum(dim=1)
        listwise_loss = (listwise_rows * weights_device).sum()
        pairwise_loss = _weighted_pairwise_loss(logits, targets_device, weights_device)
        loss = listwise_loss + RELATIVE_WHERE_PAIRWISE_WEIGHT * pairwise_loss
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("relative-where optimization became non-finite")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise RuntimeError("relative-where parameters became non-finite")
    return RelativeWhereFit(
        model=model,
        standardizer=standardizer,
        audit={
            "variant": variant,
            "seed": seed,
            "epochs": epochs,
            "learning_rate": RELATIVE_WHERE_LEARNING_RATE,
            "weight_decay": RELATIVE_WHERE_WEIGHT_DECAY,
            "pairwise_weight": RELATIVE_WHERE_PAIRWISE_WEIGHT,
            "decisions": len(source_ids),
            "sources": len(set(source_ids)),
            "embedding_dim": int(question.shape[-1]),
            "scalar_dim": int(scalars.shape[-1]),
            "projection_dim": RELATIVE_WHERE_PROJECTION_DIM,
            "fusion_dim": (
                RELATIVE_WHERE_RELATIVE_DIM if relative else RELATIVE_WHERE_ABSOLUTE_DIM
            ),
            "entropy_weighted": entropy_weighted,
            "range_scale": range_scale,
            "first_loss": losses[0],
            "last_loss": losses[-1],
        },
    )


def predict_relative_where(
    fit: RelativeWhereFit,
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
        logits = fit.model(*transformed).to(torch.float32).cpu()
    if logits.shape[1] != len(DECAR_ACTION_IDS) or not torch.isfinite(logits).all():
        raise RuntimeError("relative-where prediction is invalid")
    return logits


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


def _validate_outer_folds(
    dataset: DecarDataset, outer_fold_by_source: Mapping[str, int]
) -> None:
    sources = set(dataset.source_ids)
    if (
        dataset.decisions == 0
        or set(outer_fold_by_source) != sources
        or set(outer_fold_by_source.values()) != set(range(5))
    ):
        raise ValueError("relative-where outer-fold coverage changed")


def fit_relative_where_oof(
    dataset: DecarDataset,
    outer_fold_by_source: Mapping[str, int],
    *,
    device: str,
    epochs: int = RELATIVE_WHERE_EPOCHS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fit the frozen four-variant family and emit source-held-out rows."""

    torch = _require_torch()
    _validate_outer_folds(dataset, outer_fold_by_source)
    prediction_rows: list[dict[str, Any]] = [
        {
            "schema": RELATIVE_WHERE_SCHEMA,
            "state_id": dataset.state_ids[index],
            "replicate_id": dataset.replicate_ids[index],
            "image_id": dataset.image_ids[index],
            "source_id": dataset.source_ids[index],
            "outer_fold": int(outer_fold_by_source[dataset.source_ids[index]]),
            "variants": {},
        }
        for index in range(dataset.decisions)
    ]
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
        train_sources = {dataset.source_ids[index] for index in train_indices}
        test_sources = {dataset.source_ids[index] for index in test_indices}
        if not train_indices or not test_indices or train_sources & test_sources:
            raise ValueError("relative-where outer split is empty or leaks a source")
        variant_audits: dict[str, Any] = {}
        for variant in RELATIVE_WHERE_VARIANTS:
            targets = (
                dataset.task_deltas
                if variant == "relative_task_entropy"
                else dataset.loss_gaps
            )
            fit = fit_relative_where(
                _take_rows(dataset.question, train_indices),
                _take_rows(dataset.global_visual, train_indices),
                _take_rows(dataset.region, train_indices),
                _take_rows(dataset.scalars, train_indices),
                _take_rows(targets, train_indices),
                _take_rows(dataset.entropy_before, train_indices),
                [dataset.source_ids[index] for index in train_indices],
                variant=variant,
                seed=relative_where_seed(outer_fold, variant),
                device=device,
                epochs=epochs,
            )
            logits = predict_relative_where(
                fit,
                _take_rows(dataset.question, test_indices),
                _take_rows(dataset.global_visual, test_indices),
                _take_rows(dataset.region, test_indices),
                _take_rows(dataset.scalars, test_indices),
                device=device,
            )
            probabilities = torch.softmax(logits, dim=1)
            selected = torch.argmax(logits, dim=1)
            ordered = torch.sort(logits, dim=1, descending=True).values
            margins = ordered[:, 0] - ordered[:, 1]
            for test_position, dataset_index in enumerate(test_indices):
                prediction_rows[dataset_index]["variants"][variant] = {
                    "action_scores": [
                        float(value) for value in logits[test_position].tolist()
                    ],
                    "action_probabilities": [
                        float(value) for value in probabilities[test_position].tolist()
                    ],
                    "selected_action_id": DECAR_ACTION_IDS[
                        int(selected[test_position])
                    ],
                    "predicted_margin": float(margins[test_position]),
                }
            variant_audits[variant] = {
                "fit": dict(fit.audit),
                "model_state_sha256": _model_state_sha256(fit.model),
            }
            del fit, logits, probabilities, selected, ordered, margins
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        fold_audits.append(
            {
                "outer_fold": outer_fold,
                "training_rows": len(train_indices),
                "test_rows": len(test_indices),
                "training_sources": len(train_sources),
                "test_sources": len(test_sources),
                "source_overlap": 0,
                "variants": variant_audits,
            }
        )
    if any(
        set(row["variants"]) != set(RELATIVE_WHERE_VARIANTS) for row in prediction_rows
    ):
        raise RuntimeError("relative-where OOF prediction coverage is incomplete")
    audit = {
        "schema": "infographicvqa_relative_where_nested_oof_audit_v1",
        "seed": RELATIVE_WHERE_SEED,
        "epochs": epochs,
        "prediction_rows": len(prediction_rows),
        "prediction_outcomes_included": False,
        "variants": list(RELATIVE_WHERE_VARIANTS),
        "fits": 5 * len(RELATIVE_WHERE_VARIANTS),
        "folds": fold_audits,
    }
    return prediction_rows, audit
