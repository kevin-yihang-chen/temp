from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any

import numpy as np  # type: ignore[import-not-found]
import torch  # type: ignore[import-not-found]
from torch import nn
from torch.nn import functional as F

from .action_value import (
    _action_features,
    _semantic_feature_index,
    _validate_domains,
)
from .oof_action_value import (
    _domain_source_balanced_weights,
    _fit_heads,
    _score_heads,
    _source_folds,
)
from .rescue_gate import DecisionKey
from .scaled_evaluation import bootstrap_source_balanced_metrics
from .schema import ActionRecord


JOINT_SEED = 20260904
JOINT_FOLDS = 5
JOINT_EPOCHS = 200
JOINT_LEARNING_RATE = 0.003
JOINT_WEIGHT_DECAY = 0.0001
JOINT_LOSS_WEIGHT = 0.5
JOINT_HIDDEN_DIMS = (32, 16)
JOINT_BOOTSTRAP_RESAMPLES = 20000
JOINT_BOOTSTRAP_CONFIDENCE = 0.95
JOINT_VARIANTS = ("task_only", "loss_only", "joint")


class _JointNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, int]) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.GELU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.GELU(),
        )
        self.rescue_head = nn.Linear(hidden_dims[1], 1)
        self.harm_head = nn.Linear(hidden_dims[1], 1)
        self.loss_gap_head = nn.Linear(hidden_dims[1], 1)

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.trunk(inputs)
        return (
            self.rescue_head(hidden).squeeze(-1),
            self.harm_head(hidden).squeeze(-1),
            self.loss_gap_head(hidden).squeeze(-1),
        )


def _weighted_center_scale(
    values: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if values.ndim not in (1, 2) or weights.ndim != 1:
        raise ValueError("weighted standardization requires one- or two-dimensional data")
    if values.shape[0] != weights.shape[0] or values.shape[0] == 0:
        raise ValueError("weighted standardization inputs are not aligned")
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError("weighted standardization inputs must be finite")
    if np.any(weights <= 0.0):
        raise ValueError("source-balanced weights must be positive")
    normalized = weights / weights.sum()
    if values.ndim == 1:
        center = np.asarray(np.sum(values * normalized), dtype=np.float64)
        variance = np.asarray(
            np.sum(np.square(values - center) * normalized), dtype=np.float64
        )
    else:
        center = np.sum(values * normalized[:, None], axis=0)
        variance = np.sum(
            np.square(values - center[None, :]) * normalized[:, None], axis=0
        )
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale = np.where(scale > 1e-12, scale, 1.0)
    if not np.isfinite(center).all() or not np.isfinite(scale).all():
        raise ValueError("weighted standardization produced non-finite values")
    return center, scale


def _balanced_binary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    positive = targets > 0.5
    negative = ~positive
    if not bool(positive.any()) or not bool(negative.any()):
        raise ValueError("joint binary heads require both target classes")
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    positive_loss = torch.sum(losses[positive] * weights[positive]) / torch.sum(
        weights[positive]
    )
    negative_loss = torch.sum(losses[negative] * weights[negative]) / torch.sum(
        weights[negative]
    )
    return 0.5 * (positive_loss + negative_loss)


def _serialize_network(model: _JointNetwork) -> dict[str, Any]:
    return {
        name: tensor.detach().cpu().tolist()
        for name, tensor in sorted(model.state_dict().items())
    }


def _fit_variant(
    features: np.ndarray,
    rescue: np.ndarray,
    harm: np.ndarray,
    loss_gap: np.ndarray,
    weights: np.ndarray,
    *,
    variant: str,
    seed: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    hidden_dims: tuple[int, int],
    loss_weight: float,
    device: str,
) -> tuple[dict[str, Any], _JointNetwork]:
    if variant not in JOINT_VARIANTS:
        raise ValueError(f"unsupported joint auxiliary variant: {variant}")
    if epochs < 1 or learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("invalid joint auxiliary optimizer configuration")
    if loss_weight < 0.0 or min(hidden_dims) < 1:
        raise ValueError("invalid joint auxiliary architecture")
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("joint auxiliary features must be a non-empty matrix")
    n_rows = features.shape[0]
    for values in (rescue, harm, loss_gap, weights):
        if values.shape != (n_rows,):
            raise ValueError("joint auxiliary targets and weights are not aligned")
    required_finite = [features, weights]
    if variant in {"task_only", "joint"}:
        required_finite.extend((rescue, harm))
    if variant in {"loss_only", "joint"}:
        required_finite.append(loss_gap)
    if not all(np.isfinite(values).all() for values in required_finite):
        raise ValueError("joint auxiliary training arrays must be finite")

    feature_center, feature_scale = _weighted_center_scale(features, weights)
    if variant in {"loss_only", "joint"}:
        loss_center, loss_scale = _weighted_center_scale(loss_gap, weights)
        standardized_loss = (loss_gap - float(loss_center)) / float(loss_scale)
    else:
        # The task-only control must not consume the teacher target, including
        # through preprocessing statistics that do not affect gradients.
        loss_center = np.asarray(0.0, dtype=np.float64)
        loss_scale = np.asarray(1.0, dtype=np.float64)
        standardized_loss = np.zeros(n_rows, dtype=np.float64)
    standardized_features = (features - feature_center[None, :]) / feature_scale[
        None, :
    ]

    torch.manual_seed(seed)
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA joint training was requested but is unavailable")
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    model = _JointNetwork(features.shape[1], hidden_dims).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        foreach=False,
        fused=False,
    )
    tensors: dict[str, torch.Tensor] = {
        "features": torch.as_tensor(
            standardized_features, dtype=torch.float32, device=device
        ),
        "weights": torch.as_tensor(weights, dtype=torch.float32, device=device),
    }
    if variant in {"task_only", "joint"}:
        tensors["rescue"] = torch.as_tensor(
            rescue, dtype=torch.float32, device=device
        )
        tensors["harm"] = torch.as_tensor(harm, dtype=torch.float32, device=device)
    if variant in {"loss_only", "joint"}:
        tensors["loss_gap"] = torch.as_tensor(
            standardized_loss, dtype=torch.float32, device=device
        )
    final_terms: dict[str, float] = {}
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        rescue_logits, harm_logits, predicted_loss = model(tensors["features"])
        terms: dict[str, torch.Tensor] = {}
        if variant in {"task_only", "joint"}:
            terms["rescue"] = _balanced_binary_loss(
                rescue_logits, tensors["rescue"], tensors["weights"]
            )
            terms["harm"] = _balanced_binary_loss(
                harm_logits, tensors["harm"], tensors["weights"]
            )
        if variant in {"loss_only", "joint"}:
            point_losses = F.smooth_l1_loss(
                predicted_loss, tensors["loss_gap"], beta=1.0, reduction="none"
            )
            terms["loss_gap"] = torch.sum(
                point_losses * tensors["weights"]
            ) / torch.sum(tensors["weights"])
        objective = sum(
            value * (loss_weight if name == "loss_gap" and variant == "joint" else 1.0)
            for name, value in terms.items()
        )
        if not bool(torch.isfinite(objective)):
            raise RuntimeError("joint auxiliary objective became non-finite")
        objective.backward()
        optimizer.step()
        final_terms = {
            name: float(value.detach().cpu()) for name, value in terms.items()
        }
        final_terms["objective"] = float(objective.detach().cpu())

    payload = {
        "variant": variant,
        "seed": seed,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "loss_gap_weight": loss_weight if variant == "joint" else 0.0,
        "input_dim": int(features.shape[1]),
        "hidden_dims": list(hidden_dims),
        "feature_center": [float(value) for value in feature_center.tolist()],
        "feature_scale": [float(value) for value in feature_scale.tolist()],
        "loss_gap_center": float(loss_center),
        "loss_gap_scale": float(loss_scale),
        "final_training_loss": final_terms,
        "state_dict": _serialize_network(model),
    }
    return payload, model


def _predict_variant(
    model: _JointNetwork,
    payload: Mapping[str, Any],
    features: np.ndarray,
    *,
    device: str,
) -> np.ndarray:
    center = np.asarray(payload["feature_center"], dtype=np.float64)
    scale = np.asarray(payload["feature_scale"], dtype=np.float64)
    if features.ndim != 2 or features.shape[1:] != center.shape:
        raise ValueError("joint auxiliary prediction feature dimension changed")
    inputs = torch.as_tensor(
        (features - center[None, :]) / scale[None, :],
        dtype=torch.float32,
        device=device,
    )
    model.eval()
    with torch.no_grad():
        rescue_logits, harm_logits, predicted_loss = model(inputs)
        if payload["variant"] == "loss_only":
            scores = predicted_loss
        else:
            scores = torch.sigmoid(rescue_logits) - torch.sigmoid(harm_logits)
    result = scores.detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(result).all():
        raise RuntimeError("joint auxiliary prediction produced non-finite scores")
    return result


def _nll_index(
    records_by_domain: Mapping[str, Sequence[ActionRecord]],
    nll_rows_by_domain: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[tuple[str, str, str], float], dict[str, list[str]]]:
    if set(records_by_domain) != set(nll_rows_by_domain):
        raise ValueError("answer-NLL domains must exactly match rollout domains")
    expected: dict[tuple[str, str, str], ActionRecord] = {}
    for records in records_by_domain.values():
        for record in records:
            key = (record.state_id, record.replicate_id, record.action_id)
            if key in expected:
                raise ValueError("rollout action identities are not unique")
            expected[key] = record
    result: dict[tuple[str, str, str], float] = {}
    configs_by_domain: dict[str, list[str]] = {}
    for domain, rows in nll_rows_by_domain.items():
        configs: set[str] = set()
        config_by_decision: dict[DecisionKey, str] = {}
        for row in rows:
            key = (
                str(row.get("state_id", "")),
                str(row.get("replicate_id", "")),
                str(row.get("action_id", "")),
            )
            if not all(key) or key in result or key not in expected:
                raise ValueError("answer-NLL action identities are invalid")
            record = expected[key]
            if (
                str(row.get("source_id", "")) != record.source_id
                or str(row.get("action_type", "")) != record.action_type
                or float(row.get("correct_before", math.nan))
                != record.correct_before
                or float(row.get("correct_after", math.nan)) != record.correct_after
            ):
                raise ValueError("answer-NLL row differs from its rollout action")
            value = float(row.get("answer_mean_nll", math.nan))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("answer-NLL values must be finite and non-negative")
            config = str(row.get("config_sha256", ""))
            if not config:
                raise ValueError("answer-NLL row is missing its measurement config")
            decision = key[:2]
            prior_config = config_by_decision.setdefault(decision, config)
            if prior_config != config:
                raise ValueError(
                    "one decision mixes answer-NLL configurations, so its loss "
                    "gaps are not comparable"
                )
            configs.add(config)
            result[key] = value
        if not configs:
            raise ValueError(f"domain {domain!r} has no answer-NLL configuration")
        configs_by_domain[domain] = sorted(configs)
    if set(result) != set(expected):
        raise ValueError("answer-NLL rows do not exactly cover rollout actions")
    return result, configs_by_domain


def _source_means(
    values: Mapping[DecisionKey, float], source_by_key: Mapping[DecisionKey, str]
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for key, value in values.items():
        grouped.setdefault(source_by_key[key], []).append(float(value))
    return {source: mean(items) for source, items in grouped.items()}


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0.0 else None


def _evaluate_proposals(
    *,
    actions_by_method: Mapping[str, Mapping[DecisionKey, str] | None],
    baselines: Mapping[DecisionKey, ActionRecord],
    zooms: Mapping[DecisionKey, Sequence[ActionRecord]],
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    source_by_key = {key: baseline.source_id for key, baseline in baselines.items()}
    helpful_state = {
        key: float(any(action.delta_success > 0.0 for action in zooms[key]))
        for key in baselines
    }
    method_values: dict[str, dict[str, dict[DecisionKey, float]]] = {}
    for method, actions in actions_by_method.items():
        gain: dict[DecisionKey, float] = {}
        harm: dict[DecisionKey, float] = {}
        helpful: dict[DecisionKey, float] = {}
        for key in baselines:
            candidates = sorted(zooms[key], key=lambda action: action.action_id)
            if method == "random_exact":
                gains = [action.delta_success for action in candidates]
                gain[key] = mean(gains)
                harm[key] = mean(max(-value, 0.0) for value in gains)
                helpful[key] = mean(float(value > 0.0) for value in gains)
                continue
            if actions is None:
                raise ValueError(f"proposal actions are missing for {method!r}")
            matches = [action for action in candidates if action.action_id == actions[key]]
            if len(matches) != 1:
                raise ValueError(f"proposal {method!r} is invalid for {key!r}")
            selected = matches[0]
            gain[key] = selected.delta_success
            harm[key] = max(-selected.delta_success, 0.0)
            helpful[key] = float(selected.delta_success > 0.0)
        method_values[method] = {"gain": gain, "induced_harm": harm, "helpful": helpful}

    source_metrics: dict[str, dict[str, float]] = {
        source: {} for source in sorted(set(source_by_key.values()))
    }
    source_points: dict[str, dict[str, float | None]] = {}
    question_points: dict[str, dict[str, float | None]] = {}
    source_helpful_state = _source_means(helpful_state, source_by_key)
    helpful_state_source_mass = mean(source_helpful_state.values())
    helpful_state_question_mass = mean(helpful_state.values())
    for method, values in method_values.items():
        per_source = {
            name: _source_means(metric, source_by_key)
            for name, metric in values.items()
        }
        for source in source_metrics:
            for name in values:
                source_metrics[source][f"{method}_{name}"] = per_source[name][source]
        source_points[method] = {
            "gain": mean(per_source["gain"].values()),
            "induced_harm": mean(per_source["induced_harm"].values()),
            "helpful_selection_mass": mean(per_source["helpful"].values()),
            "helpful_state_mass": helpful_state_source_mass,
            "helpful_state_recovery": _safe_ratio(
                mean(per_source["helpful"].values()), helpful_state_source_mass
            ),
        }
        question_points[method] = {
            "gain": mean(values["gain"].values()),
            "induced_harm": mean(values["induced_harm"].values()),
            "helpful_selection_mass": mean(values["helpful"].values()),
            "helpful_state_mass": helpful_state_question_mass,
            "helpful_state_recovery": _safe_ratio(
                mean(values["helpful"].values()), helpful_state_question_mass
            ),
        }

    comparisons = {
        "joint_minus_task_only_gain": ("joint", "task_only"),
        "joint_minus_factorized_gain": ("joint", "factorized"),
        "joint_minus_loss_only_gain": ("joint", "loss_only"),
    }
    for name, (left, right) in comparisons.items():
        for source in source_metrics:
            source_metrics[source][name] = (
                source_metrics[source][f"{left}_gain"]
                - source_metrics[source][f"{right}_gain"]
            )
    bootstrap = bootstrap_source_balanced_metrics(
        source_metrics,
        n_resamples=bootstrap_resamples,
        confidence_level=JOINT_BOOTSTRAP_CONFIDENCE,
        seed=bootstrap_seed,
    )
    comparison_points = {
        name: bootstrap["metrics"][name] for name in comparisons
    }
    return {
        "source_balanced": source_points,
        "question_balanced": question_points,
        "pairwise_source_bootstrap": bootstrap,
        "primary_comparisons": comparison_points,
    }


def fit_joint_auxiliary_action_proposer(
    records_by_domain: Mapping[str, Sequence[ActionRecord]],
    nll_rows_by_domain: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    semantic_decisions_by_domain: Mapping[
        str, Mapping[DecisionKey, Mapping[str, Any]]
    ],
    feature_mode: str = "hybrid-context-semantic",
    n_folds: int = JOINT_FOLDS,
    seed: int = JOINT_SEED,
    epochs: int = JOINT_EPOCHS,
    learning_rate: float = JOINT_LEARNING_RATE,
    weight_decay: float = JOINT_WEIGHT_DECAY,
    loss_weight: float = JOINT_LOSS_WEIGHT,
    hidden_dims: tuple[int, int] = JOINT_HIDDEN_DIMS,
    bootstrap_resamples: int = JOINT_BOOTSTRAP_RESAMPLES,
    device: str = "cpu",
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Fit frozen shared-trunk proposal ablations using development sources only."""

    if feature_mode != "hybrid-context-semantic":
        raise ValueError("the frozen joint proposer requires hybrid-context-semantic")
    domain_by_key, baselines, zooms = _validate_domains(records_by_domain)
    semantic_by_key = _semantic_feature_index(
        feature_mode=feature_mode,
        records_by_domain=records_by_domain,
        domain_by_key=domain_by_key,
        semantic_decisions_by_domain=semantic_decisions_by_domain,
    )
    nll_by_action, nll_configs_by_domain = _nll_index(
        records_by_domain, nll_rows_by_domain
    )
    fold_by_key, fold_source_counts = _source_folds(
        domain_by_key, baselines, n_folds=n_folds, seed=seed
    )
    keys = sorted(baselines)
    action_rows: list[tuple[DecisionKey, ActionRecord]] = [
        (key, action)
        for key in keys
        for action in sorted(zooms[key], key=lambda item: item.action_id)
    ]
    features = np.asarray(
        [
            _action_features(
                baselines[key],
                action,
                feature_mode=feature_mode,
                semantic_decision=semantic_by_key.get(key),
            )
            for key, action in action_rows
        ],
        dtype=np.float64,
    )
    if features.shape != (4 * len(keys), 46):
        raise ValueError("frozen joint action feature inventory changed")
    rescue = np.asarray(
        [float(action.delta_success > 0.0) for _, action in action_rows],
        dtype=np.float64,
    )
    harm = np.asarray(
        [float(action.delta_success < 0.0) for _, action in action_rows],
        dtype=np.float64,
    )
    loss_gap = np.asarray(
        [
            nll_by_action[(key[0], key[1], "answer-now")]
            - nll_by_action[(key[0], key[1], action.action_id)]
            for key, action in action_rows
        ],
        dtype=np.float64,
    )
    domains = [domain_by_key[key] for key, _ in action_rows]
    sources = [baselines[key].source_id for key, _ in action_rows]
    weights = np.asarray(
        _domain_source_balanced_weights(domains, sources), dtype=np.float64
    )

    row_indices_by_fold = {
        fold: np.asarray(
            [
                index
                for index, (key, _) in enumerate(action_rows)
                if fold_by_key[key] == fold
            ],
            dtype=np.int64,
        )
        for fold in range(n_folds)
    }
    scores_by_variant = {
        variant: np.full(len(action_rows), np.nan, dtype=np.float64)
        for variant in JOINT_VARIANTS
    }
    fold_training: dict[str, list[dict[str, Any]]] = {
        variant: [] for variant in JOINT_VARIANTS
    }
    for fold in range(n_folds):
        test_indices = row_indices_by_fold[fold]
        train_indices = np.concatenate(
            [row_indices_by_fold[other] for other in range(n_folds) if other != fold]
        )
        for variant in JOINT_VARIANTS:
            payload, model = _fit_variant(
                features[train_indices],
                rescue[train_indices],
                harm[train_indices],
                loss_gap[train_indices],
                weights[train_indices],
                variant=variant,
                seed=seed + fold,
                epochs=epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                hidden_dims=hidden_dims,
                loss_weight=loss_weight,
                device=device,
            )
            scores_by_variant[variant][test_indices] = _predict_variant(
                model, payload, features[test_indices], device=device
            )
            fold_training[variant].append(
                {
                    "fold": fold,
                    "train_sources": len(
                        {sources[index] for index in train_indices.tolist()}
                    ),
                    "test_sources": len(
                        {sources[index] for index in test_indices.tolist()}
                    ),
                    "train_rows": int(len(train_indices)),
                    "test_rows": int(len(test_indices)),
                    "final_training_loss": payload["final_training_loss"],
                }
            )
    if any(not np.isfinite(values).all() for values in scores_by_variant.values()):
        raise RuntimeError("OOF joint auxiliary predictions are incomplete")

    actions_by_variant: dict[str, dict[DecisionKey, str]] = {
        variant: {} for variant in JOINT_VARIANTS
    }
    score_by_variant_key: dict[str, dict[DecisionKey, float]] = {
        variant: {} for variant in JOINT_VARIANTS
    }
    row_index_by_key: dict[DecisionKey, list[int]] = {}
    for index, (key, _) in enumerate(action_rows):
        row_index_by_key.setdefault(key, []).append(index)
    for variant in JOINT_VARIANTS:
        for key in keys:
            selected_index = min(
                row_index_by_key[key],
                key=lambda index: (
                    -float(scores_by_variant[variant][index]),
                    action_rows[index][1].action_id,
                ),
            )
            actions_by_variant[variant][key] = action_rows[selected_index][1].action_id
            score_by_variant_key[variant][key] = float(
                scores_by_variant[variant][selected_index]
            )

    incumbent_folds, _ = _source_folds(
        domain_by_key, baselines, n_folds=5, seed=20260829
    )
    factorized_actions: dict[DecisionKey, str] = {}
    for fold in range(5):
        train_keys = [key for key in keys if incumbent_folds[key] != fold]
        test_keys = [key for key in keys if incumbent_folds[key] == fold]
        heads = _fit_heads(
            train_keys,
            alpha=1.0,
            seed=20260829 + fold,
            feature_mode=feature_mode,
            baselines=baselines,
            zooms=zooms,
            domain_by_key=domain_by_key,
            semantic_by_key=semantic_by_key,
        )
        _, actions = _score_heads(
            heads,
            test_keys,
            lambda_cost=0.05,
            feature_mode=feature_mode,
            baselines=baselines,
            zooms=zooms,
            semantic_by_key=semantic_by_key,
        )
        factorized_actions.update(actions)
    if set(factorized_actions) != set(keys):
        raise RuntimeError("frozen factorized OOF proposer did not reproduce")

    entropy_actions = {
        key: min(
            zooms[key], key=lambda action: (action.entropy_after, action.action_id)
        ).action_id
        for key in keys
    }
    fixed_actions = {
        key: min(zooms[key], key=lambda action: action.action_id).action_id
        for key in keys
    }
    oracle_actions = {
        key: min(
            zooms[key], key=lambda action: (-action.delta_success, action.action_id)
        ).action_id
        for key in keys
    }
    evaluated = _evaluate_proposals(
        actions_by_method={
            **actions_by_variant,
            "factorized": factorized_actions,
            "entropy": entropy_actions,
            "fixed": fixed_actions,
            "random_exact": None,
            "oracle": oracle_actions,
        },
        baselines=baselines,
        zooms=zooms,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=seed,
    )
    source_points = evaluated["source_balanced"]
    joint_task_interval = evaluated["primary_comparisons"][
        "joint_minus_task_only_gain"
    ]
    pass_rule = {
        "joint_minus_task_only_gain_ci_low_positive": float(
            joint_task_interval["ci_low"]
        )
        > 0.0,
        "joint_gain_above_factorized": float(source_points["joint"]["gain"])
        > float(source_points["factorized"]["gain"]),
        "joint_recovery_above_task_only_and_factorized": float(
            source_points["joint"]["helpful_state_recovery"]
        )
        > max(
            float(source_points["task_only"]["helpful_state_recovery"]),
            float(source_points["factorized"]["helpful_state_recovery"]),
        ),
        "joint_harm_no_greater_than_task_only_and_factorized": float(
            source_points["joint"]["induced_harm"]
        )
        <= min(
            float(source_points["task_only"]["induced_harm"]),
            float(source_points["factorized"]["induced_harm"]),
        ),
    }

    full_models: dict[str, Any] = {}
    for variant in JOINT_VARIANTS:
        payload, _ = _fit_variant(
            features,
            rescue,
            harm,
            loss_gap,
            weights,
            variant=variant,
            seed=seed + n_folds,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            hidden_dims=hidden_dims,
            loss_weight=loss_weight,
            device=device,
        )
        full_models[variant] = payload

    report = {
        "scientific_status": (
            "non-ScreenQA development-only source-held-out proposal experiment; "
            "protected ScreenQA roles excluded"
        ),
        "decision": (
            "joint_auxiliary_proposer_advanced"
            if all(pass_rule.values())
            else "joint_auxiliary_proposer_not_advanced"
        ),
        "pass_rule": pass_rule,
        "feature_mode": feature_mode,
        "feature_count": int(features.shape[1]),
        "seed": seed,
        "n_folds": n_folds,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "loss_gap_weight": loss_weight,
        "hidden_dims": list(hidden_dims),
        "n_domains": len(set(domain_by_key.values())),
        "n_sources": len(set(sources)),
        "n_decisions": len(keys),
        "n_zoom_rows": len(action_rows),
        "target_counts": {
            "positive_gain": int(rescue.sum()),
            "negative_gain": int(harm.sum()),
            "neutral_gain": int(len(action_rows) - rescue.sum() - harm.sum()),
        },
        "fold_source_counts": fold_source_counts,
        "fold_training": fold_training,
        "nll_configs_by_domain": nll_configs_by_domain,
        "oof": evaluated,
        "protected_role_inputs_used": False,
        "screenqa_inputs_used": False,
        "docvqa_calibration_formal_reserve_inputs_used": False,
        "target_answer_available_at_inference": False,
    }
    model_payload = {
        "model_type": "joint_auxiliary_action_proposer_v1",
        "scientific_status": "full non-ScreenQA development refits after OOF",
        "feature_mode": feature_mode,
        "feature_count": int(features.shape[1]),
        "decision_rule": {
            "task_and_joint": "sigmoid(rescue)-sigmoid(harm); max with action-id tie break",
            "loss_only": "predicted standardized answer-loss gap; max with action-id tie break",
        },
        "seed": seed,
        "n_folds": n_folds,
        "variants": full_models,
        "screenqa_inputs_used": False,
        "target_answer_available_at_inference": False,
    }
    prediction_rows = [
        {
            "state_id": key[0],
            "replicate_id": key[1],
            "source_id": baselines[key].source_id,
            **{
                f"{variant}_action_id": actions_by_variant[variant][key]
                for variant in JOINT_VARIANTS
            },
            **{
                f"{variant}_score": score_by_variant_key[variant][key]
                for variant in JOINT_VARIANTS
            },
            "factorized_action_id": factorized_actions[key],
            "entropy_action_id": entropy_actions[key],
            "fixed_action_id": fixed_actions[key],
        }
        for key in keys
    ]
    return report, model_payload, prediction_rows
