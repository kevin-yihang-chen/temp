from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import replace
from statistics import mean
from typing import Any, Mapping, Sequence

from .dataset import group_by_decision
from .metrics import bootstrap_policy_evaluation, evaluate_policy
from .rescue_gate import (
    DecisionKey,
    PrecomputedActionGatePolicy,
    compact_action_features,
    pre_action_context_features,
)
from .schema import ActionRecord, BBox


def normalized_gate_question(question: str) -> str:
    """Remove known task wrappers while retaining the actual visible question.

    This is deliberately a surface normalization, not a benchmark-ID feature.
    It prevents long evaluation instructions from becoming a domain shortcut.
    """

    normalized = " ".join(str(question).strip().split())
    question_markers = list(re.finditer(r"\bquestion\s*:\s*", normalized, re.I))
    if question_markers:
        normalized = normalized[question_markers[-1].end() :].strip()
    normalized = re.sub(r"\s+answer\s*:\s*$", "", normalized, flags=re.I).strip()
    if not normalized:
        raise ValueError("normalized gate-visible question must be non-empty")
    return normalized


def _bbox_geometry(bbox: BBox) -> list[float]:
    center_x = (bbox.x1 + bbox.x2) / 2.0
    center_y = (bbox.y1 + bbox.y2) / 2.0
    return [
        bbox.x1,
        bbox.y1,
        bbox.x2,
        bbox.y2,
        bbox.width,
        bbox.height,
        bbox.area,
        center_x,
        center_y,
    ]


def context_geometry_action_features(
    baseline: ActionRecord,
    action: ActionRecord,
) -> list[float]:
    """Build domain-agnostic pre-action state-by-crop geometry features."""

    if baseline.action_type != "ANSWER":
        raise ValueError("action-value baseline must be ANSWER")
    if action.action_type != "ZOOM" or action.candidate_bbox is None:
        raise ValueError("action-value candidate must be a bounded ZOOM")
    if (baseline.state_id, baseline.replicate_id) != (
        action.state_id,
        action.replicate_id,
    ):
        raise ValueError("baseline and action must belong to one decision")
    normalized_baseline = replace(
        baseline,
        question=normalized_gate_question(baseline.question),
    )
    context = pre_action_context_features(normalized_baseline)
    geometry = _bbox_geometry(action.candidate_bbox)
    grid_size = float(action.pre_action_features.get("ug_grid_size", 0.0))
    if not math.isfinite(grid_size) or grid_size < 0.0:
        raise ValueError("ug_grid_size must be finite and non-negative")
    action_surface = [*geometry, math.log1p(grid_size)]
    interactions = [
        context_value * action_value
        for context_value in context
        for action_value in action_surface
    ]
    return [*context, *action_surface, *interactions]


def semantic_context_action_features(
    baseline: ActionRecord,
    action: ActionRecord,
    decision: Mapping[str, Any],
) -> list[float]:
    """Fuse normalized state context with frozen pre-action ROI similarities."""

    if action.action_type != "ZOOM":
        raise ValueError("semantic action-value candidate must be ZOOM")
    action_ids = [str(value) for value in decision.get("action_ids", [])]
    try:
        action_index = action_ids.index(action.action_id)
    except ValueError as exc:
        raise ValueError(
            f"semantic decision is missing action {action.action_id!r}"
        ) from exc
    normalized_baseline = replace(
        baseline,
        question=normalized_gate_question(baseline.question),
    )
    return [
        *pre_action_context_features(normalized_baseline),
        *compact_action_features(decision, action_index),
    ]


def _action_features(
    baseline: ActionRecord,
    action: ActionRecord,
    *,
    feature_mode: str,
    semantic_decision: Mapping[str, Any] | None,
) -> list[float]:
    if feature_mode == "context-geometry":
        return context_geometry_action_features(baseline, action)
    if feature_mode == "semantic-context":
        if semantic_decision is None:
            raise ValueError("semantic-context mode requires frozen semantic decisions")
        return semantic_context_action_features(baseline, action, semantic_decision)
    raise ValueError(f"unsupported action-value feature mode: {feature_mode}")


def _decision_rows(
    records: Sequence[ActionRecord],
) -> tuple[
    dict[DecisionKey, ActionRecord],
    dict[DecisionKey, list[ActionRecord]],
]:
    baselines: dict[DecisionKey, ActionRecord] = {}
    zooms: dict[DecisionKey, list[ActionRecord]] = {}
    for key, siblings in group_by_decision(records).items():
        answers = [record for record in siblings if record.action_type == "ANSWER"]
        candidates = sorted(
            (record for record in siblings if record.action_type == "ZOOM"),
            key=lambda record: record.action_id,
        )
        if len(answers) != 1 or not candidates:
            raise ValueError(f"decision {key!r} lacks one ANSWER and at least one ZOOM")
        baselines[key] = answers[0]
        zooms[key] = candidates
    if not baselines:
        raise ValueError("action-value records must be non-empty")
    return baselines, zooms


def _validate_domains(
    records_by_domain: Mapping[str, Sequence[ActionRecord]],
) -> tuple[
    dict[DecisionKey, str],
    dict[DecisionKey, ActionRecord],
    dict[DecisionKey, list[ActionRecord]],
]:
    if not records_by_domain:
        raise ValueError("at least one development domain is required")
    domain_by_key: dict[DecisionKey, str] = {}
    all_baselines: dict[DecisionKey, ActionRecord] = {}
    all_zooms: dict[DecisionKey, list[ActionRecord]] = {}
    for raw_domain, records in records_by_domain.items():
        domain = str(raw_domain).strip()
        if not domain:
            raise ValueError("development domain names must be non-empty")
        baselines, zooms = _decision_rows(records)
        overlap = set(baselines) & set(all_baselines)
        if overlap:
            raise ValueError(
                "decision IDs must be namespaced across domains: "
                f"{sorted(overlap)[:1]}"
            )
        domain_by_key.update({key: domain for key in baselines})
        all_baselines.update(baselines)
        all_zooms.update(zooms)
    return domain_by_key, all_baselines, all_zooms


def _hash_fraction(*parts: object) -> float:
    payload = "\0".join(str(part) for part in parts).encode()
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / float(2**64)


def _domain_source_split(
    domain_by_key: Mapping[DecisionKey, str],
    baselines: Mapping[DecisionKey, ActionRecord],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[DecisionKey], list[DecisionKey], dict[str, dict[str, int]]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    sources_by_domain: dict[str, set[str]] = {}
    for key, domain in domain_by_key.items():
        sources_by_domain.setdefault(domain, set()).add(baselines[key].source_id)
    validation_sources: dict[str, set[str]] = {}
    split_counts: dict[str, dict[str, int]] = {}
    for domain, sources in sorted(sources_by_domain.items()):
        if len(sources) < 2:
            raise ValueError(f"domain {domain!r} needs at least two source groups")
        ordered = sorted(
            sources,
            key=lambda source: (
                _hash_fraction("action-value-split-v1", seed, domain, source),
                source,
            ),
        )
        validation_count = min(
            max(1, round(len(ordered) * validation_fraction)),
            len(ordered) - 1,
        )
        validation_sources[domain] = set(ordered[:validation_count])
        split_counts[domain] = {
            "train_sources": len(ordered) - validation_count,
            "validation_sources": validation_count,
        }
    train_keys: list[DecisionKey] = []
    validation_keys: list[DecisionKey] = []
    for key in sorted(baselines):
        domain = domain_by_key[key]
        destination = (
            validation_keys
            if baselines[key].source_id in validation_sources[domain]
            else train_keys
        )
        destination.append(key)
    return train_keys, validation_keys, split_counts


def _action_rows(
    keys: Sequence[DecisionKey],
    baselines: Mapping[DecisionKey, ActionRecord],
    zooms: Mapping[DecisionKey, Sequence[ActionRecord]],
    domain_by_key: Mapping[DecisionKey, str],
    *,
    feature_mode: str,
    semantic_by_key: Mapping[DecisionKey, Mapping[str, Any]],
) -> tuple[list[list[float]], list[float], list[str]]:
    features: list[list[float]] = []
    gains: list[float] = []
    domains: list[str] = []
    for key in keys:
        for action in zooms[key]:
            features.append(
                _action_features(
                    baselines[key],
                    action,
                    feature_mode=feature_mode,
                    semantic_decision=semantic_by_key.get(key),
                )
            )
            gains.append(action.delta_success)
            domains.append(domain_by_key[key])
    return features, gains, domains


def _domain_balanced_weights(domains: Sequence[str]) -> list[float]:
    if not domains:
        raise ValueError("domain-balanced weights require rows")
    counts = Counter(domains)
    domain_count = len(counts)
    return [len(domains) / (domain_count * counts[domain]) for domain in domains]


def _score_candidates(
    *,
    model: Any,
    scaler: Any,
    keys: Sequence[DecisionKey],
    baselines: Mapping[DecisionKey, ActionRecord],
    zooms: Mapping[DecisionKey, Sequence[ActionRecord]],
    lambda_cost: float,
    feature_mode: str,
    semantic_by_key: Mapping[DecisionKey, Mapping[str, Any]],
) -> tuple[
    dict[DecisionKey, str | None],
    dict[DecisionKey, float],
    dict[DecisionKey, str],
]:
    import numpy as np  # type: ignore[import-not-found]

    selected: dict[DecisionKey, str | None] = {}
    best_net_values: dict[DecisionKey, float] = {}
    top_actions: dict[DecisionKey, str] = {}
    for key in keys:
        candidates = zooms[key]
        features = np.asarray(
            [
                _action_features(
                    baselines[key],
                    action,
                    feature_mode=feature_mode,
                    semantic_decision=semantic_by_key.get(key),
                )
                for action in candidates
            ],
            dtype=np.float64,
        )
        predicted_gains = model.predict(scaler.transform(features)).tolist()
        net_values = [
            float(predicted_gain) - lambda_cost * action.tool_cost
            for predicted_gain, action in zip(predicted_gains, candidates)
        ]
        best_index = max(
            range(len(candidates)),
            key=lambda index: (net_values[index], candidates[index].action_id),
        )
        best_action = candidates[best_index]
        best_value = net_values[best_index]
        top_actions[key] = best_action.action_id
        best_net_values[key] = best_value
        selected[key] = best_action.action_id if best_value > 0.0 else None
    return selected, best_net_values, top_actions


def _validation_objective(
    selected: Mapping[DecisionKey, str | None],
    keys: Sequence[DecisionKey],
    zooms: Mapping[DecisionKey, Sequence[ActionRecord]],
    domain_by_key: Mapping[DecisionKey, str],
    *,
    lambda_cost: float,
) -> dict[str, Any]:
    utility_by_domain: dict[str, list[float]] = {}
    gain_by_domain: dict[str, list[float]] = {}
    calls_by_domain: dict[str, list[float]] = {}
    for key in keys:
        selected_id = selected[key]
        action = next(
            (candidate for candidate in zooms[key] if candidate.action_id == selected_id),
            None,
        )
        gain = 0.0 if action is None else action.delta_success
        utility = 0.0 if action is None else action.voi(lambda_cost)
        call = float(action is not None)
        domain = domain_by_key[key]
        utility_by_domain.setdefault(domain, []).append(utility)
        gain_by_domain.setdefault(domain, []).append(gain)
        calls_by_domain.setdefault(domain, []).append(call)
    per_domain = {
        domain: {
            "decisions": len(utility_by_domain[domain]),
            "mean_utility": mean(utility_by_domain[domain]),
            "mean_gain": mean(gain_by_domain[domain]),
            "tool_rate": mean(calls_by_domain[domain]),
        }
        for domain in sorted(utility_by_domain)
    }
    domain_utilities = [payload["mean_utility"] for payload in per_domain.values()]
    all_utilities = [value for values in utility_by_domain.values() for value in values]
    all_gains = [value for values in gain_by_domain.values() for value in values]
    all_calls = [value for values in calls_by_domain.values() for value in values]
    return {
        "domain_balanced_mean_utility": mean(domain_utilities),
        "worst_domain_utility": min(domain_utilities),
        "pooled_mean_utility": mean(all_utilities),
        "pooled_mean_gain": mean(all_gains),
        "pooled_tool_rate": mean(all_calls),
        "per_domain": per_domain,
    }


def _calibrate_domain_robust_threshold(
    top_actions: Mapping[DecisionKey, str],
    best_values: Mapping[DecisionKey, float],
    keys: Sequence[DecisionKey],
    zooms: Mapping[DecisionKey, Sequence[ActionRecord]],
    domain_by_key: Mapping[DecisionKey, str],
    *,
    lambda_cost: float,
) -> tuple[float, dict[DecisionKey, str | None], dict[str, Any]]:
    """Choose a validation margin with no-call included as a safe candidate."""

    ordered = sorted(set(best_values[key] for key in keys), reverse=True)
    if not ordered:
        raise ValueError("threshold calibration requires validation decisions")
    thresholds = [ordered[0] + 1e-9]
    thresholds.extend((left + right) / 2.0 for left, right in zip(ordered, ordered[1:]))
    thresholds.append(ordered[-1] - 1e-9)
    candidates = []
    for threshold in thresholds:
        selected = {
            key: top_actions[key] if best_values[key] >= threshold else None
            for key in keys
        }
        metrics = _validation_objective(
            selected,
            keys,
            zooms,
            domain_by_key,
            lambda_cost=lambda_cost,
        )
        candidates.append((threshold, selected, metrics))
    return max(
        candidates,
        key=lambda candidate: (
            candidate[2]["worst_domain_utility"],
            candidate[2]["domain_balanced_mean_utility"],
            candidate[2]["pooled_mean_utility"],
            -candidate[2]["pooled_tool_rate"],
            candidate[0],
        ),
    )


def fit_multidomain_action_value_model(
    records_by_domain: Mapping[str, Sequence[ActionRecord]],
    *,
    feature_mode: str = "context-geometry",
    semantic_decisions_by_domain: Mapping[
        str, Mapping[DecisionKey, Mapping[str, Any]]
    ]
    | None = None,
    validation_fraction: float = 0.2,
    lambda_cost: float = 0.05,
    alpha_values: Sequence[float] = (0.1, 1.0, 10.0, 100.0, 1000.0),
    seed: int = 20260828,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit a source-held-out, domain-balanced direct crop-value regressor."""

    import numpy as np  # type: ignore[import-not-found]
    from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

    if lambda_cost < 0.0:
        raise ValueError("lambda_cost must be non-negative")
    if not alpha_values or any(alpha <= 0.0 for alpha in alpha_values):
        raise ValueError("alpha_values must be positive")
    domain_by_key, baselines, zooms = _validate_domains(records_by_domain)
    if feature_mode not in {"context-geometry", "semantic-context"}:
        raise ValueError(f"unsupported action-value feature mode: {feature_mode}")
    semantic_by_key: dict[DecisionKey, Mapping[str, Any]] = {}
    if feature_mode == "semantic-context":
        if semantic_decisions_by_domain is None or set(
            semantic_decisions_by_domain
        ) != set(records_by_domain):
            raise ValueError(
                "semantic-context mode requires one feature mapping per domain"
            )
        for domain in records_by_domain:
            domain_decisions = semantic_decisions_by_domain[domain]
            expected = {key for key, value in domain_by_key.items() if value == domain}
            if set(domain_decisions) != expected:
                raise ValueError(
                    f"semantic decisions do not exactly cover domain {domain!r}"
                )
            semantic_by_key.update(domain_decisions)
    train_keys, validation_keys, split_counts = _domain_source_split(
        domain_by_key,
        baselines,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    train_features, train_gains, train_domains = _action_rows(
        train_keys,
        baselines,
        zooms,
        domain_by_key,
        feature_mode=feature_mode,
        semantic_by_key=semantic_by_key,
    )
    scaler = StandardScaler().fit(np.asarray(train_features, dtype=np.float64))
    transformed_train = scaler.transform(np.asarray(train_features, dtype=np.float64))
    train_weights = np.asarray(_domain_balanced_weights(train_domains), dtype=np.float64)
    candidates = []
    for alpha in alpha_values:
        model = Ridge(alpha=float(alpha)).fit(
            transformed_train,
            np.asarray(train_gains, dtype=np.float64),
            sample_weight=train_weights,
        )
        _, best_values, top_actions = _score_candidates(
            model=model,
            scaler=scaler,
            keys=validation_keys,
            baselines=baselines,
            zooms=zooms,
            lambda_cost=lambda_cost,
            feature_mode=feature_mode,
            semantic_by_key=semantic_by_key,
        )
        threshold, selected, metrics = _calibrate_domain_robust_threshold(
            top_actions,
            best_values,
            validation_keys,
            zooms,
            domain_by_key,
            lambda_cost=lambda_cost,
        )
        candidates.append(
            {
                "alpha": float(alpha),
                "threshold": threshold,
                "model": model,
                "selected": selected,
                "best_values": best_values,
                "top_actions": top_actions,
                "metrics": metrics,
            }
        )
    winner = max(
        candidates,
        key=lambda candidate: (
            candidate["metrics"]["worst_domain_utility"],
            candidate["metrics"]["domain_balanced_mean_utility"],
            candidate["metrics"]["pooled_mean_utility"],
            -candidate["metrics"]["pooled_tool_rate"],
            -candidate["alpha"],
        ),
    )
    model = winner["model"]
    validation_key_set = set(validation_keys)
    validation_records = [
        record
        for records in records_by_domain.values()
        for record in records
        if (record.state_id, record.replicate_id) in validation_key_set
    ]
    validation_policy = PrecomputedActionGatePolicy(
        winner["selected"],
        name="multidomain_direct_action_value",
    )
    validation_policy_result = dict(
        evaluate_policy(validation_records, validation_policy, lambda_cost=lambda_cost)
    )
    report = {
        "scientific_status": (
            "development-only source-held-out selection; formal benchmark outcomes "
            "are excluded"
        ),
        "model_type": "multidomain_direct_action_value",
        "feature_mode": feature_mode,
        "decision_rule": (
            "argmax predicted_gain - lambda * tool_cost; call iff above frozen "
            "source-held-out domain-robust margin"
        ),
        "target": "counterfactual task-score gain",
        "seed": seed,
        "lambda_cost": lambda_cost,
        "validation_fraction": validation_fraction,
        "domains": sorted(set(domain_by_key.values())),
        "split_counts": split_counts,
        "train_decisions": len(train_keys),
        "validation_decisions": len(validation_keys),
        "train_action_rows": len(train_features),
        "feature_count": len(train_features[0]),
        "selected_alpha": winner["alpha"],
        "selected_threshold": winner["threshold"],
        "selection_objective": (
            "worst-domain utility, then domain-balanced mean utility, pooled utility, "
            "lower tool rate, lower alpha; no-call is an explicit candidate"
        ),
        "candidate_validation_metrics": [
            {
                "alpha": candidate["alpha"],
                "threshold": candidate["threshold"],
                **candidate["metrics"],
            }
            for candidate in candidates
        ],
        "validation_metrics": winner["metrics"],
        "validation_policy_result": validation_policy_result,
    }
    model_payload = {
        "model_type": "multidomain_direct_action_value",
        "feature_mode": feature_mode,
        "decision_rule": "predicted_net_value_above_frozen_margin",
        "seed": seed,
        "lambda_cost": lambda_cost,
        "selected_alpha": winner["alpha"],
        "threshold": winner["threshold"],
        "domains": sorted(set(domain_by_key.values())),
        "feature_count": len(train_features[0]),
        "scaler_mean": [float(value) for value in scaler.mean_.tolist()],
        "scaler_scale": [float(value) for value in scaler.scale_.tolist()],
        "coefficient": [float(value) for value in model.coef_.tolist()],
        "intercept": float(model.intercept_),
    }
    return report, model_payload


def select_frozen_action_value_actions(
    model: Mapping[str, Any],
    records: Sequence[ActionRecord],
    *,
    semantic_decisions: Mapping[DecisionKey, Mapping[str, Any]] | None = None,
) -> tuple[dict[DecisionKey, str | None], dict[DecisionKey, float]]:
    """Apply a serialized action-value model without reading target outcomes."""

    if model.get("model_type") != "multidomain_direct_action_value":
        raise ValueError("unsupported frozen action-value model type")
    baselines, zooms = _decision_rows(records)
    center = [float(value) for value in model["scaler_mean"]]
    scale = [float(value) for value in model["scaler_scale"]]
    coefficient = [float(value) for value in model["coefficient"]]
    if not center or len(center) != len(scale) or len(center) != len(coefficient):
        raise ValueError("frozen action-value feature dimensions are inconsistent")
    if any(value <= 0.0 for value in scale):
        raise ValueError("frozen action-value scaler has non-positive scales")
    selected: dict[DecisionKey, str | None] = {}
    scores: dict[DecisionKey, float] = {}
    lambda_cost = float(model["lambda_cost"])
    feature_mode = str(model["feature_mode"])
    semantic_by_key = {} if semantic_decisions is None else semantic_decisions
    if feature_mode == "semantic-context" and set(semantic_by_key) != set(baselines):
        raise ValueError("semantic decisions do not exactly cover frozen target")
    for key in sorted(baselines):
        scored_actions = []
        for action in zooms[key]:
            features = _action_features(
                baselines[key],
                action,
                feature_mode=feature_mode,
                semantic_decision=semantic_by_key.get(key),
            )
            if len(features) != len(center):
                raise ValueError("target action features differ from frozen model")
            predicted_gain = float(model["intercept"]) + sum(
                weight * (value - mean_value) / scale_value
                for weight, value, mean_value, scale_value in zip(
                    coefficient,
                    features,
                    center,
                    scale,
                )
            )
            net_value = predicted_gain - lambda_cost * action.tool_cost
            scored_actions.append((net_value, action.action_id))
        best_value, best_action_id = max(scored_actions)
        scores[key] = best_value
        selected[key] = (
            best_action_id if best_value >= float(model["threshold"]) else None
        )
    return selected, scores


def evaluate_frozen_action_value_model(
    model: Mapping[str, Any],
    records: Sequence[ActionRecord],
    *,
    bootstrap_resamples: int = 5000,
    bootstrap_seed: int = 0,
    cluster_by: str = "state_id",
    semantic_decisions: Mapping[DecisionKey, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate a frozen concrete-crop value policy on one labeled target."""

    selected, scores = select_frozen_action_value_actions(
        model,
        records,
        semantic_decisions=semantic_decisions,
    )
    policy = PrecomputedActionGatePolicy(
        selected,
        name="frozen_multidomain_direct_action_value",
    )
    lambda_cost = float(model["lambda_cost"])
    result: dict[str, Any] = dict(
        evaluate_policy(records, policy, lambda_cost=lambda_cost)
    )
    result["bootstrap"] = bootstrap_policy_evaluation(
        records,
        policy,
        lambda_cost=lambda_cost,
        n_resamples=bootstrap_resamples,
        seed=bootstrap_seed,
        cluster_by=cluster_by,  # type: ignore[arg-type]
    )
    result["predicted_net_value_summary"] = {
        "minimum": min(scores.values()),
        "mean": mean(scores.values()),
        "maximum": max(scores.values()),
        "threshold": float(model["threshold"]),
        "above_threshold": sum(
            value >= float(model["threshold"]) for value in scores.values()
        ),
        "decisions": len(scores),
    }
    return result
