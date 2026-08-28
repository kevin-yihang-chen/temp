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
    compact_rescue_features,
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


def spatial_question_features(question: str) -> list[float]:
    """Encode explicit layout language without looking at tool outcomes.

    The original compact context was designed for chart reasoning and therefore
    omitted words such as ``bottom`` and ``left``.  Those words are direct,
    pre-action evidence for crop selection in scene-text and document tasks.
    Keep this as a separate feature family so previously frozen models retain
    their exact dimensions and semantics.
    """

    normalized = normalized_gate_question(question).lower()
    patterns = (
        r"\b(?:left|leftmost)\b",
        r"\b(?:right|rightmost)\b",
        r"\b(?:top|upper|above|highest)\b",
        r"\b(?:bottom|lower|below|lowest)\b",
        r"\b(?:center|centre|middle)\b",
        r"\b(?:first|beginning|start)\b",
        r"\b(?:last|ending|end)\b",
        r"\b(?:corner|edge|side)\b",
        r"\b(?:next to|beside|near)\b",
        r"\b(?:row|column|page|header|footer)\b",
    )
    return [float(bool(re.search(pattern, normalized))) for pattern in patterns]


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


def spatial_context_geometry_action_features(
    baseline: ActionRecord,
    action: ActionRecord,
) -> list[float]:
    """Cross compact uncertainty and spatial-language state with crop geometry."""

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
    context = [
        *pre_action_context_features(normalized_baseline),
        *spatial_question_features(normalized_baseline.question),
    ]
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
    if feature_mode == "spatial-context-geometry":
        return spatial_context_geometry_action_features(baseline, action)
    if feature_mode in {"semantic-context", "hybrid-context-semantic"}:
        if semantic_decision is None:
            raise ValueError("semantic action mode requires frozen semantic decisions")
        return semantic_context_action_features(baseline, action, semantic_decision)
    raise ValueError(f"unsupported action-value feature mode: {feature_mode}")


def _state_features(
    baseline: ActionRecord,
    *,
    feature_mode: str,
    semantic_decision: Mapping[str, Any] | None,
) -> list[float]:
    normalized_baseline = replace(
        baseline,
        question=normalized_gate_question(baseline.question),
    )
    if feature_mode == "context-geometry":
        return pre_action_context_features(normalized_baseline)
    if feature_mode == "spatial-context-geometry":
        return [
            *pre_action_context_features(normalized_baseline),
            *spatial_question_features(normalized_baseline.question),
        ]
    if feature_mode == "hybrid-context-semantic":
        return pre_action_context_features(normalized_baseline)
    if feature_mode == "semantic-context":
        if semantic_decision is None:
            raise ValueError("semantic-context mode requires frozen semantic decisions")
        return compact_rescue_features(semantic_decision, normalized_baseline)
    raise ValueError(f"unsupported action-value feature mode: {feature_mode}")


def _semantic_feature_index(
    *,
    feature_mode: str,
    records_by_domain: Mapping[str, Sequence[ActionRecord]],
    domain_by_key: Mapping[DecisionKey, str],
    semantic_decisions_by_domain: Mapping[
        str, Mapping[DecisionKey, Mapping[str, Any]]
    ]
    | None,
) -> dict[DecisionKey, Mapping[str, Any]]:
    if feature_mode not in {
        "context-geometry",
        "spatial-context-geometry",
        "semantic-context",
        "hybrid-context-semantic",
    }:
        raise ValueError(f"unsupported action-value feature mode: {feature_mode}")
    semantic_by_key: dict[DecisionKey, Mapping[str, Any]] = {}
    if feature_mode in {"semantic-context", "hybrid-context-semantic"}:
        if semantic_decisions_by_domain is None or set(
            semantic_decisions_by_domain
        ) != set(records_by_domain):
            raise ValueError(
                "semantic action modes require one feature mapping per domain"
            )
        for domain in records_by_domain:
            domain_decisions = semantic_decisions_by_domain[domain]
            expected = {key for key, value in domain_by_key.items() if value == domain}
            if set(domain_decisions) != expected:
                raise ValueError(
                    f"semantic decisions do not exactly cover domain {domain!r}"
                )
            semantic_by_key.update(domain_decisions)
    return semantic_by_key


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


def _ranking_diagnostics(
    top_actions: Mapping[DecisionKey, str],
    keys: Sequence[DecisionKey],
    zooms: Mapping[DecisionKey, Sequence[ActionRecord]],
    domain_by_key: Mapping[DecisionKey, str],
    *,
    lambda_cost: float,
) -> dict[str, Any]:
    by_domain: dict[str, list[DecisionKey]] = {}
    for key in keys:
        by_domain.setdefault(domain_by_key[key], []).append(key)

    def summarize(domain_keys: Sequence[DecisionKey]) -> dict[str, Any]:
        learned = [
            next(
                action
                for action in zooms[key]
                if action.action_id == top_actions[key]
            )
            for key in domain_keys
        ]
        helpful_keys = [
            key
            for key in domain_keys
            if any(action.delta_success > 0.0 for action in zooms[key])
        ]
        return {
            "decisions": len(domain_keys),
            "helpful_states": len(helpful_keys),
            "learned_top1_mean_gain": mean(action.delta_success for action in learned),
            "learned_top1_mean_utility_if_always_called": mean(
                action.voi(lambda_cost) for action in learned
            ),
            "random_crop_mean_gain": mean(
                mean(action.delta_success for action in zooms[key])
                for key in domain_keys
            ),
            "random_crop_mean_utility_if_always_called": mean(
                mean(action.voi(lambda_cost) for action in zooms[key])
                for key in domain_keys
            ),
            "oracle_crop_mean_gain_if_always_called": mean(
                max(action.delta_success for action in zooms[key])
                for key in domain_keys
            ),
            "oracle_voi_mean_utility": mean(
                max(0.0, *(action.voi(lambda_cost) for action in zooms[key]))
                for key in domain_keys
            ),
            "learned_top1_rescue_rate_within_helpful_states": (
                mean(
                    next(
                        action.delta_success
                        for action in zooms[key]
                        if action.action_id == top_actions[key]
                    )
                    > 0.0
                    for key in helpful_keys
                )
                if helpful_keys
                else None
            ),
            "random_rescue_rate_within_helpful_states": (
                mean(
                    mean(action.delta_success > 0.0 for action in zooms[key])
                    for key in helpful_keys
                )
                if helpful_keys
                else None
            ),
        }

    return {
        "pooled": summarize(keys),
        "per_domain": {
            domain: summarize(domain_keys)
            for domain, domain_keys in sorted(by_domain.items())
        },
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
    semantic_by_key = _semantic_feature_index(
        feature_mode=feature_mode,
        records_by_domain=records_by_domain,
        domain_by_key=domain_by_key,
        semantic_decisions_by_domain=semantic_decisions_by_domain,
    )
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
                "ranking_diagnostics": _ranking_diagnostics(
                    top_actions,
                    validation_keys,
                    zooms,
                    domain_by_key,
                    lambda_cost=lambda_cost,
                ),
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
                "ranking_diagnostics": candidate["ranking_diagnostics"],
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


def _score_factorized_candidates(
    *,
    error_model: Any,
    rescue_model: Any,
    harm_model: Any,
    error_scaler: Any,
    rescue_scaler: Any,
    harm_scaler: Any,
    rescue_magnitude: float,
    harm_magnitude: float,
    keys: Sequence[DecisionKey],
    baselines: Mapping[DecisionKey, ActionRecord],
    zooms: Mapping[DecisionKey, Sequence[ActionRecord]],
    lambda_cost: float,
    feature_mode: str,
    semantic_by_key: Mapping[DecisionKey, Mapping[str, Any]],
) -> tuple[dict[DecisionKey, float], dict[DecisionKey, str]]:
    import numpy as np  # type: ignore[import-not-found]

    best_values: dict[DecisionKey, float] = {}
    top_actions: dict[DecisionKey, str] = {}
    for key in keys:
        state = np.asarray(
            [
                _state_features(
                    baselines[key],
                    feature_mode=feature_mode,
                    semantic_decision=semantic_by_key.get(key),
                )
            ],
            dtype=np.float64,
        )
        error_probability = float(
            error_model.predict_proba(error_scaler.transform(state))[0, 1]
        )
        candidates = zooms[key]
        action_rows = np.asarray(
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
        rescue_probabilities = rescue_model.predict_proba(
            rescue_scaler.transform(action_rows)
        )[:, 1].tolist()
        harm_probabilities = harm_model.predict_proba(
            harm_scaler.transform(action_rows)
        )[:, 1].tolist()
        net_values = [
            error_probability * float(rescue_probability) * rescue_magnitude
            - (1.0 - error_probability) * float(harm_probability) * harm_magnitude
            - lambda_cost * action.tool_cost
            for rescue_probability, harm_probability, action in zip(
                rescue_probabilities,
                harm_probabilities,
                candidates,
            )
        ]
        best_index = max(
            range(len(candidates)),
            key=lambda index: (net_values[index], candidates[index].action_id),
        )
        best_values[key] = net_values[best_index]
        top_actions[key] = candidates[best_index].action_id
    return best_values, top_actions


def fit_multidomain_factorized_action_value_model(
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
    """Fit explicit baseline-error, conditional-rescue, and harm heads."""

    import numpy as np  # type: ignore[import-not-found]
    from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

    if lambda_cost < 0.0:
        raise ValueError("lambda_cost must be non-negative")
    if not alpha_values or any(alpha <= 0.0 for alpha in alpha_values):
        raise ValueError("alpha_values must be positive")
    domain_by_key, baselines, zooms = _validate_domains(records_by_domain)
    semantic_by_key = _semantic_feature_index(
        feature_mode=feature_mode,
        records_by_domain=records_by_domain,
        domain_by_key=domain_by_key,
        semantic_decisions_by_domain=semantic_decisions_by_domain,
    )
    train_keys, validation_keys, split_counts = _domain_source_split(
        domain_by_key,
        baselines,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    state_features = [
        _state_features(
            baselines[key],
            feature_mode=feature_mode,
            semantic_decision=semantic_by_key.get(key),
        )
        for key in train_keys
    ]
    error_labels = [int(baselines[key].correct_before < 0.5) for key in train_keys]
    state_domains = [domain_by_key[key] for key in train_keys]
    rescue_rows = [
        (key, action)
        for key in train_keys
        if baselines[key].correct_before < 0.5
        for action in zooms[key]
    ]
    harm_rows = [
        (key, action)
        for key in train_keys
        if baselines[key].correct_before >= 0.5
        for action in zooms[key]
    ]
    if not rescue_rows or not harm_rows:
        raise ValueError("factorized action value requires wrong and correct baselines")
    rescue_features = [
        _action_features(
            baselines[key],
            action,
            feature_mode=feature_mode,
            semantic_decision=semantic_by_key.get(key),
        )
        for key, action in rescue_rows
    ]
    harm_features = [
        _action_features(
            baselines[key],
            action,
            feature_mode=feature_mode,
            semantic_decision=semantic_by_key.get(key),
        )
        for key, action in harm_rows
    ]
    rescue_labels = [int(action.delta_success > 0.0) for _, action in rescue_rows]
    harm_labels = [int(action.delta_success < 0.0) for _, action in harm_rows]
    for name, labels in (
        ("error", error_labels),
        ("rescue", rescue_labels),
        ("harm", harm_labels),
    ):
        if len(set(labels)) != 2:
            raise ValueError(f"factorized {name} head requires both outcome classes")
    error_scaler = StandardScaler().fit(np.asarray(state_features, dtype=np.float64))
    rescue_scaler = StandardScaler().fit(
        np.asarray(rescue_features, dtype=np.float64)
    )
    harm_scaler = StandardScaler().fit(np.asarray(harm_features, dtype=np.float64))
    transformed_error = error_scaler.transform(
        np.asarray(state_features, dtype=np.float64)
    )
    transformed_rescue = rescue_scaler.transform(
        np.asarray(rescue_features, dtype=np.float64)
    )
    transformed_harm = harm_scaler.transform(
        np.asarray(harm_features, dtype=np.float64)
    )
    error_weights = np.asarray(
        _domain_balanced_weights(state_domains), dtype=np.float64
    )
    rescue_domains = [domain_by_key[key] for key, _ in rescue_rows]
    harm_domains = [domain_by_key[key] for key, _ in harm_rows]
    rescue_weights = np.asarray(
        _domain_balanced_weights(rescue_domains), dtype=np.float64
    )
    harm_weights = np.asarray(_domain_balanced_weights(harm_domains), dtype=np.float64)
    positive_indices = [
        index for index, label in enumerate(rescue_labels) if label == 1
    ]
    negative_indices = [index for index, label in enumerate(harm_labels) if label == 1]
    rescue_magnitude = float(
        np.average(
            [rescue_rows[index][1].delta_success for index in positive_indices],
            weights=rescue_weights[positive_indices],
        )
    )
    harm_magnitude = float(
        np.average(
            [-harm_rows[index][1].delta_success for index in negative_indices],
            weights=harm_weights[negative_indices],
        )
    )
    candidates = []
    for alpha in alpha_values:
        model_kwargs = {
            "C": 1.0 / float(alpha),
            "solver": "liblinear",
            "max_iter": 2000,
            "random_state": seed,
        }
        error_model = LogisticRegression(**model_kwargs).fit(
            transformed_error,
            np.asarray(error_labels, dtype=np.int64),
            sample_weight=error_weights,
        )
        rescue_model = LogisticRegression(**model_kwargs).fit(
            transformed_rescue,
            np.asarray(rescue_labels, dtype=np.int64),
            sample_weight=rescue_weights,
        )
        harm_model = LogisticRegression(**model_kwargs).fit(
            transformed_harm,
            np.asarray(harm_labels, dtype=np.int64),
            sample_weight=harm_weights,
        )
        best_values, top_actions = _score_factorized_candidates(
            error_model=error_model,
            rescue_model=rescue_model,
            harm_model=harm_model,
            error_scaler=error_scaler,
            rescue_scaler=rescue_scaler,
            harm_scaler=harm_scaler,
            rescue_magnitude=rescue_magnitude,
            harm_magnitude=harm_magnitude,
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
                "error_model": error_model,
                "rescue_model": rescue_model,
                "harm_model": harm_model,
                "selected": selected,
                "metrics": metrics,
                "ranking_diagnostics": _ranking_diagnostics(
                    top_actions,
                    validation_keys,
                    zooms,
                    domain_by_key,
                    lambda_cost=lambda_cost,
                ),
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
    validation_key_set = set(validation_keys)
    validation_records = [
        record
        for records in records_by_domain.values()
        for record in records
        if (record.state_id, record.replicate_id) in validation_key_set
    ]
    validation_policy = PrecomputedActionGatePolicy(
        winner["selected"],
        name="multidomain_factorized_action_value",
    )
    validation_policy_result = dict(
        evaluate_policy(validation_records, validation_policy, lambda_cost=lambda_cost)
    )
    report = {
        "scientific_status": (
            "development-only source-held-out factorized selection; formal benchmark "
            "outcomes are excluded"
        ),
        "model_type": "multidomain_factorized_action_value",
        "feature_mode": feature_mode,
        "decision_rule": (
            "P(error)*P(rescue|error,action)*rescue_magnitude - "
            "P(correct)*P(harm|correct,action)*harm_magnitude - cost; call above "
            "frozen domain-robust margin"
        ),
        "seed": seed,
        "lambda_cost": lambda_cost,
        "validation_fraction": validation_fraction,
        "domains": sorted(set(domain_by_key.values())),
        "split_counts": split_counts,
        "train_decisions": len(train_keys),
        "validation_decisions": len(validation_keys),
        "state_feature_count": len(state_features[0]),
        "action_feature_count": len(rescue_features[0]),
        "error_train_rows": len(train_keys),
        "rescue_train_rows": len(rescue_rows),
        "harm_train_rows": len(harm_rows),
        "rescue_magnitude": rescue_magnitude,
        "harm_magnitude": harm_magnitude,
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
                "ranking_diagnostics": candidate["ranking_diagnostics"],
            }
            for candidate in candidates
        ],
        "validation_metrics": winner["metrics"],
        "validation_policy_result": validation_policy_result,
    }

    def serialized_head(prefix: str, scaler: Any, model: Any) -> dict[str, Any]:
        return {
            f"{prefix}_scaler_mean": [float(value) for value in scaler.mean_.tolist()],
            f"{prefix}_scaler_scale": [float(value) for value in scaler.scale_.tolist()],
            f"{prefix}_coefficient": [
                float(value) for value in model.coef_[0].tolist()
            ],
            f"{prefix}_intercept": float(model.intercept_[0]),
        }

    model_payload = {
        "model_type": "multidomain_factorized_action_value",
        "feature_mode": feature_mode,
        "decision_rule": "factorized_expected_net_value_above_frozen_margin",
        "seed": seed,
        "lambda_cost": lambda_cost,
        "selected_alpha": winner["alpha"],
        "threshold": winner["threshold"],
        "domains": sorted(set(domain_by_key.values())),
        "state_feature_count": len(state_features[0]),
        "action_feature_count": len(rescue_features[0]),
        "rescue_magnitude": rescue_magnitude,
        "harm_magnitude": harm_magnitude,
        **serialized_head("error", error_scaler, winner["error_model"]),
        **serialized_head("rescue", rescue_scaler, winner["rescue_model"]),
        **serialized_head("harm", harm_scaler, winner["harm_model"]),
    }
    return report, model_payload


def _serialized_probability(
    model: Mapping[str, Any],
    prefix: str,
    features: Sequence[float],
) -> float:
    center = [float(value) for value in model[f"{prefix}_scaler_mean"]]
    scale = [float(value) for value in model[f"{prefix}_scaler_scale"]]
    coefficient = [float(value) for value in model[f"{prefix}_coefficient"]]
    if not center or len(center) != len(scale) or len(center) != len(coefficient):
        raise ValueError(f"frozen {prefix} head dimensions are inconsistent")
    if len(features) != len(center) or any(value <= 0.0 for value in scale):
        raise ValueError(f"frozen {prefix} head features or scales are invalid")
    logit = float(model[f"{prefix}_intercept"]) + sum(
        weight * (value - mean_value) / scale_value
        for weight, value, mean_value, scale_value in zip(
            coefficient,
            features,
            center,
            scale,
        )
    )
    if logit >= 0.0:
        inverse = math.exp(-logit)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(logit)
    return exponent / (1.0 + exponent)


def predict_frozen_factorized_action_values(
    model: Mapping[str, Any],
    records: Sequence[ActionRecord],
    *,
    semantic_decisions: Mapping[DecisionKey, Mapping[str, Any]] | None = None,
) -> tuple[dict[DecisionKey, str], dict[DecisionKey, float]]:
    """Predict every decision's best crop and value without applying a gate."""

    if model.get("model_type") != "multidomain_factorized_action_value":
        raise ValueError("unsupported frozen factorized action-value model type")
    baselines, zooms = _decision_rows(records)
    feature_mode = str(model["feature_mode"])
    semantic_by_key = {} if semantic_decisions is None else semantic_decisions
    if feature_mode in {
        "semantic-context",
        "hybrid-context-semantic",
    } and set(semantic_by_key) != set(baselines):
        raise ValueError("semantic decisions do not exactly cover frozen target")
    actions: dict[DecisionKey, str] = {}
    scores: dict[DecisionKey, float] = {}
    for key in sorted(baselines):
        error_probability = _serialized_probability(
            model,
            "error",
            _state_features(
                baselines[key],
                feature_mode=feature_mode,
                semantic_decision=semantic_by_key.get(key),
            ),
        )
        scored_actions = []
        for action in zooms[key]:
            features = _action_features(
                baselines[key],
                action,
                feature_mode=feature_mode,
                semantic_decision=semantic_by_key.get(key),
            )
            rescue_probability = _serialized_probability(
                model, "rescue", features
            )
            harm_probability = _serialized_probability(model, "harm", features)
            net_value = (
                error_probability
                * rescue_probability
                * float(model["rescue_magnitude"])
                - (1.0 - error_probability)
                * harm_probability
                * float(model["harm_magnitude"])
                - float(model["lambda_cost"]) * action.tool_cost
            )
            scored_actions.append((net_value, action.action_id))
        best_value, best_action_id = max(scored_actions)
        scores[key] = best_value
        actions[key] = best_action_id
    return actions, scores


def select_frozen_factorized_action_value_actions(
    model: Mapping[str, Any],
    records: Sequence[ActionRecord],
    *,
    semantic_decisions: Mapping[DecisionKey, Mapping[str, Any]] | None = None,
) -> tuple[dict[DecisionKey, str | None], dict[DecisionKey, float]]:
    """Apply serialized risk/rescue/harm heads without reading target outcomes."""

    actions, scores = predict_frozen_factorized_action_values(
        model,
        records,
        semantic_decisions=semantic_decisions,
    )
    raw_threshold = model.get("threshold")
    if not isinstance(raw_threshold, (int, float)) or not math.isfinite(
        float(raw_threshold)
    ):
        raise ValueError("frozen factorized model requires a finite threshold")
    threshold = float(raw_threshold)
    selected = {
        key: actions[key] if score >= threshold else None
        for key, score in scores.items()
    }
    return selected, scores


def evaluate_frozen_factorized_action_value_model(
    model: Mapping[str, Any],
    records: Sequence[ActionRecord],
    *,
    bootstrap_resamples: int = 5000,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int = 0,
    cluster_by: str = "state_id",
    semantic_decisions: Mapping[DecisionKey, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    selected, scores = select_frozen_factorized_action_value_actions(
        model,
        records,
        semantic_decisions=semantic_decisions,
    )
    policy = PrecomputedActionGatePolicy(
        selected,
        name="frozen_multidomain_factorized_action_value",
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
        confidence=bootstrap_confidence,
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
    if feature_mode in {
        "semantic-context",
        "hybrid-context-semantic",
    } and set(semantic_by_key) != set(baselines):
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
    bootstrap_confidence: float = 0.95,
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
        confidence=bootstrap_confidence,
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
