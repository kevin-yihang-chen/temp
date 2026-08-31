from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any

from .action_value import (
    _action_features,
    _semantic_feature_index,
    _validate_domains,
    predict_frozen_factorized_action_values,
)
from .dataset import group_by_decision
from .oof_action_value import (
    _domain_source_balanced_weights,
    _fit_heads,
    _score_heads,
    _source_folds,
)
from .rescue_gate import DecisionKey
from .scaled_evaluation import bootstrap_source_balanced_metrics
from .schema import ActionRecord


COMPARATOR_SEED = 20260829
COMPARATOR_FOLDS = 5
COMPARATOR_ALPHA = 1.0
COMPARATOR_FEATURE_MODE = "hybrid-context-semantic"
COMPARATOR_LAMBDA_COST = 0.05
POLICY_A_OOF_THRESHOLD = -0.0136405068067658
POLICY_A_TAIL_THRESHOLD = -0.007506966937259205
POLICY_A_DEVELOPMENT_SOURCE_CALL_RATE = 0.014781869717584004
POLICY_B_FROZEN_THRESHOLD = 0.0841866014878169
BOOTSTRAP_RESAMPLES = 20000
BOOTSTRAP_CONFIDENCE = 0.95


def _source_balanced_rate(
    called_by_key: Mapping[DecisionKey, bool],
    source_by_key: Mapping[DecisionKey, str],
) -> float:
    grouped: dict[str, list[float]] = {}
    if set(called_by_key) != set(source_by_key) or not called_by_key:
        raise ValueError("source-balanced calls must be non-empty and aligned")
    for key, called in called_by_key.items():
        grouped.setdefault(source_by_key[key], []).append(float(called))
    return mean(mean(values) for values in grouped.values())


def match_source_balanced_threshold(
    scores: Mapping[DecisionKey, float],
    source_by_key: Mapping[DecisionKey, str],
    *,
    target_rate: float,
) -> dict[str, Any]:
    """Choose a tie-preserving threshold without reading any task outcome."""

    if set(scores) != set(source_by_key) or not scores:
        raise ValueError("matched-budget scores and sources must align")
    if not 0.0 <= target_rate <= 1.0:
        raise ValueError("matched-budget target rate must lie in [0,1]")
    normalized = {key: float(value) for key, value in scores.items()}
    if any(not math.isfinite(value) for value in normalized.values()):
        raise ValueError("matched-budget scores must be finite")
    candidates: list[dict[str, Any]] = []
    for threshold in sorted(set(normalized.values()), reverse=True):
        called = {key: value >= threshold for key, value in normalized.items()}
        source_rate = _source_balanced_rate(called, source_by_key)
        candidates.append(
            {
                "threshold": threshold,
                "source_call_rate": source_rate,
                "pooled_call_rate": mean(called.values()),
                "calls": sum(called.values()),
                "absolute_source_rate_error": abs(source_rate - target_rate),
            }
        )
    selected = min(
        candidates,
        key=lambda item: (
            item["absolute_source_rate_error"],
            item["source_call_rate"],
            -item["threshold"],
        ),
    )
    return {
        **selected,
        "target_source_call_rate": target_rate,
        "selection_uses_outcomes": False,
        "ties_preserved": True,
    }


def _validate_policy_a_model(model: Mapping[str, Any]) -> None:
    expected = {
        "model_type": "multidomain_factorized_action_value",
        "training_protocol": "source_grouped_oof_domain_source_balanced_v2",
        "sample_weighting": "equal_domain_then_equal_source_then_equal_row",
        "feature_mode": COMPARATOR_FEATURE_MODE,
        "seed": COMPARATOR_SEED,
        "n_folds": COMPARATOR_FOLDS,
        "selected_alpha": COMPARATOR_ALPHA,
        "lambda_cost": COMPARATOR_LAMBDA_COST,
        "threshold": POLICY_A_OOF_THRESHOLD,
        "action_feature_count": 46,
    }
    for name, value in expected.items():
        if model.get(name) != value:
            raise ValueError(f"reserve Policy A mismatch for {name}")


def _validate_policy_b_model(model: Mapping[str, Any]) -> None:
    expected = {
        "model_type": "toolgate_style_binary_execute_proxy",
        "training_protocol": "source_grouped_oof_then_full_refit_v1",
        "sample_weighting": "equal_domain_then_equal_source_then_equal_row",
        "feature_mode": COMPARATOR_FEATURE_MODE,
        "seed": COMPARATOR_SEED,
        "n_folds": COMPARATOR_FOLDS,
        "alpha": COMPARATOR_ALPHA,
        "penalty": "l2",
        "solver": "liblinear",
        "max_iter": 2000,
        "feature_count": 46,
        "threshold": POLICY_B_FROZEN_THRESHOLD,
        "reserve_outcomes_used": False,
        "formal_outcomes_used": False,
    }
    for name, value in expected.items():
        if model.get(name) != value:
            raise ValueError(f"reserve Policy B mismatch for {name}")
    for name in ("scaler_mean", "scaler_scale", "coefficient"):
        values = model.get(name)
        if not isinstance(values, list) or len(values) != 46:
            raise ValueError(f"reserve Policy B has invalid {name}")
    if any(float(value) <= 0.0 for value in model["scaler_scale"]):
        raise ValueError("reserve Policy B scaler has non-positive scales")


def _serialized_execute_probability(
    model: Mapping[str, Any], features: Sequence[float]
) -> float:
    center = [float(value) for value in model["scaler_mean"]]
    scale = [float(value) for value in model["scaler_scale"]]
    coefficient = [float(value) for value in model["coefficient"]]
    if not len(features) == len(center) == len(scale) == len(coefficient):
        raise ValueError("reserve Policy B feature dimensions changed")
    logit = float(model["intercept"]) + sum(
        weight * (float(value) - mean_value) / scale_value
        for weight, value, mean_value, scale_value in zip(
            coefficient, features, center, scale
        )
    )
    if logit >= 0.0:
        inverse = math.exp(-logit)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(logit)
    return exponent / (1.0 + exponent)


def score_reserve_policies(
    policy_a_model: Mapping[str, Any],
    policy_b_model: Mapping[str, Any],
    records: Sequence[ActionRecord],
    *,
    semantic_decisions: Mapping[DecisionKey, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Freeze both gates and the feature-only matched budget before evaluation.

    The emitted rows contain identifiers, scores, and calls only.  They never
    serialize correctness, reward, answers, targets, or post-action entropy.
    """

    _validate_policy_a_model(policy_a_model)
    _validate_policy_b_model(policy_b_model)
    grouped = group_by_decision(records)
    actions, policy_a_values = predict_frozen_factorized_action_values(
        policy_a_model,
        records,
        semantic_decisions=semantic_decisions,
    )
    if set(grouped) != set(actions) or set(grouped) != set(policy_a_values):
        raise ValueError("reserve policy scores do not cover every decision")
    source_by_key: dict[DecisionKey, str] = {}
    policy_b_probabilities: dict[DecisionKey, float] = {}
    candidate_by_key: dict[DecisionKey, ActionRecord] = {}
    for key, siblings in grouped.items():
        answers = [record for record in siblings if record.action_type == "ANSWER"]
        zooms = [record for record in siblings if record.action_type == "ZOOM"]
        if len(answers) != 1 or len(zooms) != 4:
            raise ValueError(f"reserve decision {key!r} must have one ANSWER/four ZOOM")
        baseline = answers[0]
        matches = [record for record in zooms if record.action_id == actions[key]]
        if len(matches) != 1:
            raise ValueError(f"reserve proposed action is not unique for {key!r}")
        selected = matches[0]
        features = _action_features(
            baseline,
            selected,
            feature_mode=COMPARATOR_FEATURE_MODE,
            semantic_decision=semantic_decisions.get(key),
        )
        if len(features) != 46:
            raise ValueError("reserve pending-action feature inventory changed")
        source_by_key[key] = baseline.source_id
        candidate_by_key[key] = selected
        policy_b_probabilities[key] = _serialized_execute_probability(
            policy_b_model, features
        )

    policy_a_called = {
        key: value >= POLICY_A_TAIL_THRESHOLD
        for key, value in policy_a_values.items()
    }
    policy_b_frozen_called = {
        key: value >= float(policy_b_model["threshold"])
        for key, value in policy_b_probabilities.items()
    }
    policy_a_source_rate = _source_balanced_rate(policy_a_called, source_by_key)
    matched = match_source_balanced_threshold(
        policy_b_probabilities,
        source_by_key,
        target_rate=policy_a_source_rate,
    )
    policy_b_matched_called = {
        key: value >= float(matched["threshold"])
        for key, value in policy_b_probabilities.items()
    }
    rows = [
        {
            "state_id": key[0],
            "replicate_id": key[1],
            "source_id": source_by_key[key],
            "action_id": candidate_by_key[key].action_id,
            "policy_a_value": float(policy_a_values[key]),
            "policy_a_called": policy_a_called[key],
            "policy_b_probability": policy_b_probabilities[key],
            "policy_b_frozen_called": policy_b_frozen_called[key],
            "policy_b_matched_called": policy_b_matched_called[key],
        }
        for key in sorted(grouped)
    ]
    report = {
        "scientific_status": (
            "outcome-blind reserve policy scores frozen before one-shot evaluation"
        ),
        "n_sources": len(set(source_by_key.values())),
        "n_decisions": len(grouped),
        "selection_uses_outcomes": False,
        "serialized_outcome_fields": [],
        "shared_action_proposer": "frozen_factorized_action_value_refit_top_crop",
        "policy_a": {
            "threshold": POLICY_A_TAIL_THRESHOLD,
            "source_call_rate": policy_a_source_rate,
            "pooled_call_rate": mean(policy_a_called.values()),
            "calls": sum(policy_a_called.values()),
        },
        "policy_b_frozen": {
            "threshold": float(policy_b_model["threshold"]),
            "source_call_rate": _source_balanced_rate(
                policy_b_frozen_called, source_by_key
            ),
            "pooled_call_rate": mean(policy_b_frozen_called.values()),
            "calls": sum(policy_b_frozen_called.values()),
        },
        "policy_b_test_feature_matched": matched,
    }
    return report, rows


def _source_means(
    values: Mapping[DecisionKey, float], source_by_key: Mapping[DecisionKey, str]
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for key, value in values.items():
        grouped.setdefault(source_by_key[key], []).append(float(value))
    return {source: mean(items) for source, items in grouped.items()}


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0.0 else None


def evaluate_reserve_policies(
    records: Sequence[ActionRecord],
    score_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = COMPARATOR_SEED,
) -> dict[str, Any]:
    """Evaluate frozen score rows once with paired whole-source resampling."""

    grouped = group_by_decision(records)
    score_by_key: dict[DecisionKey, Mapping[str, Any]] = {}
    for row in score_rows:
        key = (str(row.get("state_id", "")), str(row.get("replicate_id", "")))
        if not all(key) or key in score_by_key:
            raise ValueError("reserve score rows have invalid decision identities")
        forbidden = {
            "correct_before",
            "correct_after",
            "answer_before",
            "answer_after",
            "target",
            "reward",
            "entropy_after",
        }
        if forbidden.intersection(row):
            raise ValueError("reserve score rows contain outcome fields")
        score_by_key[key] = row
    if set(score_by_key) != set(grouped):
        raise ValueError("reserve score rows do not exactly cover rollout decisions")

    variants = {
        "policy_a": "policy_a_called",
        "policy_b_frozen": "policy_b_frozen_called",
        "policy_b_test_feature_matched": "policy_b_matched_called",
    }
    metric_names = (
        "gain",
        "utility",
        "call",
        "induced_harm",
        "negative_value_call",
        "positive_utility_call",
        "helpful_proposal",
        "helpful_call",
        "gate_false_positive",
        "gate_false_negative",
    )
    values = {
        variant: {name: {} for name in metric_names} for variant in variants
    }
    source_by_key: dict[DecisionKey, str] = {}
    action_error: dict[DecisionKey, float] = {}
    helpful_state: dict[DecisionKey, float] = {}
    gate_disagreement: dict[DecisionKey, float] = {}
    for key, siblings in grouped.items():
        answers = [record for record in siblings if record.action_type == "ANSWER"]
        zooms = [record for record in siblings if record.action_type == "ZOOM"]
        if len(answers) != 1 or len(zooms) != 4:
            raise ValueError(f"invalid reserve rollout decision {key!r}")
        baseline = answers[0]
        row = score_by_key[key]
        if str(row.get("source_id", "")) != baseline.source_id:
            raise ValueError(f"reserve score source differs for {key!r}")
        matches = [
            action
            for action in zooms
            if action.action_id == str(row.get("action_id", ""))
        ]
        if len(matches) != 1:
            raise ValueError(f"reserve score action differs for {key!r}")
        selected = matches[0]
        source_by_key[key] = baseline.source_id
        any_helpful = any(action.delta_success > 0.0 for action in zooms)
        proposal_helpful = selected.delta_success > 0.0
        helpful_state[key] = float(any_helpful)
        action_error[key] = float(any_helpful and not proposal_helpful)
        gate_disagreement[key] = float(
            bool(row["policy_a_called"]) != bool(row["policy_b_frozen_called"])
        )
        for variant, call_field in variants.items():
            called = bool(row[call_field])
            gain = selected.delta_success if called else 0.0
            utility = (
                gain - COMPARATOR_LAMBDA_COST * selected.tool_cost if called else 0.0
            )
            values[variant]["gain"][key] = gain
            values[variant]["utility"][key] = utility
            values[variant]["call"][key] = float(called)
            values[variant]["induced_harm"][key] = max(-gain, 0.0)
            values[variant]["negative_value_call"][key] = float(
                called and utility < 0.0
            )
            values[variant]["positive_utility_call"][key] = float(
                called and utility > 0.0
            )
            values[variant]["helpful_proposal"][key] = float(proposal_helpful)
            values[variant]["helpful_call"][key] = float(
                called and proposal_helpful
            )
            values[variant]["gate_false_positive"][key] = float(
                called and not proposal_helpful
            )
            values[variant]["gate_false_negative"][key] = float(
                not called and proposal_helpful
            )

    source_metrics: dict[str, dict[str, float]] = {
        source: {} for source in sorted(set(source_by_key.values()))
    }
    source_points: dict[str, dict[str, Any]] = {}
    question_points: dict[str, dict[str, Any]] = {}
    for variant in variants:
        source_values = {
            name: _source_means(metric, source_by_key)
            for name, metric in values[variant].items()
        }
        source_points[variant] = {
            name: mean(per_source.values())
            for name, per_source in source_values.items()
        }
        question_points[variant] = {
            name: mean(metric.values()) for name, metric in values[variant].items()
        }
        for source in source_metrics:
            for name in metric_names:
                source_metrics[source][f"{variant}_{name}"] = source_values[name][source]
        for points in (source_points[variant], question_points[variant]):
            points["gain_per_call"] = _safe_ratio(points["gain"], points["call"])
            points["helpful_call_recall"] = _safe_ratio(
                points["helpful_call"], points["helpful_proposal"]
            )
            points["positive_utility_precision"] = _safe_ratio(
                points["positive_utility_call"], points["call"]
            )

    difference_by_source = {
        source: source_metrics[source]["policy_a_utility"]
        - source_metrics[source]["policy_b_frozen_utility"]
        for source in source_metrics
    }
    for source, difference in difference_by_source.items():
        source_metrics[source]["policy_a_minus_policy_b_utility"] = difference
    bootstrap = bootstrap_source_balanced_metrics(
        source_metrics,
        n_resamples=bootstrap_resamples,
        confidence_level=BOOTSTRAP_CONFIDENCE,
        seed=bootstrap_seed,
    )
    primary = bootstrap["metrics"]["policy_a_minus_policy_b_utility"]
    source_action_error = _source_means(action_error, source_by_key)
    source_helpful_state = _source_means(helpful_state, source_by_key)
    source_disagreement = _source_means(gate_disagreement, source_by_key)
    pass_rule = {
        "paired_mean_positive": float(primary["point_estimate"]) > 0.0,
        "paired_95pct_ci_low_positive": float(primary["ci_low"]) > 0.0,
    }
    return {
        "scientific_status": (
            "one-shot outcome-sealed reserve comparison; cannot revise primary formal"
        ),
        "supports_policy_a_over_policy_b": all(pass_rule.values()),
        "pass_rule": pass_rule,
        "n_sources": len(source_metrics),
        "n_decisions": len(grouped),
        "lambda_cost": COMPARATOR_LAMBDA_COST,
        "primary_estimand": {
            "name": "source_balanced_utility_policy_a_minus_policy_b_frozen",
            **primary,
        },
        "source_balanced": source_points,
        "question_balanced": question_points,
        "paired_source_bootstrap": bootstrap,
        "gate_disagreement": {
            "source_balanced_rate": mean(source_disagreement.values()),
            "question_balanced_rate": mean(gate_disagreement.values()),
        },
        "shared_action_selection": {
            "helpful_state_source_mass": mean(source_helpful_state.values()),
            "helpful_state_question_mass": mean(helpful_state.values()),
            "action_selection_error_source_mass": mean(source_action_error.values()),
            "action_selection_error_question_mass": mean(action_error.values()),
            "action_selection_error_given_helpful_question": _safe_ratio(
                sum(action_error.values()), sum(helpful_state.values())
            ),
        },
        "test_feature_matched_comparison_is_secondary": True,
        "formal_outcomes_used_for_thresholds": False,
        "reserve_outcomes_used_for_thresholds": False,
    }


def fit_reserve_toolgate_comparator(
    records_by_domain: Mapping[str, Sequence[ActionRecord]],
    *,
    semantic_decisions_by_domain: Mapping[
        str, Mapping[DecisionKey, Mapping[str, Any]]
    ],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reproduce the frozen Policy-B OOF fit from development data only."""

    import numpy as np  # type: ignore[import-not-found]
    from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

    domain_by_key, baselines, zooms = _validate_domains(records_by_domain)
    semantic_by_key = _semantic_feature_index(
        feature_mode=COMPARATOR_FEATURE_MODE,
        records_by_domain=records_by_domain,
        domain_by_key=domain_by_key,
        semantic_decisions_by_domain=semantic_decisions_by_domain,
    )
    fold_by_key, fold_source_counts = _source_folds(
        domain_by_key,
        baselines,
        n_folds=COMPARATOR_FOLDS,
        seed=COMPARATOR_SEED,
    )
    keys = sorted(baselines)
    policy_a_values: dict[DecisionKey, float] = {}
    policy_a_actions: dict[DecisionKey, str] = {}
    fold_counts: list[dict[str, int]] = []
    for fold in range(COMPARATOR_FOLDS):
        train_keys = [key for key in keys if fold_by_key[key] != fold]
        test_keys = [key for key in keys if fold_by_key[key] == fold]
        heads = _fit_heads(
            train_keys,
            alpha=COMPARATOR_ALPHA,
            seed=COMPARATOR_SEED + fold,
            feature_mode=COMPARATOR_FEATURE_MODE,
            baselines=baselines,
            zooms=zooms,
            domain_by_key=domain_by_key,
            semantic_by_key=semantic_by_key,
        )
        values, actions = _score_heads(
            heads,
            test_keys,
            lambda_cost=COMPARATOR_LAMBDA_COST,
            feature_mode=COMPARATOR_FEATURE_MODE,
            baselines=baselines,
            zooms=zooms,
            semantic_by_key=semantic_by_key,
        )
        policy_a_values.update(values)
        policy_a_actions.update(actions)
        fold_counts.append(
            {
                "fold": fold,
                "train_decisions": len(train_keys),
                "test_decisions": len(test_keys),
            }
        )
    if set(policy_a_values) != set(keys) or set(policy_a_actions) != set(keys):
        raise RuntimeError("Policy A OOF reconstruction is incomplete")
    tail_threshold = sorted(policy_a_values.values(), reverse=True)[
        math.ceil(0.015 * len(keys)) - 1
    ]
    if tail_threshold != POLICY_A_TAIL_THRESHOLD:
        raise ValueError("Policy A 1.5-percent tail threshold no longer reproduces")
    policy_a_called = {
        key: value >= POLICY_A_TAIL_THRESHOLD
        for key, value in policy_a_values.items()
    }
    source_by_key = {key: baselines[key].source_id for key in keys}
    policy_a_source_rate = _source_balanced_rate(policy_a_called, source_by_key)
    if not math.isclose(
        policy_a_source_rate,
        POLICY_A_DEVELOPMENT_SOURCE_CALL_RATE,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("Policy A development source call rate no longer reproduces")

    feature_by_key: dict[DecisionKey, list[float]] = {}
    label_by_key: dict[DecisionKey, int] = {}
    for key in keys:
        matches = [
            action
            for action in zooms[key]
            if action.action_id == policy_a_actions[key]
        ]
        if len(matches) != 1:
            raise RuntimeError("Policy A OOF proposal is not unique")
        action = matches[0]
        feature_by_key[key] = _action_features(
            baselines[key],
            action,
            feature_mode=COMPARATOR_FEATURE_MODE,
            semantic_decision=semantic_by_key.get(key),
        )
        label_by_key[key] = int(
            baselines[key].correct_before < 0.5 and action.correct_after >= 0.5
        )
    if any(len(features) != 46 for features in feature_by_key.values()):
        raise ValueError("Policy B development feature inventory changed")

    oof_probabilities: dict[DecisionKey, float] = {}
    detailed_fold_counts: list[dict[str, int]] = []
    for fold in range(COMPARATOR_FOLDS):
        train_keys = [key for key in keys if fold_by_key[key] != fold]
        test_keys = [key for key in keys if fold_by_key[key] == fold]
        train_array = np.asarray(
            [feature_by_key[key] for key in train_keys], dtype=np.float64
        )
        test_array = np.asarray(
            [feature_by_key[key] for key in test_keys], dtype=np.float64
        )
        labels = np.asarray([label_by_key[key] for key in train_keys], dtype=np.int64)
        weights = np.asarray(
            _domain_source_balanced_weights(
                [domain_by_key[key] for key in train_keys],
                [source_by_key[key] for key in train_keys],
            ),
            dtype=np.float64,
        )
        scaler = StandardScaler().fit(train_array)
        gate = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="liblinear",
            max_iter=2000,
            random_state=COMPARATOR_SEED + fold,
        ).fit(scaler.transform(train_array), labels, sample_weight=weights)
        probabilities = gate.predict_proba(scaler.transform(test_array))[:, 1]
        oof_probabilities.update(
            {key: float(value) for key, value in zip(test_keys, probabilities)}
        )
        detailed_fold_counts.append(
            {
                "fold": fold,
                "train_decisions": len(train_keys),
                "test_decisions": len(test_keys),
                "train_positive_labels": int(labels.sum()),
                "test_positive_labels": sum(label_by_key[key] for key in test_keys),
            }
        )
    if set(oof_probabilities) != set(keys):
        raise RuntimeError("Policy B OOF probabilities are incomplete")
    matched = match_source_balanced_threshold(
        oof_probabilities,
        source_by_key,
        target_rate=POLICY_A_DEVELOPMENT_SOURCE_CALL_RATE,
    )
    if float(matched["threshold"]) != POLICY_B_FROZEN_THRESHOLD:
        raise ValueError("Policy B frozen threshold no longer reproduces")

    full_array = np.asarray([feature_by_key[key] for key in keys], dtype=np.float64)
    full_labels = np.asarray([label_by_key[key] for key in keys], dtype=np.int64)
    full_weights = np.asarray(
        _domain_source_balanced_weights(
            [domain_by_key[key] for key in keys],
            [source_by_key[key] for key in keys],
        ),
        dtype=np.float64,
    )
    scaler = StandardScaler().fit(full_array)
    gate = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="liblinear",
        max_iter=2000,
        random_state=COMPARATOR_SEED,
    ).fit(scaler.transform(full_array), full_labels, sample_weight=full_weights)
    model = {
        "model_type": "toolgate_style_binary_execute_proxy",
        "training_protocol": "source_grouped_oof_then_full_refit_v1",
        "sample_weighting": "equal_domain_then_equal_source_then_equal_row",
        "feature_mode": COMPARATOR_FEATURE_MODE,
        "proposal_training_mode": "policy_a_source_oof_top_crop",
        "pending_action_proposer": "frozen_factorized_action_value_refit_top_crop",
        "proxy_target": "baseline_incorrect_and_proposed_crop_correct",
        "seed": COMPARATOR_SEED,
        "n_folds": COMPARATOR_FOLDS,
        "alpha": COMPARATOR_ALPHA,
        "penalty": "l2",
        "solver": "liblinear",
        "max_iter": 2000,
        "feature_count": full_array.shape[1],
        "threshold": float(matched["threshold"]),
        "target_source_call_rate": POLICY_A_DEVELOPMENT_SOURCE_CALL_RATE,
        "oof_source_call_rate": float(matched["source_call_rate"]),
        "oof_pooled_call_rate": float(matched["pooled_call_rate"]),
        "scaler_mean": [float(value) for value in scaler.mean_.tolist()],
        "scaler_scale": [float(value) for value in scaler.scale_.tolist()],
        "coefficient": [float(value) for value in gate.coef_[0].tolist()],
        "intercept": float(gate.intercept_[0]),
        "formal_outcomes_used": False,
        "reserve_outcomes_used": False,
    }
    called = {
        key: probability >= float(matched["threshold"])
        for key, probability in oof_probabilities.items()
    }
    true_positives = sum(called[key] and label_by_key[key] for key in keys)
    report = {
        "scientific_status": (
            "development-only OOF comparator fit; reserve outcomes remain unused"
        ),
        "policy_a_reconstruction": {
            "alpha": COMPARATOR_ALPHA,
            "feature_mode": COMPARATOR_FEATURE_MODE,
            "fold_source_counts": fold_source_counts,
            "fold_counts": fold_counts,
            "oof_threshold": POLICY_A_OOF_THRESHOLD,
            "tail_threshold": tail_threshold,
            "tail_source_call_rate": policy_a_source_rate,
            "tail_pooled_call_rate": mean(policy_a_called.values()),
        },
        "policy_b_fit": {
            "fold_counts": detailed_fold_counts,
            "positive_labels": sum(label_by_key.values()),
            "positive_prevalence": mean(label_by_key.values()),
        },
        "matched_budget": matched,
        "policy_b_proxy_oof": {
            "decisions": len(keys),
            "calls": sum(called.values()),
            "true_positives": true_positives,
            "precision": _safe_ratio(true_positives, sum(called.values())),
            "recall": _safe_ratio(true_positives, sum(label_by_key.values())),
        },
        "formal_outcomes_used": False,
        "reserve_outcomes_used": False,
    }
    return report, model
