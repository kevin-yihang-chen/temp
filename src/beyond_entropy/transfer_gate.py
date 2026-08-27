from __future__ import annotations

from statistics import mean
from typing import Any, Mapping, Sequence

from .dataset import group_by_decision, split_by_group
from .metrics import bootstrap_policy_evaluation, evaluate_policy
from .policies import EntropySearchPolicy, ExpectedRandomZoomPolicy, OracleVOIPolicy
from .rescue_gate import (
    DecisionKey,
    PrecomputedActionGatePolicy,
    PrecomputedRescueGatePolicy,
    context_quadrant_action_features,
    pre_action_context_feature_subset,
    pre_action_context_features,
    tune_rescue_gate_threshold,
)
from .schema import ActionRecord


def threshold_for_target_rate(scores: Sequence[float], target_rate: float) -> float:
    """Choose an unlabeled target-score threshold matching a source call budget."""

    if not scores or not 0.0 <= target_rate <= 1.0:
        raise ValueError("target-rate calibration requires scores and a rate in [0, 1]")
    ordered = sorted((float(score) for score in scores), reverse=True)
    selected_count = round(target_rate * len(ordered))
    if selected_count <= 0:
        return ordered[0] + 1e-9
    if selected_count >= len(ordered):
        return ordered[-1] - 1e-9
    return (ordered[selected_count - 1] + ordered[selected_count]) / 2.0


def score_frozen_factorized_context_model(
    model: Mapping[str, Any],
    records: Sequence[ActionRecord],
) -> dict[DecisionKey, float]:
    """Score states from serialized scaler/logistic parameters without refitting."""

    import math

    if model.get("model_type") != "factorized_context_cross_benchmark_transfer":
        raise ValueError("unsupported frozen factorized model type")
    baselines = _baselines(records)
    keys = sorted(baselines)
    error_mean = [float(value) for value in model["error_scaler_mean"]]
    error_scale = [float(value) for value in model["error_scaler_scale"]]
    error_coefficient = [float(value) for value in model["error_coefficient"]]
    rescue_mean = [float(value) for value in model["rescue_scaler_mean"]]
    rescue_scale = [float(value) for value in model["rescue_scaler_scale"]]
    rescue_coefficient = [float(value) for value in model["rescue_coefficient"]]
    error_dimensions = {len(error_mean), len(error_scale), len(error_coefficient)}
    rescue_dimensions = {len(rescue_mean), len(rescue_scale), len(rescue_coefficient)}
    if (
        len(error_dimensions) != 1
        or len(rescue_dimensions) != 1
        or any(value <= 0.0 for value in error_scale + rescue_scale)
    ):
        raise ValueError("frozen factorized model has inconsistent feature dimensions")
    error_feature_mode = str(model.get("error_feature_mode", "context"))
    rescue_feature_mode = str(model.get("rescue_feature_mode", "context"))

    def sigmoid(value: float) -> float:
        if value >= 0.0:
            inverse = math.exp(-value)
            return 1.0 / (1.0 + inverse)
        exponent = math.exp(value)
        return exponent / (1.0 + exponent)

    scores = {}
    for key in keys:
        error_features = pre_action_context_feature_subset(
            baselines[key],
            error_feature_mode,
        )
        rescue_features = pre_action_context_feature_subset(
            baselines[key],
            rescue_feature_mode,
        )
        if len(error_features) != len(error_mean) or len(rescue_features) != len(rescue_mean):
            raise ValueError("frozen factorized feature modes differ from model dimensions")
        error_logit = float(model["error_intercept"]) + sum(
            coefficient * (feature - center) / scale
            for coefficient, feature, center, scale in zip(
                error_coefficient,
                error_features,
                error_mean,
                error_scale,
            )
        )
        rescue_logit = float(model["rescue_intercept"]) + sum(
            coefficient * (feature - center) / scale
            for coefficient, feature, center, scale in zip(
                rescue_coefficient,
                rescue_features,
                rescue_mean,
                rescue_scale,
            )
        )
        scores[key] = sigmoid(error_logit) * sigmoid(rescue_logit)
    return scores


def select_frozen_context_quadrant_actions(
    model: Mapping[str, Any],
    records: Sequence[ActionRecord],
) -> dict[DecisionKey, str]:
    """Select one crop using serialized context-by-quadrant logistic parameters."""

    if model.get("model_type") != "context_quadrant_action_ranker_transfer":
        raise ValueError("unsupported frozen context-quadrant action model type")
    baselines = _baselines(records)
    grouped = group_by_decision(records)
    center = [float(value) for value in model["action_scaler_mean"]]
    scale = [float(value) for value in model["action_scaler_scale"]]
    coefficient = [float(value) for value in model["action_coefficient"]]
    if len(center) != len(scale) or len(center) != len(coefficient):
        raise ValueError("frozen action model has inconsistent feature dimensions")
    if any(value <= 0.0 for value in scale):
        raise ValueError("frozen action model has non-positive feature scales")
    selected = {}
    for key in sorted(grouped):
        zooms = sorted(
            (record for record in grouped[key] if record.action_type == "ZOOM"),
            key=lambda record: record.action_id,
        )
        scored = []
        for index, zoom in enumerate(zooms):
            features = context_quadrant_action_features(
                baselines[key],
                index,
                action_count=len(zooms),
            )
            if len(features) != len(center):
                raise ValueError("target action features differ from the frozen model")
            score = float(model["action_intercept"]) + sum(
                weight * (value - mean_value) / scale_value
                for weight, value, mean_value, scale_value in zip(
                    coefficient,
                    features,
                    center,
                    scale,
                )
            )
            scored.append((score, zoom.action_id))
        selected[key] = max(scored)[1]
    return selected


def _baselines(records: Sequence[ActionRecord]) -> dict[DecisionKey, ActionRecord]:
    result = {}
    for key, siblings in group_by_decision(records).items():
        answers = [record for record in siblings if record.action_type == "ANSWER"]
        if len(answers) != 1:
            raise ValueError(f"decision {key!r} must contain exactly one ANSWER")
        result[key] = answers[0]
    return result


def _outcomes(
    records: Sequence[ActionRecord],
    *,
    lambda_cost: float,
) -> dict[DecisionKey, dict[str, float | bool]]:
    result = {}
    for key, siblings in group_by_decision(records).items():
        zooms = [record for record in siblings if record.action_type == "ZOOM"]
        expected_gain = mean(record.delta_success for record in zooms)
        expected_cost = mean(record.tool_cost for record in zooms)
        result[key] = {
            "helpful": any(record.delta_success > 0.0 for record in zooms),
            "expected_utility": expected_gain - lambda_cost * expected_cost,
        }
    return result


def _evaluated(
    records: Sequence[ActionRecord],
    policy: Any,
    *,
    lambda_cost: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = dict(
        evaluate_policy(records, policy, lambda_cost=lambda_cost)
    )
    result["bootstrap"] = bootstrap_policy_evaluation(
        records,
        policy,
        lambda_cost=lambda_cost,
        n_resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    return result


def evaluate_frozen_factorized_context_model(
    model: Mapping[str, Any],
    target_records: Sequence[ActionRecord],
    *,
    source_entropy_threshold: float,
    lambda_cost: float = 0.05,
    target_strata: Mapping[str, str] | None = None,
    bootstrap_resamples: int = 5000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Evaluate a serialized source-only model without any target-label fitting."""

    import numpy as np  # type: ignore[import-not-found]
    from sklearn.metrics import (  # type: ignore[import-untyped]
        average_precision_score,
        roc_auc_score,
    )

    target_baselines = _baselines(target_records)
    target_outcomes = _outcomes(target_records, lambda_cost=lambda_cost)
    target_keys = sorted(target_baselines)
    target_scores = score_frozen_factorized_context_model(model, target_records)
    transfer_policy = PrecomputedRescueGatePolicy(
        target_scores,
        threshold=float(model["threshold"]),
        name="frozen_factorized_context_uniform_random_expectation",
    )
    entropy_policy = PrecomputedRescueGatePolicy(
        {key: target_baselines[key].entropy_before for key in target_keys},
        threshold=source_entropy_threshold,
        name="frozen_source_entropy_uniform_random_expectation",
    )
    policies = {
        "frozen_factorized_context": _evaluated(
            target_records,
            transfer_policy,
            lambda_cost=lambda_cost,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        ),
        "frozen_source_entropy": _evaluated(
            target_records,
            entropy_policy,
            lambda_cost=lambda_cost,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        ),
        "always_random": _evaluated(
            target_records,
            ExpectedRandomZoomPolicy(),
            lambda_cost=lambda_cost,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        ),
        "exhaustive_entropy": _evaluated(
            target_records,
            EntropySearchPolicy(),
            lambda_cost=lambda_cost,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        ),
        "oracle": _evaluated(
            target_records,
            OracleVOIPolicy(lambda_cost),
            lambda_cost=lambda_cost,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        ),
    }
    error_labels = np.asarray(
        [target_baselines[key].correct_before < 0.5 for key in target_keys],
        dtype=np.int64,
    )
    helpful_labels = np.asarray(
        [bool(target_outcomes[key]["helpful"]) for key in target_keys],
        dtype=np.int64,
    )
    score_array = np.asarray([target_scores[key] for key in target_keys])
    diagnostics = {
        "target_error_rate": float(error_labels.mean()),
        "target_helpful_rate": float(helpful_labels.mean()),
        "target_helpful_roc_auc": float(roc_auc_score(helpful_labels, score_array)),
        "target_helpful_average_precision": float(
            average_precision_score(helpful_labels, score_array)
        ),
    }
    strata = {}
    if target_strata is not None:
        missing_strata = {
            baseline.state_id for baseline in target_baselines.values()
        } - set(target_strata)
        if missing_strata:
            raise ValueError(f"target strata are missing states: {sorted(missing_strata)[:5]}")
        for stratum in sorted(set(target_strata.values())):
            subset = [
                record
                for record in target_records
                if target_strata[record.state_id] == stratum
            ]
            strata[stratum] = {
                "n_decisions": len(group_by_decision(subset)),
                "frozen_factorized_context": _evaluated(
                    subset,
                    transfer_policy,
                    lambda_cost=lambda_cost,
                    bootstrap_resamples=bootstrap_resamples,
                    bootstrap_seed=bootstrap_seed,
                ),
                "frozen_source_entropy": _evaluated(
                    subset,
                    entropy_policy,
                    lambda_cost=lambda_cost,
                    bootstrap_resamples=bootstrap_resamples,
                    bootstrap_seed=bootstrap_seed,
                ),
                "oracle": _evaluated(
                    subset,
                    OracleVOIPolicy(lambda_cost),
                    lambda_cost=lambda_cost,
                    bootstrap_resamples=bootstrap_resamples,
                    bootstrap_seed=bootstrap_seed,
                ),
            }
    primary = policies["frozen_factorized_context"]
    interval = primary["bootstrap"]["metrics"]["mean_policy_utility"]
    criterion = {
        "positive_utility": primary["mean_policy_utility"] > 0.0,
        "utility_ci_lower_above_zero": interval["ci_low"] > 0.0,
        "positive_accuracy_gain": primary["accuracy_gain"] > 0.0,
        "lower_tool_rate_than_unconditional_policies": primary["tool_use_rate"] < 1.0,
    }
    criterion["passed"] = all(criterion.values())
    return {
        "scientific_status": "frozen source-model target evaluation; no target fitting",
        "lambda_cost": lambda_cost,
        "target_decisions": len(target_keys),
        "frozen_threshold": float(model["threshold"]),
        "source_entropy_threshold": source_entropy_threshold,
        "diagnostics": diagnostics,
        "policies": policies,
        "strata": strata,
        "primary_confirmation_criterion": criterion,
    }


def fit_context_quadrant_action_ranker_transfer(
    source_records: Sequence[ActionRecord],
    frozen_gate_model: Mapping[str, Any],
    *,
    source_split_group: str = "image_id",
    source_train_fraction: float = 0.8,
    lambda_cost: float = 0.05,
    c_values: Sequence[float] = (0.001, 0.01, 0.1, 1.0, 10.0),
    bootstrap_resamples: int = 5000,
    bootstrap_seed: int = 0,
    seed: int = 17,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit a crop ranker while retaining a previously frozen state gate."""

    import numpy as np  # type: ignore[import-not-found]
    from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
    from sklearn.metrics import (  # type: ignore[import-untyped]
        average_precision_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

    if not c_values or any(value <= 0.0 for value in c_values):
        raise ValueError("c_values must contain positive regularization values")
    source_train, source_validation = split_by_group(
        source_records,
        group=source_split_group,  # type: ignore[arg-type]
        train_fraction=source_train_fraction,
        seed=seed,
    )
    grouped = group_by_decision(source_records)
    baselines = _baselines(source_records)
    train_keys = sorted(group_by_decision(source_train))
    validation_keys = sorted(group_by_decision(source_validation))
    wrong_train_keys = [
        key for key in train_keys if baselines[key].correct_before < 0.5
    ]

    def action_rows(keys: Sequence[DecisionKey]) -> tuple[list[list[float]], list[int]]:
        features = []
        labels = []
        for key in keys:
            zooms = sorted(
                (record for record in grouped[key] if record.action_type == "ZOOM"),
                key=lambda record: record.action_id,
            )
            for index, zoom in enumerate(zooms):
                features.append(
                    context_quadrant_action_features(
                        baselines[key],
                        index,
                        action_count=len(zooms),
                    )
                )
                labels.append(int(zoom.delta_success > 0.0))
        return features, labels

    train_features, train_labels = action_rows(wrong_train_keys)
    scaler = StandardScaler().fit(np.asarray(train_features, dtype=np.float64))
    transformed_train = scaler.transform(np.asarray(train_features, dtype=np.float64))
    gate_scores = score_frozen_factorized_context_model(
        frozen_gate_model,
        source_validation,
    )
    candidates: list[tuple[float, float, Any, dict[DecisionKey, str | None]]] = []
    for c_value in c_values:
        action_model = LogisticRegression(
            C=float(c_value),
            class_weight="balanced",
            solver="liblinear",
            max_iter=2000,
            random_state=seed,
        ).fit(transformed_train, np.asarray(train_labels, dtype=np.int64))
        decisions: dict[DecisionKey, str | None] = {}
        for key in validation_keys:
            zooms = sorted(
                (record for record in grouped[key] if record.action_type == "ZOOM"),
                key=lambda record: record.action_id,
            )
            features = np.asarray(
                [
                    context_quadrant_action_features(
                        baselines[key],
                        index,
                        action_count=len(zooms),
                    )
                    for index in range(len(zooms))
                ],
                dtype=np.float64,
            )
            scores = action_model.decision_function(scaler.transform(features))
            selected_index = max(
                range(len(zooms)),
                key=lambda index: (float(scores[index]), zooms[index].action_id),
            )
            decisions[key] = (
                zooms[selected_index].action_id
                if gate_scores[key] >= float(frozen_gate_model["threshold"])
                else None
            )
        result = evaluate_policy(
            source_validation,
            PrecomputedActionGatePolicy(decisions),
            lambda_cost=lambda_cost,
        )
        utility_value = result["mean_policy_utility"]
        if not isinstance(utility_value, (int, float)):
            raise RuntimeError("policy utility must be numeric")
        candidates.append(
            (
                float(utility_value),
                -float(c_value),
                action_model,
                decisions,
            )
        )
    validation_utility, negative_c, action_model, decisions = max(
        candidates,
        key=lambda value: value[:2],
    )
    policy = PrecomputedActionGatePolicy(
        decisions,
        name="frozen_factorized_context_quadrant_action",
    )
    policy_result = _evaluated(
        source_validation,
        policy,
        lambda_cost=lambda_cost,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    validation_action_features, validation_action_labels = action_rows(
        [key for key in validation_keys if baselines[key].correct_before < 0.5]
    )
    validation_action_scores = action_model.decision_function(
        scaler.transform(np.asarray(validation_action_features, dtype=np.float64))
    )
    report = {
        "scientific_status": (
            "source-only secondary-policy fit; gate and all action-model selection "
            "exclude the independent target"
        ),
        "seed": seed,
        "source_split_group": source_split_group,
        "source_train_fraction": source_train_fraction,
        "source_action_train_decisions": len(wrong_train_keys),
        "source_validation_decisions": len(validation_keys),
        "selected_action_c": -negative_c,
        "validation_utility": validation_utility,
        "validation_action_roc_auc": float(
            roc_auc_score(validation_action_labels, validation_action_scores)
        ),
        "validation_action_average_precision": float(
            average_precision_score(validation_action_labels, validation_action_scores)
        ),
        "policy_result": policy_result,
    }
    model_payload = {
        "model_type": "context_quadrant_action_ranker_transfer",
        "seed": seed,
        "selected_action_c": -negative_c,
        "action_count": 4,
        "action_scaler_mean": [float(value) for value in scaler.mean_.tolist()],
        "action_scaler_scale": [float(value) for value in scaler.scale_.tolist()],
        "action_coefficient": [float(value) for value in action_model.coef_[0].tolist()],
        "action_intercept": float(action_model.intercept_[0]),
    }
    return report, model_payload


def evaluate_frozen_composed_context_quadrant_policy(
    gate_model: Mapping[str, Any],
    action_model: Mapping[str, Any],
    target_records: Sequence[ActionRecord],
    *,
    lambda_cost: float = 0.05,
    bootstrap_resamples: int = 5000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Evaluate frozen stopping and action models without target fitting."""

    gate_scores = score_frozen_factorized_context_model(gate_model, target_records)
    top_actions = select_frozen_context_quadrant_actions(action_model, target_records)
    selected_actions = {
        key: (
            action_id
            if gate_scores[key] >= float(gate_model["threshold"])
            else None
        )
        for key, action_id in top_actions.items()
    }
    return _evaluated(
        target_records,
        PrecomputedActionGatePolicy(
            selected_actions,
            name="frozen_factorized_context_quadrant_action",
        ),
        lambda_cost=lambda_cost,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )


def fit_factorized_context_transfer(
    source_records: Sequence[ActionRecord],
    target_records: Sequence[ActionRecord],
    *,
    source_split_group: str = "image_id",
    source_train_fraction: float = 0.8,
    lambda_cost: float = 0.05,
    error_feature_mode: str = "context",
    rescue_feature_mode: str = "context",
    c_values: Sequence[float] = (0.001, 0.01, 0.1, 1.0, 10.0),
    target_strata: Mapping[str, str] | None = None,
    bootstrap_resamples: int = 5000,
    bootstrap_seed: int = 0,
    seed: int = 17,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit on one benchmark and evaluate a frozen factorized gate on another."""

    import numpy as np  # type: ignore[import-not-found]
    from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
    from sklearn.metrics import (  # type: ignore[import-untyped]
        average_precision_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

    if not c_values or any(value <= 0.0 for value in c_values):
        raise ValueError("c_values must contain positive regularization values")
    source_train, source_validation = split_by_group(
        source_records,
        group=source_split_group,  # type: ignore[arg-type]
        train_fraction=source_train_fraction,
        seed=seed,
    )
    source_baselines = _baselines(source_records)
    source_outcomes = _outcomes(source_records, lambda_cost=lambda_cost)
    train_keys = sorted(group_by_decision(source_train))
    validation_keys = sorted(group_by_decision(source_validation))
    wrong_train_keys = [
        key for key in train_keys if source_baselines[key].correct_before < 0.5
    ]
    source_error_features = {
        key: pre_action_context_feature_subset(
            source_baselines[key],
            error_feature_mode,
        )
        for key in train_keys + validation_keys
    }
    source_rescue_features = {
        key: pre_action_context_feature_subset(
            source_baselines[key],
            rescue_feature_mode,
        )
        for key in train_keys + validation_keys
    }
    error_scaler = StandardScaler().fit(
        np.asarray([source_error_features[key] for key in train_keys], dtype=np.float64)
    )
    rescue_scaler = StandardScaler().fit(
        np.asarray(
            [source_rescue_features[key] for key in wrong_train_keys],
            dtype=np.float64,
        )
    )
    error_train = error_scaler.transform(
        np.asarray([source_error_features[key] for key in train_keys], dtype=np.float64)
    )
    rescue_train = rescue_scaler.transform(
        np.asarray(
            [source_rescue_features[key] for key in wrong_train_keys],
            dtype=np.float64,
        )
    )
    error_validation = error_scaler.transform(
        np.asarray(
            [source_error_features[key] for key in validation_keys],
            dtype=np.float64,
        )
    )
    rescue_validation = rescue_scaler.transform(
        np.asarray(
            [source_rescue_features[key] for key in validation_keys],
            dtype=np.float64,
        )
    )
    error_train_labels = np.asarray(
        [source_baselines[key].correct_before < 0.5 for key in train_keys],
        dtype=np.int64,
    )
    rescue_train_labels = np.asarray(
        [bool(source_outcomes[key]["helpful"]) for key in wrong_train_keys],
        dtype=np.int64,
    )
    error_models = []
    rescue_models = []
    for c_value in c_values:
        error_model = LogisticRegression(
            C=float(c_value),
            class_weight="balanced",
            solver="liblinear",
            max_iter=2000,
            random_state=seed,
        ).fit(error_train, error_train_labels)
        error_models.append(
            (
                float(c_value),
                error_model,
                error_model.predict_proba(error_validation)[:, 1],
            )
        )
        rescue_model = LogisticRegression(
            C=float(c_value),
            class_weight="balanced",
            solver="liblinear",
            max_iter=2000,
            random_state=seed,
        ).fit(rescue_train, rescue_train_labels)
        rescue_models.append(
            (
                float(c_value),
                rescue_model,
                rescue_model.predict_proba(rescue_validation)[:, 1],
            )
        )
    validation_utilities = [
        float(source_outcomes[key]["expected_utility"]) for key in validation_keys
    ]
    candidates: list[tuple[float, float, float, float, float, Any, Any]] = []
    for error_c, error_model, error_probabilities in error_models:
        for rescue_c, rescue_model, rescue_probabilities in rescue_models:
            scores = error_probabilities * rescue_probabilities
            threshold, utility, tool_rate = tune_rescue_gate_threshold(
                scores.tolist(),
                validation_utilities,
            )
            candidates.append(
                (
                    utility,
                    -tool_rate,
                    -error_c,
                    -rescue_c,
                    threshold,
                    error_model,
                    rescue_model,
                )
            )
    (
        validation_utility,
        negative_tool_rate,
        negative_error_c,
        negative_rescue_c,
        threshold,
        error_model,
        rescue_model,
    ) = max(candidates, key=lambda value: value[:4])

    target_baselines = _baselines(target_records)
    target_outcomes = _outcomes(target_records, lambda_cost=lambda_cost)
    target_keys = sorted(group_by_decision(target_records))
    target_error_features = np.asarray(
        [
            pre_action_context_feature_subset(target_baselines[key], error_feature_mode)
            for key in target_keys
        ],
        dtype=np.float64,
    )
    target_rescue_features = np.asarray(
        [
            pre_action_context_feature_subset(target_baselines[key], rescue_feature_mode)
            for key in target_keys
        ],
        dtype=np.float64,
    )
    target_error_probabilities = error_model.predict_proba(
        error_scaler.transform(target_error_features)
    )[:, 1]
    target_rescue_probabilities = rescue_model.predict_proba(
        rescue_scaler.transform(target_rescue_features)
    )[:, 1]
    target_scores_array = target_error_probabilities * target_rescue_probabilities
    target_scores = {
        key: float(score)
        for key, score in zip(target_keys, target_scores_array.tolist())
    }
    transfer_policy = PrecomputedRescueGatePolicy(
        target_scores,
        threshold=threshold,
        name="factorized_context_transfer_uniform_random_expectation",
    )
    source_entropy_threshold, source_entropy_utility, source_entropy_tool_rate = (
        tune_rescue_gate_threshold(
            [source_baselines[key].entropy_before for key in validation_keys],
            validation_utilities,
        )
    )
    target_entropy_policy = PrecomputedRescueGatePolicy(
        {key: target_baselines[key].entropy_before for key in target_keys},
        threshold=source_entropy_threshold,
        name="source_tuned_entropy_uniform_random_expectation",
    )
    source_tool_rate = -negative_tool_rate
    target_quantile_threshold = threshold_for_target_rate(
        target_scores_array.tolist(),
        source_tool_rate,
    )
    quantile_transfer_policy = PrecomputedRescueGatePolicy(
        target_scores,
        threshold=target_quantile_threshold,
        name="factorized_context_quantile_transfer_random_expectation",
    )
    target_entropy_scores = [
        target_baselines[key].entropy_before for key in target_keys
    ]
    target_entropy_quantile_threshold = threshold_for_target_rate(
        target_entropy_scores,
        source_entropy_tool_rate,
    )
    target_entropy_quantile_policy = PrecomputedRescueGatePolicy(
        {
            key: score
            for key, score in zip(target_keys, target_entropy_scores)
        },
        threshold=target_entropy_quantile_threshold,
        name="entropy_quantile_transfer_random_expectation",
    )
    policies = {
        "factorized_context_transfer": _evaluated(
            target_records,
            transfer_policy,
            lambda_cost=lambda_cost,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        ),
        "source_tuned_entropy_transfer": _evaluated(
            target_records,
            target_entropy_policy,
            lambda_cost=lambda_cost,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        ),
        "factorized_context_quantile_transfer": _evaluated(
            target_records,
            quantile_transfer_policy,
            lambda_cost=lambda_cost,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        ),
        "entropy_quantile_transfer": _evaluated(
            target_records,
            target_entropy_quantile_policy,
            lambda_cost=lambda_cost,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        ),
        "always_random": _evaluated(
            target_records,
            ExpectedRandomZoomPolicy(),
            lambda_cost=lambda_cost,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        ),
        "exhaustive_entropy": _evaluated(
            target_records,
            EntropySearchPolicy(),
            lambda_cost=lambda_cost,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        ),
        "oracle": _evaluated(
            target_records,
            OracleVOIPolicy(lambda_cost),
            lambda_cost=lambda_cost,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        ),
    }
    target_error_labels = np.asarray(
        [target_baselines[key].correct_before < 0.5 for key in target_keys],
        dtype=np.int64,
    )
    target_helpful_labels = np.asarray(
        [bool(target_outcomes[key]["helpful"]) for key in target_keys],
        dtype=np.int64,
    )
    wrong_target_indices = [
        index for index, label in enumerate(target_error_labels.tolist()) if label
    ]
    wrong_target_helpful = target_helpful_labels[wrong_target_indices]
    diagnostics = {
        "target_error_roc_auc": float(
            roc_auc_score(target_error_labels, target_error_probabilities)
        ),
        "target_conditional_rescue_roc_auc": (
            float(
                roc_auc_score(
                    wrong_target_helpful,
                    target_rescue_probabilities[wrong_target_indices],
                )
            )
            if len(set(wrong_target_helpful.tolist())) == 2
            else None
        ),
        "target_helpful_roc_auc": float(
            roc_auc_score(target_helpful_labels, target_scores_array)
        ),
        "target_helpful_average_precision": float(
            average_precision_score(target_helpful_labels, target_scores_array)
        ),
    }
    strata = {}
    if target_strata is not None:
        missing_strata = {
            baseline.state_id for baseline in target_baselines.values()
        } - set(target_strata)
        if missing_strata:
            raise ValueError(f"target strata are missing states: {sorted(missing_strata)[:5]}")
        for stratum in sorted(set(target_strata.values())):
            subset = [
                record
                for record in target_records
                if target_strata[record.state_id] == stratum
            ]
            strata[stratum] = {
                "n_decisions": len(group_by_decision(subset)),
                "factorized_context_transfer": _evaluated(
                    subset,
                    transfer_policy,
                    lambda_cost=lambda_cost,
                    bootstrap_resamples=bootstrap_resamples,
                    bootstrap_seed=bootstrap_seed,
                ),
                "source_tuned_entropy_transfer": _evaluated(
                    subset,
                    target_entropy_policy,
                    lambda_cost=lambda_cost,
                    bootstrap_resamples=bootstrap_resamples,
                    bootstrap_seed=bootstrap_seed,
                ),
                "factorized_context_quantile_transfer": _evaluated(
                    subset,
                    quantile_transfer_policy,
                    lambda_cost=lambda_cost,
                    bootstrap_resamples=bootstrap_resamples,
                    bootstrap_seed=bootstrap_seed,
                ),
                "entropy_quantile_transfer": _evaluated(
                    subset,
                    target_entropy_quantile_policy,
                    lambda_cost=lambda_cost,
                    bootstrap_resamples=bootstrap_resamples,
                    bootstrap_seed=bootstrap_seed,
                ),
                "oracle": _evaluated(
                    subset,
                    OracleVOIPolicy(lambda_cost),
                    lambda_cost=lambda_cost,
                    bootstrap_resamples=bootstrap_resamples,
                    bootstrap_seed=bootstrap_seed,
                ),
            }
    report = {
        "scientific_status": (
            "cross-benchmark transfer diagnostic; model selection and threshold use "
            "source labels only"
        ),
        "seed": seed,
        "lambda_cost": lambda_cost,
        "source_split_group": source_split_group,
        "source_train_fraction": source_train_fraction,
        "error_feature_mode": error_feature_mode,
        "rescue_feature_mode": rescue_feature_mode,
        "source_model_train_decisions": len(train_keys),
        "source_model_train_wrong_decisions": len(wrong_train_keys),
        "source_validation_decisions": len(validation_keys),
        "selected_error_c": -negative_error_c,
        "selected_rescue_c": -negative_rescue_c,
        "source_validation_threshold": threshold,
        "source_validation_utility": validation_utility,
        "source_validation_tool_rate": -negative_tool_rate,
        "target_quantile_threshold": target_quantile_threshold,
        "source_entropy_threshold": source_entropy_threshold,
        "source_entropy_validation_utility": source_entropy_utility,
        "source_entropy_validation_tool_rate": source_entropy_tool_rate,
        "target_entropy_quantile_threshold": target_entropy_quantile_threshold,
        "target_decisions": len(target_keys),
        "diagnostics": diagnostics,
        "policies": policies,
        "strata": strata,
    }
    model_payload = {
        "model_type": "factorized_context_cross_benchmark_transfer",
        "seed": seed,
        "error_feature_mode": error_feature_mode,
        "rescue_feature_mode": rescue_feature_mode,
        "selected_error_c": -negative_error_c,
        "selected_rescue_c": -negative_rescue_c,
        "threshold": threshold,
        "error_scaler_mean": [float(value) for value in error_scaler.mean_.tolist()],
        "error_scaler_scale": [float(value) for value in error_scaler.scale_.tolist()],
        "error_coefficient": [
            float(value) for value in error_model.coef_[0].tolist()
        ],
        "error_intercept": float(error_model.intercept_[0]),
        "rescue_scaler_mean": [
            float(value) for value in rescue_scaler.mean_.tolist()
        ],
        "rescue_scaler_scale": [
            float(value) for value in rescue_scaler.scale_.tolist()
        ],
        "rescue_coefficient": [
            float(value) for value in rescue_model.coef_[0].tolist()
        ],
        "rescue_intercept": float(rescue_model.intercept_[0]),
    }
    return report, model_payload
