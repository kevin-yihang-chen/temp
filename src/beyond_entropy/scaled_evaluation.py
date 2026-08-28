from __future__ import annotations

import math
from statistics import mean
from typing import Any, Mapping, Sequence

from .dataset import group_by_decision
from .rescue_gate import DecisionKey
from .scaled_action_value import (
    ScaledActionValuePrediction,
    predict_scaled_action_value,
)
from .schema import ActionRecord


def _source_means(
    values: Mapping[DecisionKey, float],
    source_by_key: Mapping[DecisionKey, str],
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for key, value in values.items():
        grouped.setdefault(source_by_key[key], []).append(float(value))
    return {source: mean(items) for source, items in grouped.items()}


def bootstrap_source_balanced_metrics(
    source_metrics: Mapping[str, Mapping[str, float]],
    *,
    n_resamples: int = 20000,
    confidence_level: float = 0.975,
    seed: int = 20260828,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Percentile intervals from iid whole-source bootstrap resamples."""

    import numpy as np  # type: ignore[import-not-found]

    if n_resamples <= 0 or batch_size <= 0:
        raise ValueError("bootstrap resamples and batch size must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0,1)")
    sources = sorted(source_metrics)
    if len(sources) < 2:
        raise ValueError("source bootstrap requires at least two sources")
    metric_names = sorted(next(iter(source_metrics.values())))
    if any(set(source_metrics[source]) != set(metric_names) for source in sources):
        raise ValueError("source bootstrap metric keys must be identical")
    arrays = {
        name: np.asarray(
            [float(source_metrics[source][name]) for source in sources],
            dtype=np.float64,
        )
        for name in metric_names
    }
    if any(not bool(np.isfinite(values).all()) for values in arrays.values()):
        raise ValueError("source bootstrap metrics must be finite")
    rng = np.random.default_rng(seed)
    draws = {
        name: np.empty(n_resamples, dtype=np.float64) for name in metric_names
    }
    completed = 0
    while completed < n_resamples:
        current = min(batch_size, n_resamples - completed)
        indices = rng.integers(0, len(sources), size=(current, len(sources)))
        for name, values in arrays.items():
            draws[name][completed : completed + current] = values[indices].mean(axis=1)
        completed += current
    alpha = 1.0 - confidence_level
    return {
        "method": "iid_whole_source_percentile_bootstrap",
        "n_sources": len(sources),
        "n_resamples": n_resamples,
        "confidence_level": confidence_level,
        "seed": seed,
        "metrics": {
            name: {
                "point_estimate": float(values.mean()),
                "ci_low": float(np.quantile(draws[name], alpha / 2.0)),
                "ci_high": float(np.quantile(draws[name], 1.0 - alpha / 2.0)),
            }
            for name, values in arrays.items()
        },
    }


def evaluate_scaled_risk_controlled_policy(
    model: Mapping[str, Any],
    records: Sequence[ActionRecord],
    *,
    semantic_decisions: Mapping[DecisionKey, Mapping[str, Any]] | None = None,
    bootstrap_resamples: int = 20000,
    bootstrap_confidence: float = 0.975,
    bootstrap_seed: int = 20260828,
) -> dict[str, Any]:
    """Evaluate one already calibrated policy without changing its threshold."""

    threshold = model.get("calibrated_threshold")
    calibration = model.get("risk_calibration")
    if threshold is None or not isinstance(calibration, Mapping):
        raise ValueError("formal evaluation requires a non-degenerate calibrated model")
    if calibration.get("selection_status") != "selected_non_degenerate_safe_threshold":
        raise ValueError("formal evaluation requires successful risk calibration")
    threshold = float(threshold)
    if not math.isfinite(threshold):
        raise ValueError("calibrated threshold must be finite")
    selected_threshold = calibration.get("selected_threshold")
    if selected_threshold is None or float(selected_threshold) != threshold:
        raise ValueError("model threshold does not match the frozen calibration report")
    lambda_cost = float(model["lambda_cost"])
    predictions = predict_scaled_action_value(
        model,
        records,
        semantic_decisions=semantic_decisions,
    )
    prediction_by_key = {
        (prediction.state_id, prediction.replicate_id): prediction
        for prediction in predictions
    }
    grouped = group_by_decision(records)
    if set(prediction_by_key) != set(grouped):
        raise ValueError("scaled evaluation predictions do not cover rollout decisions")

    source_by_key: dict[DecisionKey, str] = {}
    values: dict[str, dict[DecisionKey, float]] = {
        name: {}
        for name in (
            "utility",
            "gain",
            "call",
            "induced_harm",
            "net_negative_call",
            "negative_net_value",
            "oracle_utility",
            "random_utility",
            "entropy_search_utility",
        )
    }
    fixed_utilities: dict[str, dict[DecisionKey, float]] = {}
    helpful_states = 0
    selected_rescues = 0.0
    random_rescue_total = 0.0
    calls = 0
    positive_utility_calls = 0
    unnecessary_calls = 0
    stoppable_states = 0
    correct_stops = 0
    for key, siblings in grouped.items():
        answers = [record for record in siblings if record.action_type == "ANSWER"]
        zooms = sorted(
            (record for record in siblings if record.action_type == "ZOOM"),
            key=lambda record: record.action_id,
        )
        if len(answers) != 1 or len(zooms) < 2:
            raise ValueError(f"invalid scaled evaluation decision {key!r}")
        baseline = answers[0]
        prediction: ScaledActionValuePrediction = prediction_by_key[key]
        matches = [zoom for zoom in zooms if zoom.action_id == prediction.action_id]
        if len(matches) != 1 or prediction.source_id != baseline.source_id:
            raise ValueError(f"scaled evaluation action mismatch for {key!r}")
        selected = matches[0]
        called = prediction.score >= threshold
        gain = selected.delta_success if called else 0.0
        utility = gain - lambda_cost * selected.tool_cost if called else 0.0
        source_by_key[key] = baseline.source_id
        values["utility"][key] = utility
        values["gain"][key] = gain
        values["call"][key] = float(called)
        values["induced_harm"][key] = max(-gain, 0.0)
        values["net_negative_call"][key] = float(called and utility < 0.0)
        values["negative_net_value"][key] = max(-utility, 0.0)
        net_values = [
            zoom.delta_success - lambda_cost * zoom.tool_cost for zoom in zooms
        ]
        values["oracle_utility"][key] = max(0.0, max(net_values))
        values["random_utility"][key] = mean(net_values)
        entropy_action = min(
            zooms,
            key=lambda zoom: (zoom.entropy_after, zoom.action_id),
        )
        values["entropy_search_utility"][key] = (
            entropy_action.delta_success
            - lambda_cost * sum(zoom.tool_cost for zoom in zooms)
        )
        for zoom in zooms:
            fixed_utilities.setdefault(zoom.action_id, {})[key] = (
                zoom.delta_success - lambda_cost * zoom.tool_cost
            )
        helpful = any(zoom.delta_success > 0.0 for zoom in zooms)
        if helpful:
            helpful_states += 1
            selected_rescues += float(selected.delta_success > 0.0)
            random_rescue_total += mean(zoom.delta_success > 0.0 for zoom in zooms)
        no_positive_net_action = max(net_values) <= 0.0
        if no_positive_net_action:
            stoppable_states += 1
            correct_stops += int(not called)
        if called:
            calls += 1
            positive_utility_calls += int(utility > 0.0)
            unnecessary_calls += int(utility <= 0.0)

    source_metric_names = tuple(values)
    source_metric_values = {
        name: _source_means(metric_values, source_by_key)
        for name, metric_values in values.items()
    }
    sources = sorted(source_metric_values["utility"])
    source_metrics = {
        source: {
            name: source_metric_values[name][source] for name in source_metric_names
        }
        for source in sources
    }
    bootstrap = bootstrap_source_balanced_metrics(
        source_metrics,
        n_resamples=bootstrap_resamples,
        confidence_level=bootstrap_confidence,
        seed=bootstrap_seed,
    )
    source_point = {
        name: mean(source_metric_values[name].values()) for name in source_metric_names
    }
    question_point = {
        name: mean(metric_values.values()) for name, metric_values in values.items()
    }
    fixed_source_utilities = {
        action_id: mean(_source_means(metric, source_by_key).values())
        for action_id, metric in sorted(fixed_utilities.items())
    }
    primary_interval = bootstrap["metrics"]["utility"]
    passed = (
        source_point["utility"] > 0.0
        and float(primary_interval["ci_low"]) > 0.0
        and question_point["utility"] > 0.0
        and source_point["call"] >= 0.01
    )
    return {
        "scientific_status": "one-shot evaluation of a frozen risk-controlled policy",
        "passed": passed,
        "threshold": threshold,
        "lambda_cost": lambda_cost,
        "n_sources": len(sources),
        "n_decisions": len(grouped),
        "source_balanced": source_point,
        "question_weighted": question_point,
        "source_bootstrap": bootstrap,
        "ranking": {
            "helpful_states": helpful_states,
            "top1_rescue_rate_within_helpful_states": (
                selected_rescues / helpful_states if helpful_states else 0.0
            ),
            "random_rescue_rate_within_helpful_states": (
                random_rescue_total / helpful_states if helpful_states else 0.0
            ),
            "fixed_crop_source_utilities": fixed_source_utilities,
        },
        "selection": {
            "calls": calls,
            "source_balanced_raw_gain_per_call": (
                source_point["gain"] / source_point["call"]
                if source_point["call"] > 0.0
                else 0.0
            ),
            "source_balanced_utility_per_call": (
                source_point["utility"] / source_point["call"]
                if source_point["call"] > 0.0
                else 0.0
            ),
            "positive_utility_call_precision": (
                positive_utility_calls / calls if calls else 0.0
            ),
            "question_weighted_raw_gain_per_call": (
                sum(values["gain"].values()) / calls if calls else 0.0
            ),
            "question_weighted_utility_per_call": (
                sum(values["utility"].values()) / calls if calls else 0.0
            ),
            "unnecessary_call_rate": unnecessary_calls / calls if calls else 0.0,
            "stoppable_states": stoppable_states,
            "correct_stopping_rate": (
                correct_stops / stoppable_states if stoppable_states else 0.0
            ),
        },
        "oracle_regret": source_point["oracle_utility"] - source_point["utility"],
        "pass_rule": {
            "source_utility_positive": source_point["utility"] > 0.0,
            "source_utility_97_5pct_ci_low_positive": float(primary_interval["ci_low"])
            > 0.0,
            "question_weighted_utility_positive": question_point["utility"] > 0.0,
            "source_call_rate_at_least_0_01": source_point["call"] >= 0.01,
        },
    }
