from __future__ import annotations

import math
from dataclasses import replace

from beyond_entropy.predictability_audit import collapse_fixed_entropy_tool
from beyond_entropy.predictability_evaluation import (
    Prediction,
    calls_at_rate,
    calls_at_threshold,
    paired_source_bootstrap_policy_difference,
    paired_source_bootstrap_utility,
    policy_curve,
    policy_metrics,
    prediction_metrics,
    select_validation_threshold,
)
from beyond_entropy.predictability_audit import BinaryToolOutcome
from test_predictability_audit import _siblings


def _outcomes_and_predictions() -> tuple[list, list[Prediction]]:
    records = []
    for index in range(4):
        records.extend(
            _siblings(state_id=f"s{index}", source_id=f"source-{index // 2}")
        )
    outcomes = collapse_fixed_entropy_tool(records)
    predictions = [
        Prediction(
            state_id=item.state_id,
            replicate_id=item.replicate_id,
            score=float(index),
            positive_gain_probability=0.8,
            rescue_probability=0.8,
            harm_probability=0.1,
        )
        for index, item in enumerate(outcomes)
    ]
    return outcomes, predictions


def test_validation_threshold_is_selected_without_test_inputs() -> None:
    outcomes, predictions = _outcomes_and_predictions()
    selected = select_validation_threshold(outcomes, predictions, lambda_cost=0.05)
    assert selected["threshold"] == 0.0
    assert selected["validation_calls"] == 4
    assert selected["validation_utility"] == 0.8
    assert calls_at_threshold(predictions, float(selected["threshold"])) == [True] * 4


def test_validation_no_call_threshold_is_finite_for_strict_json() -> None:
    outcomes, predictions = _outcomes_and_predictions()
    harmful = [
        type(item)(**{**item.__dict__, "y_tool": max(0.0, item.y0 - 1.0)})
        for item in outcomes
    ]
    selected = select_validation_threshold(harmful, predictions, lambda_cost=0.05)
    assert selected["validation_calls"] == 0
    assert math.isfinite(float(selected["threshold"]))
    assert calls_at_threshold(predictions, float(selected["threshold"])) == [False] * 4


def test_policy_metrics_and_fixed_rate_curve() -> None:
    outcomes, predictions = _outcomes_and_predictions()
    calls = calls_at_rate(predictions, 0.5)
    assert calls == [False, False, True, True]
    metrics = policy_metrics(outcomes, calls, lambda_cost=0.05)
    assert metrics["call_rate"] == 0.5
    assert metrics["incremental_utility"] == 0.4
    curve = policy_curve(
        outcomes, predictions, lambda_cost=0.05, call_rates=(0.0, 0.5, 1.0)
    )
    assert [item["calls"] for item in curve] == [0, 2, 4]


def test_prediction_metrics_include_frozen_required_set() -> None:
    outcomes, predictions = _outcomes_and_predictions()
    outcomes[0] = type(outcomes[0])(**{**outcomes[0].__dict__, "y_tool": 0.0})
    metrics = prediction_metrics(outcomes, predictions)
    assert set(metrics) == {
        "auroc",
        "auprc",
        "brier",
        "calibration_error",
        "rescue_auprc",
        "harm_auprc",
    }
    changed_cost = [replace(item, tool_cost=item.tool_cost * 100.0) for item in outcomes]
    assert prediction_metrics(changed_cost, predictions) == metrics


def test_paired_bootstrap_uses_source_level_differences() -> None:
    outcomes, _ = _outcomes_and_predictions()
    report = paired_source_bootstrap_utility(
        outcomes,
        [True, True, True, True],
        [False, False, False, False],
        lambda_cost=0.05,
        resamples=100,
        confidence_level=0.95,
        seed=17,
    )
    assert report["sources"] == 2
    assert report["point"] == 0.8
    assert report["lower"] == 0.8
    assert report["upper"] == 0.8


def test_paired_bootstrap_compares_independent_outcomes_and_costs() -> None:
    candidate = [
        BinaryToolOutcome(
            state_id=f"s{index}",
            replicate_id="r0",
            image_id=f"i{index}",
            source_id=f"source-{index // 2}",
            selected_action_id="exhaustive",
            y0=0.0,
            y_tool=1.0,
            tool_cost=4.0,
            tool_calls=4,
        )
        for index in range(4)
    ]
    baseline = [
        BinaryToolOutcome(
            state_id=item.state_id,
            replicate_id=item.replicate_id,
            image_id=item.image_id,
            source_id=item.source_id,
            selected_action_id="one-crop",
            y0=item.y0,
            y_tool=0.5,
            tool_cost=1.0,
            tool_calls=1,
        )
        for item in reversed(candidate)
    ]
    report = paired_source_bootstrap_policy_difference(
        candidate,
        [True] * 4,
        baseline,
        [True] * 4,
        lambda_cost=0.1,
        resamples=100,
        confidence_level=0.95,
        seed=17,
    )
    assert report["candidate_utility"] == 0.6
    assert report["baseline_utility"] == 0.4
    assert abs(report["point"] - 0.2) < 1e-12
    assert abs(report["lower"] - 0.2) < 1e-12
    assert abs(report["upper"] - 0.2) < 1e-12


def test_paired_bootstrap_rejects_mismatched_answer_now_outcome() -> None:
    outcomes, _ = _outcomes_and_predictions()
    baseline = list(outcomes)
    baseline[0] = type(baseline[0])(
        **{**baseline[0].__dict__, "y0": 1.0 - baseline[0].y0}
    )
    try:
        paired_source_bootstrap_policy_difference(
            outcomes,
            [True] * len(outcomes),
            baseline,
            [False] * len(outcomes),
            lambda_cost=0.05,
            resamples=10,
            confidence_level=0.95,
            seed=17,
        )
    except ValueError as exc:
        assert "identity or Y0 mismatch" in str(exc)
    else:
        raise AssertionError("unpaired policy ledgers should fail closed")
