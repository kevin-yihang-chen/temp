from __future__ import annotations

import math

import pytest

from beyond_entropy.risk_control import (
    AcquisitionCalibrationRow,
    RiskConstraint,
    bernoulli_relative_entropy,
    bounded_mean_lower_tail_pvalue,
    calibrate_source_risk_threshold,
    calibrate_source_risk_threshold_fixed_sequence,
    threshold_grid_from_training_scores,
)


def test_threshold_grid_is_deterministic_bounded_and_outcome_free():
    scores = [float(value) for value in range(20)]
    assert threshold_grid_from_training_scores(scores, max_thresholds=5) == [
        19.0,
        14.0,
        9.0,
        5.0,
        0.0,
    ]
    assert threshold_grid_from_training_scores(reversed(scores), max_thresholds=5) == [
        19.0,
        14.0,
        9.0,
        5.0,
        0.0,
    ]


def test_bounded_mean_kl_pvalue_has_expected_endpoints():
    assert bernoulli_relative_entropy(0.0, 0.1) == pytest.approx(-math.log(0.9))
    assert bounded_mean_lower_tail_pvalue(
        0.0,
        null_mean=0.1,
        n_samples=10,
    ) == pytest.approx(0.9**10)
    assert bounded_mean_lower_tail_pvalue(
        0.1,
        null_mean=0.1,
        n_samples=10,
    ) == 1.0


def test_bounded_mean_kl_rejection_event_is_conservative_at_binary_null():
    n_samples = 20
    null_mean = 0.2
    alpha = 0.05
    rejection_probability = 0.0
    for successes in range(n_samples + 1):
        p_value = bounded_mean_lower_tail_pvalue(
            successes / n_samples,
            null_mean=null_mean,
            n_samples=n_samples,
        )
        if p_value <= alpha:
            rejection_probability += (
                math.comb(n_samples, successes)
                * null_mean**successes
                * (1.0 - null_mean) ** (n_samples - successes)
            )
    assert rejection_probability <= alpha


def test_source_aggregation_prevents_question_duplication_from_changing_risk():
    rows = [
        *[
            AcquisitionCalibrationRow(
                source_id="many-questions",
                score=1.0,
                gain=-1.0,
            )
            for _ in range(20)
        ],
        AcquisitionCalibrationRow(
            source_id="one-question",
            score=1.0,
            gain=0.0,
        ),
    ]
    report = calibrate_source_risk_threshold(
        rows,
        [0.5],
        constraints=[RiskConstraint("induced_harm", 0.25)],
        family_error=0.1,
        min_source_call_rate=0.0,
        min_source_utility=-1.0,
    )
    risk = report["candidates"][0]["risks"]["induced_harm"]
    assert risk["source_balanced_mean"] == pytest.approx(0.5)
    assert risk["pooled_decision_mean"] == pytest.approx(20.0 / 21.0)
    assert not risk["passed"]


def test_calibration_selects_safe_non_degenerate_threshold():
    rows = []
    for index in range(400):
        if index < 80:
            score = 1.0
            gain = 0.2
        else:
            score = 0.0
            gain = -0.2
        rows.append(
            AcquisitionCalibrationRow(
                source_id=f"source-{index:04d}",
                score=score,
                gain=gain,
            )
        )
    report = calibrate_source_risk_threshold(
        rows,
        [1.0, 0.0],
        constraints=[
            RiskConstraint("induced_harm", 0.03),
            RiskConstraint("net_negative_call_mass", 0.05),
        ],
        family_error=0.05,
        min_source_call_rate=0.1,
        min_source_utility=0.0,
    )
    assert report["hypothesis_count"] == 4
    assert report["selection_status"] == "selected_non_degenerate_safe_threshold"
    assert report["selected_threshold"] == 1.0
    selected = report["selected"]
    assert selected["source_call_rate"] == pytest.approx(0.2)
    assert selected["source_utility"] == pytest.approx(0.03)
    assert selected["risk_accepted"]
    unsafe = next(
        candidate for candidate in report["candidates"] if candidate["threshold"] == 0.0
    )
    assert not unsafe["risk_accepted"]


def test_zero_call_answer_now_does_not_satisfy_non_degenerate_selection():
    rows = [
        AcquisitionCalibrationRow(
            source_id=f"source-{index:04d}",
            score=0.0,
            gain=-0.5,
        )
        for index in range(300)
    ]
    report = calibrate_source_risk_threshold(
        rows,
        [0.0],
        constraints=[RiskConstraint("negative_net_value", 0.02)],
        min_source_call_rate=0.01,
        min_source_utility=0.0,
    )
    assert report["answer_now"]["risk_accepted"]
    assert report["answer_now"]["source_call_rate"] == 0.0
    assert report["selected_threshold"] is None
    assert report["selection_status"] == "no_non_degenerate_safe_threshold"


def test_calibration_rejects_unfrozen_or_invalid_contracts():
    row = AcquisitionCalibrationRow("source-a", score=0.0, gain=0.0)
    other = AcquisitionCalibrationRow("source-b", score=0.0, gain=0.0)
    constraint = RiskConstraint("induced_harm", 0.1)
    with pytest.raises(ValueError, match="unique"):
        calibrate_source_risk_threshold(
            [row, other],
            [0.0, 0.0],
            constraints=[constraint],
        )
    with pytest.raises(ValueError, match="frozen maximum"):
        calibrate_source_risk_threshold(
            [
                AcquisitionCalibrationRow("source-a", 0.0, 0.0, tool_cost=2.0),
                AcquisitionCalibrationRow("source-b", 0.0, 0.0, tool_cost=2.0),
            ],
            [0.0],
            constraints=[constraint],
            max_tool_cost=1.0,
        )


def test_fixed_sequence_stops_at_first_joint_risk_failure():
    rows = []
    for index in range(1000):
        if index < 100:
            score = 1.5
            gain = 0.2
        else:
            score = 0.5
            gain = -0.2
        rows.append(
            AcquisitionCalibrationRow(
                source_id=f"source-{index:04d}",
                score=score,
                gain=gain,
            )
        )
    report = calibrate_source_risk_threshold_fixed_sequence(
        rows,
        [2.0, 1.0, 0.0, -1.0],
        constraints=[
            RiskConstraint("induced_harm", 0.03),
            RiskConstraint("net_negative_call_mass", 0.05),
        ],
        family_error=0.05,
        min_source_call_rate=0.05,
        min_source_utility=0.0,
    )
    assert report["method"] == "fixed_sequence_bounded_mean_kl_ltt_v1"
    assert report["adjusted_p_cutoff"] == pytest.approx(0.025)
    assert report["tested_threshold_count"] == 3
    assert report["stopping_threshold"] == 0.0
    assert report["untested_thresholds"] == [-1.0]
    assert report["selected_threshold"] == 1.0
    assert report["candidates"][0]["source_call_rate"] == 0.0
    assert report["candidates"][1]["risk_accepted"]
    assert not report["candidates"][2]["risk_accepted"]


def test_fixed_sequence_requires_strict_to_permissive_order():
    rows = [
        AcquisitionCalibrationRow("source-a", score=1.0, gain=0.0),
        AcquisitionCalibrationRow("source-b", score=0.0, gain=0.0),
    ]
    with pytest.raises(ValueError, match="strictly descending"):
        calibrate_source_risk_threshold_fixed_sequence(
            rows,
            [0.0, 1.0],
            constraints=[RiskConstraint("induced_harm", 0.1)],
        )
    with pytest.raises(ValueError, match="strictly descending"):
        calibrate_source_risk_threshold_fixed_sequence(
            rows,
            [1.0, 1.0],
            constraints=[RiskConstraint("induced_harm", 0.1)],
        )
