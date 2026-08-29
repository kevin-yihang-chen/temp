from __future__ import annotations

import pytest

from beyond_entropy.docvqa_calibration import calibrate_frozen_candidate_rows
from beyond_entropy.risk_control import AcquisitionCalibrationRow


def _candidate():
    thresholds = [2.0, 1.0, 0.0]
    return {
        "model_type": "multidomain_factorized_action_value",
        "training_protocol": "source_grouped_oof_domain_source_balanced_v2",
        "sample_weighting": "equal_domain_then_equal_source_then_equal_row",
        "feature_mode": "hybrid-context-semantic",
        "seed": 20260829,
        "n_folds": 5,
        "lambda_cost": 0.05,
        "selected_alpha": 1.0,
        "domains": ["docvqa"],
        "state_feature_count": 27,
        "action_feature_count": 46,
        "threshold": None,
        "threshold_grid": thresholds,
        "calibration_contract": {
            "method": "fixed_sequence_bounded_mean_kl_ltt_v1",
            "threshold_order": "strict_to_permissive_descending",
            "threshold_rate_weighting": "equal_source_then_equal_question",
            "target_source_balanced_development_call_rates": [0.01],
            "threshold_summaries": [
                {
                    "threshold": threshold,
                    "source_balanced_development_call_rate": 0.0,
                    "pooled_development_call_rate": 0.0,
                }
                for threshold in thresholds
            ],
            "constraints": [
                {"kind": "induced_harm", "limit": 0.005},
                {"kind": "net_negative_call_mass", "limit": 0.02},
            ],
            "family_error": 0.05,
            "per_step_p_cutoff": 0.025,
            "min_source_call_rate": 0.01,
            "min_source_utility": 0.001,
            "calibration_sources": 2500,
            "formal_sources": 3500,
        },
        "candidate_freeze": {
            "ranker_training_outcomes_used": True,
            "calibration_outcomes_used": False,
            "formal_outcomes_used": False,
        },
    }


def _positive_rows(count: int = 2500):
    return [
        AcquisitionCalibrationRow(
            source_id=f"document-{index:04d}",
            score=1.0 if index < 500 else 0.0,
            gain=0.2 if index < 500 else -0.2,
        )
        for index in range(count)
    ]


def test_docvqa_fixed_sequence_selects_safe_nondegenerate_threshold():
    calibration, model = calibrate_frozen_candidate_rows(
        _candidate(),
        _positive_rows(),
        expected_sources=2500,
        run_provenance={"candidate_sha256": "a" * 64},
    )
    assert calibration["selection_status"] == "selected_non_degenerate_safe_threshold"
    assert calibration["selected_threshold"] == 1.0
    assert calibration["tested_threshold_count"] == 3
    assert calibration["stopping_threshold"] == 0.0
    assert calibration["run"]["ranker_training_outcomes_used"] is True
    assert calibration["run"]["calibration_outcomes_used"] is True
    assert calibration["run"]["formal_outcomes_used"] is False
    assert model["threshold"] == 1.0
    assert model["risk_calibration"]["selected_threshold"] == 1.0


def test_docvqa_fixed_sequence_failure_returns_answer_now():
    rows = [
        AcquisitionCalibrationRow(
            source_id=f"document-{index:04d}",
            score=1.0,
            gain=-0.5,
        )
        for index in range(2500)
    ]
    calibration, model = calibrate_frozen_candidate_rows(
        _candidate(),
        rows,
        expected_sources=2500,
    )
    assert calibration["selection_status"] == "no_non_degenerate_safe_threshold"
    assert calibration["selected_threshold"] is None
    assert calibration["stopping_threshold"] == 1.0
    assert model["threshold"] is None


def test_docvqa_calibration_rejects_source_or_candidate_drift():
    with pytest.raises(ValueError, match="2501 source groups"):
        calibrate_frozen_candidate_rows(
            _candidate(),
            _positive_rows(),
            expected_sources=2501,
        )
    candidate = _candidate()
    candidate["calibration_contract"]["family_error"] = 0.1
    with pytest.raises(ValueError, match="family_error"):
        calibrate_frozen_candidate_rows(
            candidate,
            _positive_rows(),
            expected_sources=2500,
        )
