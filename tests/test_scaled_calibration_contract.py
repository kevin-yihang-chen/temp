import pytest

from scripts.calibrate_scaled_textvqa_action_value import (
    _validate_preregistered_ranker_model,
)


def _model():
    return {
        "model_type": "source_crossfit_pairwise_ranker_call_value_v1",
        "feature_mode": "semantic-context",
        "lambda_cost": 0.05,
        "seed": 20260828,
        "n_folds": 5,
        "selected_ranker_c": 0.1,
        "selected_call_alpha": 10.0,
        "threshold_grid": [0.2, 0.1],
        "calibrated_threshold": None,
        "training_provenance": {
            "risk_calibration_outcomes_used": False,
            "formal_outcomes_used": False,
        },
    }


def test_preregistered_ranker_is_accepted_before_calibration_outcomes_open():
    assert _validate_preregistered_ranker_model(_model()) == [0.2, 0.1]


def test_wrong_feature_mode_is_rejected_before_calibration_outcomes_open():
    model = _model()
    model["feature_mode"] = "hybrid-context-semantic"
    with pytest.raises(ValueError, match="feature_mode"):
        _validate_preregistered_ranker_model(model)


def test_model_with_held_out_outcome_provenance_is_rejected():
    model = _model()
    model["training_provenance"]["risk_calibration_outcomes_used"] = True
    with pytest.raises(ValueError, match="held-out"):
        _validate_preregistered_ranker_model(model)
