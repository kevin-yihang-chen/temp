import pytest

from scripts.freeze_scaled_textvqa_policy import _validate_policy


def _successful_policy():
    constraints = [
        {"kind": "induced_harm", "limit": 0.005},
        {"kind": "net_negative_call_mass", "limit": 0.02},
    ]
    calibration = {
        "selection_status": "selected_non_degenerate_safe_threshold",
        "selected_threshold": 0.25,
        "method": "bonferroni_bounded_mean_kl_ltt_v1",
        "lambda_cost": 0.05,
        "max_tool_cost": 1.0,
        "family_error": 0.05,
        "hypothesis_count": 4,
        "constraints": constraints,
        "min_source_call_rate": 0.01,
        "min_source_utility": 0.001,
        "selection_objective": "source_call_rate",
        "selected": {
            "risk_accepted": True,
            "source_call_rate": 0.1,
            "source_utility": 0.01,
        },
        "run": {
            "ranker_model_sha256": "a" * 64,
            "formal_outcomes_used": False,
        },
    }
    model = {
        "model_type": "source_crossfit_pairwise_ranker_call_value_v1",
        "feature_mode": "semantic-context",
        "lambda_cost": 0.05,
        "seed": 20260828,
        "n_folds": 5,
        "selected_ranker_c": 0.1,
        "selected_call_alpha": 10.0,
        "threshold_grid": [0.25, 0.0],
        "calibrated_threshold": 0.25,
        "risk_calibration": {
            key: calibration[key]
            for key in (
                "selection_status",
                "selected_threshold",
                "method",
                "constraints",
                "family_error",
                "hypothesis_count",
                "min_source_call_rate",
                "min_source_utility",
                "selection_objective",
            )
        },
    }
    return model, calibration


def test_successful_scaled_policy_can_open_formal_gate():
    model, calibration = _successful_policy()
    _validate_policy(model, calibration, ranker_model_sha256="a" * 64)


def test_answer_now_failure_cannot_open_formal_gate():
    model, calibration = _successful_policy()
    model["calibrated_threshold"] = None
    calibration["selected_threshold"] = None
    calibration["selection_status"] = "no_non_degenerate_safe_threshold"
    with pytest.raises(ValueError, match="cannot open"):
        _validate_policy(model, calibration, ranker_model_sha256="a" * 64)
