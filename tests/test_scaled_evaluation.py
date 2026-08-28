import pytest

from beyond_entropy.scaled_action_value import fit_scaled_pairwise_action_value_model
from beyond_entropy.scaled_evaluation import evaluate_scaled_risk_controlled_policy
from beyond_entropy.simulate import simulate_counterfactual_dataset


def _calibrated_synthetic_model(records):
    _, model = fit_scaled_pairwise_action_value_model(
        records,
        n_folds=3,
        ranker_c_values=(0.1,),
        call_alpha_values=(10.0,),
        max_thresholds=8,
        bootstrap_resamples=20,
        seed=47,
    )
    model["calibrated_threshold"] = min(model["threshold_grid"])
    model["risk_calibration"] = {
        "selection_status": "selected_non_degenerate_safe_threshold",
        "selected_threshold": model["calibrated_threshold"],
    }
    return model


def test_scaled_evaluation_reports_source_primary_and_diagnostics():
    records = simulate_counterfactual_dataset(
        n_states=60,
        num_candidates=4,
        questions_per_image=2,
        seed=47,
    )
    model = _calibrated_synthetic_model(records)
    report = evaluate_scaled_risk_controlled_policy(
        model,
        records,
        bootstrap_resamples=50,
        bootstrap_seed=47,
    )
    assert report["n_sources"] == 30
    assert report["n_decisions"] == 60
    assert report["source_bootstrap"]["confidence_level"] == 0.975
    assert report["source_bootstrap"]["n_resamples"] == 50
    assert set(report["pass_rule"]) == {
        "source_utility_positive",
        "source_utility_97_5pct_ci_low_positive",
        "question_weighted_utility_positive",
        "source_call_rate_at_least_0_01",
    }
    assert report["source_balanced"]["oracle_utility"] >= 0.0
    assert 0.0 <= report["selection"]["unnecessary_call_rate"] <= 1.0
    assert "source_balanced_raw_gain_per_call" in report["selection"]
    assert "question_weighted_raw_gain_per_call" in report["selection"]
    assert set(report["ranking"]["fixed_crop_source_utilities"]) == {
        "zoom-0",
        "zoom-1",
        "zoom-2",
        "zoom-3",
    }


def test_scaled_evaluation_rejects_uncalibrated_model():
    records = simulate_counterfactual_dataset(
        n_states=30,
        num_candidates=4,
        questions_per_image=2,
        seed=53,
    )
    _, model = fit_scaled_pairwise_action_value_model(
        records,
        n_folds=3,
        ranker_c_values=(0.1,),
        call_alpha_values=(10.0,),
        max_thresholds=8,
        bootstrap_resamples=20,
        seed=53,
    )
    with pytest.raises(ValueError, match="non-degenerate calibrated"):
        evaluate_scaled_risk_controlled_policy(model, records, bootstrap_resamples=20)


def test_scaled_evaluation_rejects_threshold_that_differs_from_calibration():
    records = simulate_counterfactual_dataset(
        n_states=30,
        num_candidates=4,
        questions_per_image=2,
        seed=59,
    )
    model = _calibrated_synthetic_model(records)
    model["risk_calibration"]["selected_threshold"] = (
        float(model["calibrated_threshold"]) + 1.0
    )
    with pytest.raises(ValueError, match="does not match"):
        evaluate_scaled_risk_controlled_policy(model, records, bootstrap_resamples=20)
