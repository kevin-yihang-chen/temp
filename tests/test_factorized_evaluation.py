from __future__ import annotations

import pytest

from beyond_entropy.action_value import (
    fit_multidomain_factorized_action_value_model,
    predict_frozen_factorized_action_values,
)
from beyond_entropy.factorized_evaluation import (
    evaluate_factorized_risk_controlled_policy,
)
from beyond_entropy.simulate import simulate_counterfactual_dataset


def _calibrated_factorized_model(records):
    _, model = fit_multidomain_factorized_action_value_model(
        {"textvqa": records},
        alpha_values=(1.0,),
        seed=71,
    )
    _, scores = predict_frozen_factorized_action_values(model, records)
    threshold = sorted(scores.values())[len(scores) // 2]
    model["threshold"] = threshold
    model["risk_calibration"] = {
        "selection_status": "selected_non_degenerate_safe_threshold",
        "selected_threshold": threshold,
    }
    return model


def test_factorized_formal_evaluation_reports_frozen_primary_and_baselines():
    records = simulate_counterfactual_dataset(
        n_states=120,
        num_candidates=4,
        questions_per_image=2,
        seed=71,
    )
    model = _calibrated_factorized_model(records)
    report = evaluate_factorized_risk_controlled_policy(
        model,
        records,
        bootstrap_resamples=50,
        bootstrap_seed=71,
    )
    assert report["n_sources"] == 60
    assert report["n_decisions"] == 120
    assert report["source_bootstrap"]["confidence_level"] == 0.975
    assert report["source_bootstrap"]["n_resamples"] == 50
    assert report["threshold"] == model["threshold"]
    assert set(report["pass_rule"]) == {
        "source_utility_positive",
        "source_utility_97_5pct_ci_low_positive",
        "question_weighted_utility_positive",
        "source_call_rate_at_least_0_01",
    }
    assert report["source_balanced"]["oracle_utility"] >= 0.0
    assert report["baselines"]["post_action_entropy_is_pre_action"] is False
    assert report["baselines"]["post_action_entropy_single_execution_is_idealized"]
    assert report["baselines"]["ug_style_exhaustive_candidate_count"] == 4
    assert report["baselines"][
        "ug_style_exhaustive_search_charged_all_candidate_costs"
    ]
    assert "ug_style_exhaustive_entropy_source_utility" in report["baselines"]
    assert (
        report["baselines"]["ug_style_exhaustive_entropy_source_utility"]
        < report["source_balanced"]["post_action_entropy_always_call_utility"]
    )
    assert report["baselines"]["matched_budget_gate_uses_outcomes"] is False
    assert report["baselines"]["matched_budget_call_count"] == report["selection"][
        "calls"
    ]
    assert report["baselines"]["matched_budget_question_call_rate"] == pytest.approx(
        report["question_weighted"]["call"]
    )
    assert "matched_budget_entropy_gate_source_utility_learned_crop" in report[
        "baselines"
    ]
    assert set(report["baselines"]["fixed_crop_source_utility_always_call"]) == {
        "zoom-0",
        "zoom-1",
        "zoom-2",
        "zoom-3",
    }
    assert set(report["baselines"]["fixed_crop_source_utility_entropy_gate"]) == {
        "zoom-0",
        "zoom-1",
        "zoom-2",
        "zoom-3",
    }
    assert 0.0 <= report["selection"]["unnecessary_call_rate"] <= 1.0


def test_factorized_formal_evaluation_rejects_threshold_drift():
    records = simulate_counterfactual_dataset(
        n_states=80,
        num_candidates=4,
        questions_per_image=2,
        seed=73,
    )
    model = _calibrated_factorized_model(records)
    model["risk_calibration"]["selected_threshold"] = (
        float(model["threshold"]) + 1.0
    )
    with pytest.raises(ValueError, match="does not match"):
        evaluate_factorized_risk_controlled_policy(
            model,
            records,
            bootstrap_resamples=20,
        )


def test_factorized_formal_evaluation_rejects_failed_calibration():
    records = simulate_counterfactual_dataset(
        n_states=80,
        num_candidates=4,
        questions_per_image=2,
        seed=79,
    )
    model = _calibrated_factorized_model(records)
    model["risk_calibration"]["selection_status"] = "answer_now"
    with pytest.raises(ValueError, match="successful risk calibration"):
        evaluate_factorized_risk_controlled_policy(
            model,
            records,
            bootstrap_resamples=20,
        )
