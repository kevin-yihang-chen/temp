from __future__ import annotations

import pytest

from beyond_entropy.dataset import group_by_decision
from beyond_entropy.information_set_retrospective import (
    decide_n5_calibration_opening,
    evaluate_information_set_retrospective,
)
from beyond_entropy.simulate import simulate_counterfactual_dataset


def _predictions(records):
    grouped = group_by_decision(records)
    keys = sorted(grouped)
    lower_actions = {}
    higher_actions = {}
    lower_scores = {}
    higher_scores = {}
    for index, key in enumerate(keys):
        zooms = sorted(
            (record for record in grouped[key] if record.action_type == "ZOOM"),
            key=lambda record: record.action_id,
        )
        lower_actions[key] = zooms[0].action_id
        higher_actions[key] = max(
            zooms,
            key=lambda record: (record.delta_success, record.action_id),
        ).action_id
        lower_scores[key] = float(index)
        higher_scores[key] = float(len(keys) - index)
    return lower_actions, lower_scores, higher_actions, higher_scores


def test_retrospective_audit_reports_paired_matched_budget_and_cost_bounds():
    records = simulate_counterfactual_dataset(
        n_states=80,
        num_candidates=4,
        questions_per_image=2,
        seed=911,
    )
    lower_actions, lower_scores, higher_actions, higher_scores = _predictions(records)
    report = evaluate_information_set_retrospective(
        records,
        lower_actions=lower_actions,
        lower_scores=lower_scores,
        lower_threshold=40.0,
        higher_actions=higher_actions,
        higher_scores=higher_scores,
        higher_threshold=40.0,
        matched_call_rate=0.05,
        higher_information_extra_costs=(0.0, 0.01),
        bootstrap_resamples=40,
        bootstrap_seed=911,
    )

    assert report["n_decisions"] == 80
    assert report["n_sources"] == 40
    assert report["action_ids"] == ["zoom-0", "zoom-1", "zoom-2", "zoom-3"]
    assert report["matched_budget"]["target_calls"] == 4
    assert report["matched_budget"]["selection_uses_outcomes"] is False
    assert report["source_bootstrap"]["n_resamples"] == 40
    assert report["source_bootstrap"]["confidence_level"] == 0.975
    source = report["source_balanced"]
    assert source["higher_minus_lower_matched_utility"] == pytest.approx(
        source["higher_matched_utility"] - source["lower_matched_utility"]
    )
    assert source["higher_matched_utility_extra_cost_0"] == pytest.approx(
        source["higher_matched_utility"]
    )
    assert source["higher_matched_utility_extra_cost_0p01"] == pytest.approx(
        source["higher_matched_utility"] - 0.01
    )
    assert (
        source["ug_style_exhaustive_entropy_four_cost_utility"]
        <= source["post_action_entropy_single_execution_idealized_utility"]
    )
    assert report["baseline_contract"][
        "ug_style_exhaustive_charges_all_candidate_costs"
    ]


def test_retrospective_audit_rejects_prediction_coverage_drift():
    records = simulate_counterfactual_dataset(
        n_states=40,
        num_candidates=4,
        questions_per_image=2,
        seed=919,
    )
    lower_actions, lower_scores, higher_actions, higher_scores = _predictions(records)
    lower_actions.pop(next(iter(lower_actions)))
    with pytest.raises(ValueError, match="do not exactly cover"):
        evaluate_information_set_retrospective(
            records,
            lower_actions=lower_actions,
            lower_scores=lower_scores,
            lower_threshold=20.0,
            higher_actions=higher_actions,
            higher_scores=higher_scores,
            higher_threshold=20.0,
            bootstrap_resamples=10,
        )


def test_retrospective_audit_rejects_unordered_or_negative_extra_costs():
    records = simulate_counterfactual_dataset(
        n_states=40,
        num_candidates=4,
        questions_per_image=2,
        seed=929,
    )
    lower_actions, lower_scores, higher_actions, higher_scores = _predictions(records)
    with pytest.raises(ValueError, match="extra costs"):
        evaluate_information_set_retrospective(
            records,
            lower_actions=lower_actions,
            lower_scores=lower_scores,
            lower_threshold=20.0,
            higher_actions=higher_actions,
            higher_scores=higher_scores,
            higher_threshold=20.0,
            higher_information_extra_costs=(0.0, -0.01),
            bootstrap_resamples=10,
        )


def _gate_evaluation(higher: float, difference: float, ci_low: float):
    return {
        "source_balanced": {
            "higher_matched_utility": higher,
            "higher_minus_lower_matched_utility": difference,
        },
        "source_bootstrap": {
            "metrics": {
                "higher_matched_utility": {"ci_low": ci_low},
                "higher_minus_lower_matched_utility": {"ci_low": ci_low},
            }
        },
    }


def test_n5_gate_requires_information_factorial_and_material_effect():
    gate = decide_n5_calibration_opening(
        _gate_evaluation(higher=0.01, difference=0.002, ci_low=0.001),
        minimum_material_utility=0.001,
        screenqa_higher_minus_lower_utility=0.002,
        screenqa_higher_has_safe_non_degenerate_threshold=True,
        exact_registered_information_sets_available=False,
        same_method_factorial_available=False,
    )
    assert gate["passed"] is False
    assert gate["decision"] == (
        "n5_current_information_boundary_candidate_not_supported_before_calibration"
    )
    assert gate["checks"]["higher_minus_lower_matched_at_least_minimum"]
    assert not gate["checks"]["exact_registered_information_sets_available"]


def test_n5_gate_passes_only_when_every_registered_condition_passes():
    gate = decide_n5_calibration_opening(
        _gate_evaluation(higher=0.01, difference=0.002, ci_low=0.001),
        minimum_material_utility=0.001,
        screenqa_higher_minus_lower_utility=0.002,
        screenqa_higher_has_safe_non_degenerate_threshold=True,
        exact_registered_information_sets_available=True,
        same_method_factorial_available=True,
    )
    assert gate["passed"] is True
    assert gate["decision"] == (
        "n5_necessary_condition_passed_write_calibration_protocol"
    )
    assert all(gate["checks"].values())
