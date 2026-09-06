from scripts.evaluate_cv_method_stage import (
    factorized_phase_b_decision,
    phase_b_decision,
)


def benchmark(delta_baseline, delta_outcome):
    return {
        "primary_comparisons": {
            "counterfactual_minus_strongest_uncertainty": {
                "accuracy": {"observed_delta": delta_baseline},
            },
            "counterfactual_minus_outcome_only": {
                "accuracy": {"observed_delta": delta_outcome},
            },
        }
    }


def test_phase_b_go_requires_one_above_one_point_and_other_above_minus_one():
    decision, _ = phase_b_decision({
        "chartqa": benchmark(.011, .02),
        "docvqa": benchmark(-.009, .01),
    })
    assert decision == "PHASE_B_GO"
    decision, _ = phase_b_decision({
        "chartqa": benchmark(.01, .02),
        "docvqa": benchmark(0, .01),
    })
    assert decision == "PHASE_B_NO_GO"


def test_phase_b_stops_when_counterfactual_never_beats_outcome_control():
    decision, reason = phase_b_decision({
        "chartqa": benchmark(.02, 0),
        "docvqa": benchmark(.005, -.01),
    })
    assert decision == "PHASE_B_NO_GO"
    assert "outcome-only" in reason


def factorized_benchmark(delta_baseline, delta_outcome):
    return {
        "primary_comparisons": {
            "factorized_potential_outcomes_minus_strongest_uncertainty": {
                "accuracy": {"observed_delta": delta_baseline},
            },
            "factorized_potential_outcomes_minus_outcome_only": {
                "accuracy": {"observed_delta": delta_outcome},
            },
        }
    }


def test_factorized_phase_b_requires_baseline_margin_cross_domain_safety_and_control_gain():
    decision, _ = factorized_phase_b_decision({
        "chartqa": factorized_benchmark(.011, .02),
        "docvqa": factorized_benchmark(-.004, -.005),
    })
    assert decision == "FACTORIZED_PHASE_B_GO"
    decision, _ = factorized_phase_b_decision({
        "chartqa": factorized_benchmark(.011, -.02),
        "docvqa": factorized_benchmark(-.004, .005),
    })
    assert decision == "FACTORIZED_PHASE_B_NO_GO"
