from scripts.evaluate_cv_method_stage import phase_b_decision


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
