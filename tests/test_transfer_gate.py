from beyond_entropy.dataset import group_by_decision
from beyond_entropy.simulate import simulate_counterfactual_dataset
import pytest

from beyond_entropy.transfer_gate import (
    fit_factorized_context_transfer,
    threshold_for_target_rate,
)


def test_target_rate_threshold_uses_scores_but_not_labels():
    threshold = threshold_for_target_rate([0.1, 0.4, 0.2, 0.3], 0.5)
    assert threshold == pytest.approx(0.25)
    assert sum(score >= threshold for score in [0.1, 0.4, 0.2, 0.3]) == 2


def test_factorized_context_transfer_never_tunes_on_target_labels():
    source = simulate_counterfactual_dataset(
        n_states=200,
        num_candidates=4,
        questions_per_image=2,
        seed=13,
    )
    target = simulate_counterfactual_dataset(
        n_states=80,
        num_candidates=4,
        questions_per_image=2,
        seed=27,
    )
    target_strata = {
        siblings[0].state_id: ("first" if index < 40 else "second")
        for index, siblings in enumerate(group_by_decision(target).values())
    }
    report, model = fit_factorized_context_transfer(
        source,
        target,
        c_values=(0.1,),
        target_strata=target_strata,
        bootstrap_resamples=20,
        seed=4,
    )
    assert report["target_decisions"] == 80
    assert report["policies"]["factorized_context_transfer"]["n_decisions"] == 80
    assert report["policies"]["factorized_context_quantile_transfer"][
        "n_decisions"
    ] == 80
    assert sum(value["n_decisions"] for value in report["strata"].values()) == 80
    assert model["model_type"] == "factorized_context_cross_benchmark_transfer"
