import pytest

from beyond_entropy.dataset import group_by_decision
from beyond_entropy.metrics import evaluate_policy
from beyond_entropy.rescue_gate import (
    _grouped_crossfit_records,
    PrecomputedRescueGatePolicy,
    aggregate_rescue_gate_splits,
    tune_rescue_gate_threshold,
)
from beyond_entropy.simulate import simulate_counterfactual_dataset


def test_rescue_gate_threshold_prefers_validation_utility_and_fewer_calls():
    threshold, utility, tool_rate = tune_rescue_gate_threshold(
        [0.1, 0.2, 0.3],
        [-0.05, 0.2, -0.1],
    )
    assert 0.1 < threshold < 0.2
    assert utility == pytest.approx(0.1 / 3.0)
    assert tool_rate == pytest.approx(2.0 / 3.0)


def test_precomputed_rescue_gate_is_evaluable_without_action_labels_in_scores():
    records = simulate_counterfactual_dataset(n_states=8, num_candidates=4, seed=21)
    scores = {key: float(index) for index, key in enumerate(sorted(group_by_decision(records)))}
    result = evaluate_policy(
        records,
        PrecomputedRescueGatePolicy(scores, threshold=4.0),
        lambda_cost=0.05,
    )
    assert result["tool_use_rate"] == 0.5
    assert result["avg_tool_calls"] == 0.5


def test_rescue_gate_split_aggregation_does_not_claim_independent_ci():
    reports = [
        {
            "seed": 3,
            "policy_result": {
                "accuracy_gain": 0.01,
                "tool_use_rate": 0.1,
                "mean_policy_utility": 0.005,
            },
        },
        {
            "seed": 11,
            "policy_result": {
                "accuracy_gain": 0.0,
                "tool_use_rate": 0.05,
                "mean_policy_utility": -0.0025,
            },
        },
    ]
    aggregate = aggregate_rescue_gate_splits(reports)
    assert aggregate["positive_utility_splits"] == 1
    assert aggregate["mean_policy_utility"]["mean"] == pytest.approx(0.00125)
    assert "not independent" in aggregate["scientific_status"]


def test_grouped_crossfit_never_leaks_image_groups():
    records = simulate_counterfactual_dataset(
        n_states=12,
        num_candidates=4,
        questions_per_image=2,
        seed=33,
    )
    folds = _grouped_crossfit_records(
        records,
        split_group="image_id",
        n_folds=3,
        seed=5,
    )
    validation_keys = set()
    for training, validation in folds:
        train_images = {record.image_id for record in training}
        validation_images = {record.image_id for record in validation}
        assert train_images.isdisjoint(validation_images)
        validation_keys.update(group_by_decision(validation))
    assert validation_keys == set(group_by_decision(records))
