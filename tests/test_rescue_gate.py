from dataclasses import replace

import pytest

from beyond_entropy.dataset import group_by_decision
from beyond_entropy.metrics import evaluate_policy
from beyond_entropy.rescue_gate import (
    _grouped_crossfit_records,
    PrecomputedActionGatePolicy,
    PrecomputedRescueGatePolicy,
    aggregate_rescue_gate_splits,
    fit_nested_oof_entropy_gate,
    fit_nested_oof_factorized_rescue_gate,
    fit_nested_oof_rescue_gate,
    fit_nested_oof_two_stage_gate,
    context_quadrant_action_features,
    question_region_attention_features,
    pre_action_context_feature_subset,
    pre_action_context_features,
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


def test_precomputed_action_gate_selects_concrete_crop_or_stops():
    records = simulate_counterfactual_dataset(n_states=2, num_candidates=4, seed=6)
    grouped = group_by_decision(records)
    keys = sorted(grouped)
    zoom = next(record for record in grouped[keys[1]] if record.action_type == "ZOOM")
    policy = PrecomputedActionGatePolicy(
        {keys[0]: None, keys[1]: zoom.action_id},
        name="test_action_gate",
    )
    result = evaluate_policy(records, policy, lambda_cost=0.05)
    assert result["policy"] == "test_action_gate"
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


def test_context_features_use_only_pre_action_text_and_confidence():
    baseline = next(
        record
        for record in simulate_counterfactual_dataset(n_states=2, num_candidates=4, seed=2)
        if record.action_type == "ANSWER"
    )
    baseline = replace(
        baseline,
        question="What percentage had the highest growth?",
        answer_before="42%",
        entropy_before=0.2,
        metadata={"baseline_backend": {"normalized_token_entropies": [0.1, 0.3]}},
    )
    changed_outcomes = replace(
        baseline,
        answer_after="unrelated",
        entropy_after=99.0,
        correct_after=1.0 - baseline.correct_after,
    )
    features = pre_action_context_features(baseline)
    assert features == pre_action_context_features(changed_outcomes)
    assert len(features) == 27
    assert features[4:8] == [1.0, 1.0, 1.0, 0.0]
    assert features[8:14] == pytest.approx([2.0, 0.2, 0.3, 0.1, 0.1, 0.3])


def test_nested_oof_context_gate_evaluates_each_decision_once():
    records = simulate_counterfactual_dataset(
        n_states=80,
        num_candidates=4,
        questions_per_image=2,
        seed=9,
    )
    decisions = {key: {} for key in group_by_decision(records)}
    report, model = fit_nested_oof_rescue_gate(
        records,
        decisions,
        feature_mode="context",
        bootstrap_resamples=20,
        seed=4,
    )
    assert report["n_decisions"] == 80
    assert sum(fold["test_decisions"] for fold in report["folds"]) == 80
    assert report["policy_result"]["n_decisions"] == 80
    assert report["policy_result"]["bootstrap"]["n_decisions"] == 80
    assert report["feature_count"] == 27
    assert len(model["fold_models"]) == 5

    entropy_report = fit_nested_oof_entropy_gate(
        records,
        bootstrap_resamples=20,
        seed=4,
    )
    assert entropy_report["n_decisions"] == 80
    assert sum(fold["test_decisions"] for fold in entropy_report["folds"]) == 80
    assert entropy_report["policy_result"]["n_decisions"] == 80

    factorized_report, factorized_model = fit_nested_oof_factorized_rescue_gate(
        records,
        decisions,
        error_feature_mode="context",
        rescue_feature_mode="context",
        c_values=(0.1,),
        bootstrap_resamples=20,
        seed=4,
    )
    assert factorized_report["n_decisions"] == 80
    assert factorized_report["policy_result"]["n_decisions"] == 80
    assert len(factorized_model["fold_models"]) == 5


def test_nested_oof_two_stage_gate_runs_without_post_action_features():
    torch = pytest.importorskip("torch")
    records = simulate_counterfactual_dataset(
        n_states=120,
        num_candidates=4,
        questions_per_image=2,
        seed=9,
    )
    decisions = {}
    generator = torch.Generator().manual_seed(3)
    for key, siblings in group_by_decision(records).items():
        zooms = sorted(
            (record for record in siblings if record.action_type == "ZOOM"),
            key=lambda record: record.action_id,
        )
        baseline = next(record for record in siblings if record.action_type == "ANSWER")
        decisions[key] = {
            "action_ids": [record.action_id for record in zooms],
            "question_embedding": torch.randn(8, generator=generator),
            "global_visual_embedding": torch.randn(8, generator=generator),
            "region_embeddings": torch.randn(4, 8, generator=generator),
            "bboxes": torch.tensor(
                [record.candidate_bbox.to_list() for record in zooms],
                dtype=torch.float32,
            ),
            "state_signals": torch.tensor([baseline.entropy_before]),
        }
    report, model = fit_nested_oof_two_stage_gate(
        records,
        decisions,
        c_values=(0.1,),
        bootstrap_resamples=20,
        seed=4,
    )
    assert report["n_decisions"] == 120
    assert report["policy_result"]["n_decisions"] == 120
    assert report["action_feature_count"] == 15
    assert len(model["fold_models"]) == 5


def test_attention_fixed_two_stage_gate_uses_frozen_region_ranking():
    torch = pytest.importorskip("torch")
    records = simulate_counterfactual_dataset(
        n_states=120,
        num_candidates=4,
        questions_per_image=2,
        seed=9,
    )
    decisions = {}
    generator = torch.Generator().manual_seed(3)
    for key, siblings in group_by_decision(records).items():
        zooms = sorted(
            (record for record in siblings if record.action_type == "ZOOM"),
            key=lambda record: record.action_id,
        )
        baseline = next(record for record in siblings if record.action_type == "ANSWER")
        decisions[key] = {
            "action_ids": [record.action_id for record in zooms],
            "question_embedding": torch.randn(8, generator=generator),
            "global_visual_embedding": torch.randn(8, generator=generator),
            "region_embeddings": torch.randn(4, 8, generator=generator),
            "bboxes": torch.tensor(
                [record.candidate_bbox.to_list() for record in zooms],
                dtype=torch.float32,
            ),
            "state_signals": torch.tensor([baseline.entropy_before]),
            "question_region_attention": torch.tensor([0.1, 0.2, 0.6, 0.1]),
        }
    report, model = fit_nested_oof_two_stage_gate(
        records,
        decisions,
        action_feature_mode="attention-fixed",
        c_values=(0.1,),
        bootstrap_resamples=20,
        seed=4,
    )
    assert report["n_decisions"] == 120
    assert report["action_feature_count"] == 1
    assert all(fold["selected_action_c"] is None for fold in report["folds"])
    assert all("action_coefficient" not in fold for fold in model["fold_models"])
    exemplar = next(iter(decisions.values()))
    assert question_region_attention_features(exemplar, 2) == pytest.approx([0.6])


def test_context_quadrant_action_features_do_not_use_action_outcomes():
    records = simulate_counterfactual_dataset(n_states=4, num_candidates=4, seed=13)
    siblings = next(iter(group_by_decision(records).values()))
    baseline = next(record for record in siblings if record.action_type == "ANSWER")
    changed_outcomes = replace(
        baseline,
        entropy_after=baseline.entropy_after + 10.0,
        correct_after=1.0 - baseline.correct_after,
    )
    features = context_quadrant_action_features(
        baseline,
        2,
        action_count=4,
    )
    assert len(features) == 139
    assert features == context_quadrant_action_features(
        changed_outcomes,
        2,
        action_count=4,
    )
    assert features != context_quadrant_action_features(
        baseline,
        1,
        action_count=4,
    )


def test_registered_context_feature_ablations_partition_pre_action_signals():
    records = simulate_counterfactual_dataset(n_states=2, num_candidates=4, seed=14)
    baseline = next(record for record in records if record.action_type == "ANSWER")
    assert len(pre_action_context_feature_subset(baseline, "context")) == 27
    assert len(pre_action_context_feature_subset(baseline, "context-uncertainty")) == 6
    assert len(pre_action_context_feature_subset(baseline, "context-text")) == 21
    assert len(pre_action_context_feature_subset(baseline, "context-question")) == 15
    assert len(
        pre_action_context_feature_subset(baseline, "context-answer-uncertainty")
    ) == 12
