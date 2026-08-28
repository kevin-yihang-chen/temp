from __future__ import annotations

from dataclasses import replace

import pytest

from beyond_entropy.action_value import (
    context_geometry_action_features,
    evaluate_frozen_action_value_model,
    evaluate_frozen_factorized_action_value_model,
    fit_multidomain_action_value_model,
    fit_multidomain_factorized_action_value_model,
    normalized_gate_question,
    select_frozen_action_value_actions,
    select_frozen_factorized_action_value_actions,
    semantic_context_action_features,
    spatial_context_geometry_action_features,
    spatial_question_features,
)
from beyond_entropy.dataset import group_by_decision
from beyond_entropy.oof_action_value import fit_oof_factorized_action_value_model
from beyond_entropy.simulate import simulate_counterfactual_dataset


def _namespace(records, namespace):
    return [
        replace(
            record,
            state_id=f"{namespace}:{record.state_id}",
            image_id=f"{namespace}:{record.image_id}",
            source_id=f"{namespace}:{record.source_id}",
        )
        for record in records
    ]


def test_question_normalization_removes_wrappers_but_not_question():
    assert normalized_gate_question(
        "Answer briefly. Question: What is the total?"
    ) == "What is the total?"
    assert normalized_gate_question("What is the total? Answer:") == "What is the total?"


def test_context_geometry_features_ignore_counterfactual_outcomes():
    records = simulate_counterfactual_dataset(n_states=4, num_candidates=4, seed=3)
    siblings = next(iter(group_by_decision(records).values()))
    baseline = next(record for record in siblings if record.action_type == "ANSWER")
    action = next(record for record in siblings if record.action_type == "ZOOM")
    changed = replace(
        action,
        correct_after=1.0 - action.correct_after,
        entropy_after=action.entropy_after + 5.0,
        answer_after="label-only mutation",
    )
    assert context_geometry_action_features(
        baseline,
        action,
    ) == context_geometry_action_features(baseline, changed)


def test_spatial_context_features_are_pre_action_and_geometry_sensitive():
    records = simulate_counterfactual_dataset(n_states=4, num_candidates=4, seed=13)
    siblings = next(iter(group_by_decision(records).values()))
    baseline = next(record for record in siblings if record.action_type == "ANSWER")
    baseline = replace(baseline, question="What word is on the bottom left?")
    zooms = sorted(
        (record for record in siblings if record.action_type == "ZOOM"),
        key=lambda record: record.action_id,
    )
    signals = spatial_question_features(baseline.question)
    assert signals[:4] == [1.0, 0.0, 0.0, 1.0]
    assert spatial_context_geometry_action_features(
        baseline, zooms[0]
    ) != spatial_context_geometry_action_features(baseline, zooms[-1])
    changed = replace(
        zooms[0],
        correct_after=1.0 - zooms[0].correct_after,
        entropy_after=zooms[0].entropy_after + 5.0,
        answer_after="outcome mutation",
    )
    assert spatial_context_geometry_action_features(
        baseline, zooms[0]
    ) == spatial_context_geometry_action_features(baseline, changed)


def test_semantic_context_features_ignore_stored_outcomes():
    torch = pytest.importorskip("torch")
    records = simulate_counterfactual_dataset(n_states=4, num_candidates=4, seed=8)
    siblings = next(iter(group_by_decision(records).values()))
    baseline = next(record for record in siblings if record.action_type == "ANSWER")
    zooms = sorted(
        (record for record in siblings if record.action_type == "ZOOM"),
        key=lambda record: record.action_id,
    )
    decision = {
        "action_ids": [record.action_id for record in zooms],
        "question_embedding": torch.tensor([1.0, 0.0, 0.0]),
        "global_visual_embedding": torch.tensor([0.0, 1.0, 0.0]),
        "region_embeddings": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
        ),
        "bboxes": torch.tensor([record.candidate_bbox.to_list() for record in zooms]),
        "state_signals": torch.tensor([baseline.entropy_before]),
        "success_before": baseline.correct_before,
        "success_after": torch.tensor([record.correct_after for record in zooms]),
    }
    original = semantic_context_action_features(baseline, zooms[0], decision)
    changed = dict(decision)
    changed["success_before"] = 1.0 - baseline.correct_before
    changed["success_after"] = 1.0 - decision["success_after"]
    assert semantic_context_action_features(baseline, zooms[0], changed) == original


def test_multidomain_action_value_model_is_serializable_and_selects_concrete_crops():
    chart = _namespace(
        simulate_counterfactual_dataset(
            n_states=120,
            num_candidates=4,
            questions_per_image=2,
            seed=11,
        ),
        "chart",
    )
    document = _namespace(
        simulate_counterfactual_dataset(
            n_states=100,
            num_candidates=4,
            questions_per_image=2,
            seed=19,
        ),
        "document",
    )
    report, model = fit_multidomain_action_value_model(
        {"chart": chart, "document": document},
        alpha_values=(1.0, 10.0),
        seed=7,
    )
    assert report["domains"] == ["chart", "document"]
    assert report["train_decisions"] + report["validation_decisions"] == 220
    assert model["model_type"] == "multidomain_direct_action_value"
    assert isinstance(model["threshold"], float)
    assert len(model["coefficient"]) == model["feature_count"]
    selected, scores = select_frozen_action_value_actions(model, document)
    assert len(selected) == len(scores) == 100
    assert all(action is None or action.startswith("zoom-") for action in selected.values())
    evaluated = evaluate_frozen_action_value_model(
        model,
        document,
        bootstrap_resamples=20,
    )
    assert evaluated["n_decisions"] == 100


def test_frozen_selection_does_not_read_target_outcomes():
    source = _namespace(
        simulate_counterfactual_dataset(n_states=100, num_candidates=4, seed=5),
        "source",
    )
    auxiliary = _namespace(
        simulate_counterfactual_dataset(n_states=80, num_candidates=4, seed=6),
        "auxiliary",
    )
    _, model = fit_multidomain_action_value_model(
        {"source": source, "auxiliary": auxiliary},
        alpha_values=(10.0,),
        seed=9,
    )
    original, original_scores = select_frozen_action_value_actions(model, auxiliary)
    changed = [
        replace(
            record,
            correct_before=0.25 if record.action_type == "ANSWER" else 0.25,
            correct_after=(
                0.25 if record.action_type == "ANSWER" else 1.0 - record.correct_after
            ),
            entropy_after=record.entropy_after + 2.0,
            answer_after="mutated outcome",
        )
        for record in auxiliary
    ]
    mutated, mutated_scores = select_frozen_action_value_actions(model, changed)
    assert mutated == original
    assert mutated_scores == pytest.approx(original_scores)


def test_factorized_risk_rescue_harm_model_round_trips():
    source = _namespace(
        simulate_counterfactual_dataset(
            n_states=180,
            num_candidates=4,
            questions_per_image=2,
            seed=31,
        ),
        "source",
    )
    auxiliary = _namespace(
        simulate_counterfactual_dataset(
            n_states=160,
            num_candidates=4,
            questions_per_image=2,
            seed=37,
        ),
        "auxiliary",
    )
    report, model = fit_multidomain_factorized_action_value_model(
        {"source": source, "auxiliary": auxiliary},
        alpha_values=(1.0, 10.0),
        seed=12,
    )
    assert report["model_type"] == "multidomain_factorized_action_value"
    assert report["rescue_magnitude"] > 0.0
    assert report["harm_magnitude"] > 0.0
    assert model["state_feature_count"] == len(model["error_coefficient"])
    assert model["action_feature_count"] == len(model["rescue_coefficient"])
    selected, scores = select_frozen_factorized_action_value_actions(
        model, auxiliary
    )
    assert len(selected) == len(scores) == 160
    evaluated = evaluate_frozen_factorized_action_value_model(
        model,
        auxiliary,
        bootstrap_resamples=20,
        bootstrap_confidence=0.9,
    )
    assert evaluated["n_decisions"] == 160
    assert evaluated["bootstrap"]["confidence"] == 0.9


def test_source_grouped_oof_factorized_model_refits_and_round_trips():
    source = _namespace(
        simulate_counterfactual_dataset(
            n_states=180,
            num_candidates=4,
            questions_per_image=2,
            seed=41,
        ),
        "source",
    )
    auxiliary = _namespace(
        simulate_counterfactual_dataset(
            n_states=150,
            num_candidates=4,
            questions_per_image=2,
            seed=43,
        ),
        "auxiliary",
    )
    report, model = fit_oof_factorized_action_value_model(
        {"source": source, "auxiliary": auxiliary},
        n_folds=3,
        alpha_values=(10.0,),
        seed=17,
        bootstrap_resamples=20,
    )
    assert report["training_protocol"] == "source_grouped_oof_v1"
    assert report["development_decisions"] == 330
    assert report["oof_policy_result"]["n_decisions"] == 330
    tail = report["development_tail_risk_diagnostic"]
    assert tail["valid_for_formal_selection"] is False
    assert tail["n_decisions"] == 330
    assert [
        item["target_pooled_call_rate"] for item in tail["requested_thresholds"]
    ] == [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
    assert model["training_protocol"] == "source_grouped_oof_v1"
    selected, scores = select_frozen_factorized_action_value_actions(
        model, auxiliary
    )
    assert len(selected) == len(scores) == 150


def test_hybrid_factorized_model_uses_context_state_and_semantic_actions():
    torch = pytest.importorskip("torch")
    source = _namespace(
        simulate_counterfactual_dataset(n_states=140, num_candidates=4, seed=47),
        "source",
    )
    auxiliary = _namespace(
        simulate_counterfactual_dataset(n_states=120, num_candidates=4, seed=53),
        "auxiliary",
    )

    def semantic_index(records):
        result = {}
        for key, siblings in group_by_decision(records).items():
            baseline = next(
                record for record in siblings if record.action_type == "ANSWER"
            )
            zooms = sorted(
                (record for record in siblings if record.action_type == "ZOOM"),
                key=lambda record: record.action_id,
            )
            result[key] = {
                "action_ids": [record.action_id for record in zooms],
                "question_embedding": torch.tensor([1.0, 0.5, 0.25]),
                "global_visual_embedding": torch.tensor([0.25, 1.0, 0.5]),
                "region_embeddings": torch.tensor(
                    [
                        [
                            record.candidate_bbox.x1,
                            record.candidate_bbox.y1,
                            1.0,
                        ]
                        for record in zooms
                    ]
                ),
                "bboxes": torch.tensor(
                    [record.candidate_bbox.to_list() for record in zooms]
                ),
                "state_signals": torch.tensor([baseline.entropy_before]),
                "success_before": baseline.correct_before,
                "success_after": torch.tensor(
                    [record.correct_after for record in zooms]
                ),
            }
        return result

    semantic = {
        "source": semantic_index(source),
        "auxiliary": semantic_index(auxiliary),
    }
    report, model = fit_multidomain_factorized_action_value_model(
        {"source": source, "auxiliary": auxiliary},
        feature_mode="hybrid-context-semantic",
        semantic_decisions_by_domain=semantic,
        alpha_values=(10.0,),
        seed=23,
    )
    assert report["state_feature_count"] == 27
    assert report["action_feature_count"] == 42
    selected, scores = select_frozen_factorized_action_value_actions(
        model,
        auxiliary,
        semantic_decisions=semantic["auxiliary"],
    )
    assert len(selected) == len(scores) == 120
