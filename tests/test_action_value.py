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
)
from beyond_entropy.dataset import group_by_decision
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
    )
    assert evaluated["n_decisions"] == 160
