import pytest

import beyond_entropy.semantic as semantic
from beyond_entropy.dataset import group_by_decision
from beyond_entropy.qwen_semantic import (
    reshape_merged_visual_tokens,
    semantic_decision_from_records,
    validate_semantic_feature_dataset,
)
from beyond_entropy.semantic_training import (
    PrecomputedGainPolicy,
    cross_validated_linear_predictions,
    fit_affine_gain_calibration,
    grouped_kfold_records,
)
from beyond_entropy.simulate import simulate_counterfactual_dataset


def test_semantic_head_has_a_clear_optional_dependency_boundary():
    if semantic.torch is None:
        with pytest.raises(RuntimeError, match="requires PyTorch"):
            semantic.SemanticGainHead(
                question_dim=8,
                visual_dim=8,
                state_signal_dim=2,
            )
        return

    torch = semantic.torch
    head = semantic.SemanticGainHead(
        question_dim=8,
        visual_dim=8,
        state_signal_dim=2,
        hidden_dim=16,
    )
    tokens = torch.randn(2, 4, 4, 8)
    boxes = torch.tensor(
        [
            [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]],
            [[0.0, 0.0, 1.0, 1.0], [0.25, 0.25, 0.75, 0.75]],
        ]
    )
    regions = semantic.roi_pool_spatial_tokens(tokens, boxes)
    gains = head(
        question_embedding=torch.randn(2, 8),
        global_visual_embedding=tokens.mean(dim=(1, 2)),
        region_embeddings=regions,
        bboxes=boxes,
        state_signals=torch.randn(2, 2),
    )
    assert gains.shape == (2, 2)

    success_head = semantic.CounterfactualSuccessHead(
        question_dim=8,
        visual_dim=8,
        state_signal_dim=2,
        hidden_dim=16,
    )
    baseline_logits, action_logits = success_head(
        question_embedding=torch.randn(2, 8),
        global_visual_embedding=tokens.mean(dim=(1, 2)),
        region_embeddings=regions,
        bboxes=boxes,
        state_signals=torch.randn(2, 2),
    )
    assert baseline_logits.shape == (2,)
    assert action_logits.shape == (2, 2)


def test_qwen_merged_tokens_restore_raster_grid():
    if semantic.torch is None:
        pytest.skip("PyTorch is optional")
    torch = semantic.torch
    merged = torch.arange(6 * 3, dtype=torch.float32).reshape(6, 3)
    restored = reshape_merged_visual_tokens(
        merged,
        torch.tensor([1, 4, 6]),
        spatial_merge_size=2,
    )
    assert restored.shape == (2, 3, 3)
    assert torch.equal(restored.reshape(6, 3), merged)


def test_label_free_semantic_decision_is_enforced_for_frozen_inference():
    torch = pytest.importorskip("torch")
    records = simulate_counterfactual_dataset(n_states=1, num_candidates=4, seed=7)
    siblings = next(iter(group_by_decision(records).values()))
    zooms = sorted(
        (record for record in siblings if record.action_type == "ZOOM"),
        key=lambda record: record.action_id,
    )
    encoded = {
        "question_embedding": torch.randn(8),
        "global_visual_embedding": torch.randn(8),
        "region_embeddings": torch.randn(4, 8),
        "bboxes": torch.tensor(
            [record.candidate_bbox.to_list() for record in zooms],
            dtype=torch.float32,
        ),
        "visual_grid_hw": [2, 2],
    }
    decision = semantic_decision_from_records(
        siblings,
        encoded,
        include_outcomes=False,
    )
    assert "success_before" not in decision
    assert "success_after" not in decision
    payload = {
        "format_version": 1,
        "metadata": {"outcomes_included": False},
        "decisions": [decision],
    }
    validate_semantic_feature_dataset(payload, records, require_outcomes=False)
    decision["success_before"] = 0.0
    with pytest.raises(ValueError, match="contain labels"):
        validate_semantic_feature_dataset(payload, records, require_outcomes=False)


def test_affine_gain_calibration_is_monotone():
    slope, intercept = fit_affine_gain_calibration(
        [-1.0, 0.0, 1.0],
        [-0.5, 0.0, 0.5],
    )
    assert slope == pytest.approx(0.5)
    assert intercept == pytest.approx(0.0)

    flat_slope, flat_intercept = fit_affine_gain_calibration(
        [2.0, 2.0],
        [0.0, 1.0],
    )
    assert flat_slope == 0.0
    assert flat_intercept == pytest.approx(0.5)


def test_precomputed_gain_policy_stops_or_selects_without_labels():
    records = simulate_counterfactual_dataset(
        n_states=2,
        num_candidates=2,
        questions_per_image=1,
        seed=3,
    )
    siblings = next(iter(group_by_decision(records).values()))
    zooms = [record for record in siblings if record.action_type == "ZOOM"]
    positive = {
        (record.state_id, record.replicate_id, record.action_id): gain
        for record, gain in zip(zooms, (0.02, 0.2))
    }
    selected = PrecomputedGainPolicy(
        positive,
        lambda_cost=0.05,
        name="semantic",
    ).select(siblings)
    assert selected.selected.action_id == zooms[1].action_id
    assert selected.tool_calls == 1

    stopped = PrecomputedGainPolicy(
        positive,
        lambda_cost=0.5,
        name="semantic",
    ).select(siblings)
    assert stopped.selected.action_type == "ANSWER"
    assert stopped.tool_calls == 0


def test_grouped_oof_predictions_cover_actions_without_image_leakage():
    records = simulate_counterfactual_dataset(
        n_states=12,
        num_candidates=2,
        questions_per_image=2,
        seed=9,
    )
    folds = grouped_kfold_records(
        records,
        group="image_id",
        n_folds=3,
        seed=4,
    )
    validation_keys = []
    for training, validation in folds:
        assert {record.image_id for record in training}.isdisjoint(
            {record.image_id for record in validation}
        )
        validation_keys.extend(
            (record.state_id, record.replicate_id, record.action_id)
            for record in validation
            if record.action_type == "ZOOM"
        )
    expected = {
        (record.state_id, record.replicate_id, record.action_id)
        for record in records
        if record.action_type == "ZOOM"
    }
    assert set(validation_keys) == expected
    assert len(validation_keys) == len(expected)
    predictions = cross_validated_linear_predictions(
        records,
        group="image_id",
        n_folds=3,
        seed=4,
    )
    assert set(predictions) == expected
