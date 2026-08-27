import pytest

import beyond_entropy.semantic as semantic
from beyond_entropy.dataset import group_by_decision
from beyond_entropy.qwen_semantic import reshape_merged_visual_tokens
from beyond_entropy.semantic_training import (
    PrecomputedGainPolicy,
    fit_affine_gain_calibration,
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
