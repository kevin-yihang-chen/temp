import pytest

import beyond_entropy.semantic as semantic


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
