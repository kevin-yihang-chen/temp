import pytest

from beyond_entropy.attention_features import normalized_question_region_attention


def test_attention_pooling_prefers_high_mass_candidate_and_normalizes():
    torch = pytest.importorskip("torch")
    grid = torch.tensor(
        [
            [9.0, 9.0, 1.0, 1.0],
            [9.0, 9.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    boxes = torch.tensor(
        [
            [0.0, 0.0, 0.5, 0.5],
            [0.5, 0.0, 1.0, 0.5],
            [0.0, 0.5, 0.5, 1.0],
            [0.5, 0.5, 1.0, 1.0],
        ]
    )
    scores = normalized_question_region_attention(grid, boxes)
    assert scores.shape == (4,)
    assert float(scores.sum()) == pytest.approx(1.0)
    assert int(scores.argmax()) == 0


def test_attention_pooling_rejects_zero_mass():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="positive finite mass"):
        normalized_question_region_attention(
            torch.zeros(2, 2),
            torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        )
