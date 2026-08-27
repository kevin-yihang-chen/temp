import pytest

from beyond_entropy.question_reembed import masked_hidden_mean


def test_masked_hidden_mean_excludes_padding_tokens():
    torch = pytest.importorskip("torch")
    hidden = torch.tensor(
        [
            [[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]],
            [[2.0, 4.0], [200.0, 200.0], [300.0, 300.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    pooled = masked_hidden_mean(hidden, mask)
    assert pooled.tolist() == [[2.0, 4.0], [2.0, 4.0]]


def test_masked_hidden_mean_rejects_empty_sequences():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="empty token sequence"):
        masked_hidden_mean(torch.zeros(1, 2, 3), torch.zeros(1, 2))
