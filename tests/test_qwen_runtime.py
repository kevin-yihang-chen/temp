from __future__ import annotations

import pytest

from beyond_entropy.qwen_backend import (
    generated_token_statistics,
    merge_runtime_measurements,
)


def test_merge_runtime_measurements_preserves_largest_peaks() -> None:
    previous = {
        "accelerator_name": "NVIDIA H800",
        "parameter_dtype": "torch.bfloat16",
        "peak_allocated_bytes": 10,
        "peak_reserved_bytes": 20,
    }
    current = {
        "accelerator_name": "NVIDIA H800",
        "parameter_dtype": "torch.bfloat16",
        "peak_allocated_bytes": 12,
        "peak_reserved_bytes": 18,
    }
    assert merge_runtime_measurements(previous, current) == {
        **current,
        "peak_allocated_bytes": 12,
        "peak_reserved_bytes": 20,
    }


def test_merge_runtime_measurements_rejects_changed_configuration() -> None:
    with pytest.raises(ValueError, match="configuration changed"):
        merge_runtime_measurements(
            {"accelerator_name": "NVIDIA H800", "peak_allocated_bytes": 10},
            {"accelerator_name": "NVIDIA H100", "peak_allocated_bytes": 10},
        )

    with pytest.raises(ValueError, match="non-negative"):
        merge_runtime_measurements(
            {"accelerator_name": "NVIDIA H800", "peak_allocated_bytes": -1},
            {"accelerator_name": "NVIDIA H800", "peak_allocated_bytes": 10},
        )


def test_generated_token_statistics_tracks_selected_token_probability() -> None:
    torch = pytest.importorskip("torch")
    logits = [
        torch.tensor([[2.0, 1.0, -1.0]]),
        torch.tensor([[0.0, 0.5, 1.5]]),
    ]
    generated = torch.tensor([[0, 2]])
    entropies, log_probabilities = generated_token_statistics(logits, generated)
    assert len(entropies) == len(log_probabilities) == 2
    assert all(0.0 <= value <= 1.0 for value in entropies)
    assert log_probabilities == pytest.approx(
        [
            float(torch.log_softmax(logits[0][0], dim=-1)[0]),
            float(torch.log_softmax(logits[1][0], dim=-1)[2]),
        ]
    )


def test_generated_token_statistics_rejects_unaligned_steps() -> None:
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="aligned sequence"):
        generated_token_statistics([torch.tensor([[1.0, 0.0]])], torch.tensor([[0, 1]]))
