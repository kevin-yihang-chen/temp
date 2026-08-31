from __future__ import annotations

import pytest

from beyond_entropy.qwen_backend import merge_runtime_measurements


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
