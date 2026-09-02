from __future__ import annotations

import numpy as np
import pytest

from beyond_entropy.infographicvqa_literature_attention_evaluation import (
    _bootstrap_mean_interval,
    _corrected_paired_differences,
    _corrected_point_intervals,
    _spearman,
)


def _aggregate(values: list[float]) -> dict:
    return {
        "source_values": {
            f"source-{index}": {"utility": value} for index, value in enumerate(values)
        }
    }


def test_corrected_intervals_use_registered_central_97_5_percent() -> None:
    sources = ["source-0", "source-1"]
    bootstrap = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32)
    primary = _aggregate([1.0, 3.0])
    comparator = _aggregate([0.0, 2.0])
    interval = _bootstrap_mean_interval(primary, sources, bootstrap)
    differences = _corrected_paired_differences(
        primary=primary,
        comparators={"comparator": comparator},
        sources=sources,
        bootstrap_indices=bootstrap,
    )
    assert interval["confidence_level"] == 0.975
    assert interval["point_estimate"] == 2.0
    assert interval["ci_low"] == pytest.approx(1.0375)
    assert interval["ci_high"] == pytest.approx(2.9625)
    assert differences["comparator"]["point_estimate"] == 1.0
    assert differences["comparator"]["ci_low"] == 1.0
    assert differences["comparator"]["ci_high"] == 1.0
    batched = _corrected_point_intervals(
        aggregates={"primary": primary, "comparator": comparator},
        variants=("primary",),
        comparators=("comparator",),
        sources=sources,
        bootstrap_indices=bootstrap,
    )
    assert batched["primary"]["utility"] == interval
    assert batched["primary"]["differences"] == differences


def test_registered_spearman_handles_ties_and_constant_fields() -> None:
    assert _spearman([1.0, 2.0, 2.0, 4.0], [4.0, 3.0, 3.0, 1.0]) == -1.0
    assert _spearman([1.0, 1.0], [0.0, 1.0]) is None
