from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from beyond_entropy.infographicvqa_decar import DECAR_ACTION_IDS, DecarDataset
from beyond_entropy.infographicvqa_relative_where import (
    RELATIVE_WHERE_SCHEMA,
    RELATIVE_WHERE_VARIANTS,
    _average_rank_percentiles,
    _ranking_weights,
    fit_relative_where_oof,
)


def _dataset() -> tuple[DecarDataset, dict[str, int]]:
    generator = torch.Generator().manual_seed(7)
    decisions = 20
    embedding_dim = 8
    source_ids = tuple(f"source-{index // 2:02d}" for index in range(decisions))
    outer_folds = {f"source-{index:02d}": index % 5 for index in range(10)}
    loss_gaps = torch.zeros((decisions, 4), dtype=torch.float32)
    task_deltas = torch.zeros((decisions, 4), dtype=torch.float32)
    for index in range(decisions):
        preferred = index % 4
        loss_gaps[index] = torch.tensor((-0.2, -0.1, 0.0, 0.1))
        loss_gaps[index] = torch.roll(loss_gaps[index], preferred)
        task_deltas[index, preferred] = 0.5
        task_deltas[index, (preferred + 1) % 4] = 0.1
    return (
        DecarDataset(
            state_ids=tuple(f"state-{index:03d}" for index in range(decisions)),
            replicate_ids=("replicate-000",) * decisions,
            image_ids=tuple(f"image-{index:03d}" for index in range(decisions)),
            source_ids=source_ids,
            action_ids=(DECAR_ACTION_IDS,) * decisions,
            question=torch.randn(decisions, embedding_dim, generator=generator),
            global_visual=torch.randn(decisions, embedding_dim, generator=generator),
            region=torch.randn(decisions, 4, embedding_dim, generator=generator),
            scalars=torch.randn(decisions, 4, 16, generator=generator),
            loss_gaps=loss_gaps,
            task_deltas=task_deltas,
            correct_before=torch.zeros(decisions),
            entropy_before=torch.linspace(0.0, 1.0, decisions),
            entropy_after=torch.zeros((decisions, 4)),
        ),
        outer_folds,
    )


def test_average_rank_percentiles_preserves_exact_ties() -> None:
    result = _average_rank_percentiles(torch.tensor((1.0, 2.0, 2.0, 3.0)))
    assert result.tolist() == pytest.approx((0.0, 0.5, 0.5, 1.0))


def test_ranking_weights_ignore_zero_range_and_normalize() -> None:
    targets = torch.tensor(((0.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)))
    weights, scale = _ranking_weights(
        targets,
        torch.tensor((0.0, 1.0)),
        ("source-a", "source-b"),
        entropy_weighted=True,
    )
    assert scale == pytest.approx(1.0)
    assert weights[0] == 0.0
    assert weights[1] == pytest.approx(1.0)
    assert weights.sum() == pytest.approx(1.0)


def test_relative_where_oof_emits_complete_outcome_free_rows() -> None:
    dataset, outer_folds = _dataset()
    predictions, audit = fit_relative_where_oof(
        dataset,
        outer_folds,
        device="cpu",
        epochs=1,
    )
    assert audit["prediction_rows"] == dataset.decisions
    assert audit["prediction_outcomes_included"] is False
    assert audit["fits"] == 20
    assert all(fold["source_overlap"] == 0 for fold in audit["folds"])
    assert len(predictions) == dataset.decisions
    forbidden = {"task_deltas", "loss_gaps", "correct_after", "utility", "target"}
    for row in predictions:
        assert row["schema"] == RELATIVE_WHERE_SCHEMA
        assert set(row["variants"]) == set(RELATIVE_WHERE_VARIANTS)
        assert not forbidden.intersection(row)
        for variant in row["variants"].values():
            assert len(variant["action_scores"]) == 4
            assert len(variant["action_probabilities"]) == 4
            assert sum(variant["action_probabilities"]) == pytest.approx(1.0)
            assert variant["selected_action_id"] in DECAR_ACTION_IDS
            assert math.isfinite(variant["predicted_margin"])
