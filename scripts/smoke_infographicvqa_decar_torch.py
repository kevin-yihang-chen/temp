#!/usr/bin/env python3
"""Torch-runtime smoke checks for the frozen InfographicVQA DECAR models."""

from __future__ import annotations

import argparse
import math

import torch  # type: ignore[import-not-found]

from beyond_entropy.infographicvqa_decar import (
    DECAR_ACTION_IDS,
    DECAR_SCALAR_NAMES,
    DecarDataset,
    fit_nested_oof,
    fit_when,
    fit_where,
    predict_when,
    predict_where,
    score_when,
    select_where_actions,
)
from beyond_entropy.qwen_backend import generated_token_statistics


def _check_generation_statistics() -> None:
    logits = (
        torch.tensor([[2.0, 1.0, -1.0]], dtype=torch.float32),
        torch.tensor([[0.0, 0.5, 1.5]], dtype=torch.float32),
    )
    generated_ids = torch.tensor([[0, 2]], dtype=torch.long)
    entropies, log_probabilities = generated_token_statistics(logits, generated_ids)
    expected = [
        float(torch.log_softmax(logits[0][0], dim=-1)[0]),
        float(torch.log_softmax(logits[1][0], dim=-1)[2]),
    ]
    assert len(entropies) == len(log_probabilities) == 2
    assert all(0.0 <= value <= 1.0 for value in entropies)
    assert all(
        math.isclose(left, right) for left, right in zip(log_probabilities, expected)
    )


def _check_where_and_when(device: str) -> None:
    generator = torch.Generator().manual_seed(7)
    decisions = 9
    embedding_dim = 8
    question = torch.randn(decisions, embedding_dim, generator=generator)
    global_visual = torch.randn(decisions, embedding_dim, generator=generator)
    region = torch.randn(decisions, 4, embedding_dim, generator=generator)
    scalars = torch.randn(decisions, 4, len(DECAR_SCALAR_NAMES), generator=generator)
    targets = torch.tensor(
        [[0.2 * index, -0.1, 0.3, 0.0] for index in range(decisions)],
        dtype=torch.float32,
    )
    sources = [f"source-{index}" for index in range(decisions)]
    where = fit_where(
        question,
        global_visual,
        region,
        scalars,
        targets,
        sources,
        seed=17,
        device=device,
        epochs=2,
    )
    predictions = predict_where(
        where, question, global_visual, region, scalars, device=device
    )
    selected, gaps, margins = select_where_actions(predictions)
    assert predictions.shape == (decisions, 4)
    assert selected.shape == gaps.shape == margins.shape == (decisions,)

    row_indices = torch.arange(decisions)
    selected_region = region[row_indices, selected]
    selected_scalars = scalars[row_indices, selected]
    selected_deltas = torch.tensor(
        [0.3, 0.0, -0.2, 0.1, 0.0, -0.1, 0.2, 0.0, -0.3],
        dtype=torch.float32,
    )
    for binary in (False, True):
        when = fit_when(
            question,
            global_visual,
            selected_region,
            selected_scalars,
            gaps,
            margins,
            selected_deltas,
            sources,
            seed=20 if binary else 19,
            device=device,
            binary=binary,
            epochs=2,
        )
        probabilities, predicted_delta = predict_when(
            when,
            question,
            global_visual,
            selected_region,
            selected_scalars,
            gaps,
            margins,
            device=device,
        )
        scores, eligible = score_when(when, probabilities, predicted_delta)
        assert probabilities.shape == (decisions, 2 if binary else 3)
        assert predicted_delta.shape == scores.shape == eligible.shape == (decisions,)
        assert torch.allclose(
            probabilities.sum(dim=1), torch.ones(decisions), atol=1e-6
        )


def _check_nested_oof(device: str) -> None:
    generator = torch.Generator().manual_seed(11)
    sources = tuple(
        source_id
        for source_index in range(20)
        for source_id in [f"source-{source_index:02d}"] * 3
    )
    decisions = len(sources)
    embedding_dim = 8
    class_delta = torch.tensor([0.3, 0.0, -0.2], dtype=torch.float32).repeat(20)
    task_deltas = class_delta[:, None].expand(-1, 4).clone()
    loss_gaps = (
        torch.tensor([0.3, 0.1, -0.1, 0.0], dtype=torch.float32)[None, :]
        .expand(decisions, -1)
        .clone()
    )
    dataset = DecarDataset(
        state_ids=tuple(f"state-{index:03d}" for index in range(decisions)),
        replicate_ids=("replicate-000",) * decisions,
        image_ids=tuple(f"image-{index:03d}" for index in range(decisions)),
        source_ids=sources,
        action_ids=(DECAR_ACTION_IDS,) * decisions,
        question=torch.randn(decisions, embedding_dim, generator=generator),
        global_visual=torch.randn(decisions, embedding_dim, generator=generator),
        region=torch.randn(decisions, 4, embedding_dim, generator=generator),
        scalars=torch.randn(decisions, 4, len(DECAR_SCALAR_NAMES), generator=generator),
        loss_gaps=loss_gaps,
        task_deltas=task_deltas,
        correct_before=torch.zeros(decisions),
        entropy_before=torch.full((decisions,), 0.5),
        entropy_after=torch.full((decisions, 4), 0.4),
    )
    outer = {f"source-{index:02d}": index % 5 for index in range(20)}
    inner = {
        (outer_fold, f"source-{source_index:02d}"): (source_index + outer_fold) % 4
        for outer_fold in range(5)
        for source_index in range(20)
        if source_index % 5 != outer_fold
    }
    payload, audit = fit_nested_oof(
        dataset,
        outer,
        inner,
        device=device,
        epochs=1,
    )
    assert payload["metadata"]["outcomes_included"] is False
    assert len(payload["predictions"]) == decisions
    assert len(audit["folds"]) == 5
    assert all(fold["source_overlap"] == 0 for fold in audit["folds"])
    assert all(
        set(row["variants"])
        == {"decar", "task_value_only", "loss_only", "no_harm_head"}
        for row in payload["predictions"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--nested-oof", action="store_true")
    args = parser.parse_args()
    _check_generation_statistics()
    _check_where_and_when(args.device)
    if args.nested_oof:
        _check_nested_oof(args.device)
    print(
        "InfographicVQA DECAR torch smoke passed: "
        f"torch={torch.__version__}, device={args.device}, nested_oof={args.nested_oof}"
    )


if __name__ == "__main__":
    main()
