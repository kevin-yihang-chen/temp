from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from beyond_entropy.infographicvqa_decar import (  # noqa: E402
    DECAR_ACTION_IDS,
    DECAR_SCALAR_NAMES,
    assemble_decar_dataset,
    decar_fit_seed,
    fit_when,
    fit_where,
    predict_when,
    predict_where,
    score_when,
    select_where_actions,
    source_balanced_decision_weights,
)
from beyond_entropy.schema import ActionRecord, BBox  # noqa: E402


def _record(action_index: int | None) -> ActionRecord:
    is_answer = action_index is None
    action_id = "answer-now" if is_answer else DECAR_ACTION_IDS[action_index]
    after = 0.2 if is_answer else [0.5, 0.2, 0.0, 0.3][action_index]
    entropy_after = 0.4 if is_answer else [0.2, 0.3, 0.5, 0.1][action_index]
    return ActionRecord(
        state_id="state-0",
        image_id="image-0",
        source_id="source-0",
        question="What is shown?",
        original_image="image.img",
        replicate_id="replicate-000",
        generation_seed=0,
        action_id=action_id,
        action_type="ANSWER" if is_answer else "ZOOM",
        candidate_bbox=(
            None
            if is_answer
            else BBox(0.1 * action_index, 0.0, 0.4 + 0.1 * action_index, 0.5)
        ),
        entropy_before=0.4,
        entropy_after=entropy_after,
        answer_before="before",
        answer_after="before" if is_answer else "after",
        correct_before=0.2,
        correct_after=after,
        tool_cost=0.0 if is_answer else 1.0,
        pre_action_features={} if is_answer else {"ug_grid_size": 12.0},
        metadata={
            "baseline_backend": {
                "generated_tokens": 2,
                "normalized_token_entropies": [0.3, 0.5],
                "generated_token_log_probabilities": [-0.2, -0.4],
                "mean_generated_token_log_probability": -0.3,
            }
        },
    )


def _nll_row(record: ActionRecord, mean_nll: float) -> dict[str, Any]:
    return {
        "action_id": record.action_id,
        "action_type": record.action_type,
        "answer_mean_nll": mean_nll,
        "answer_sum_nll": mean_nll * 2,
        "answer_token_count": 2,
        "config_sha256": "config",
        "correct_after": record.correct_after,
        "correct_before": record.correct_before,
        "entropy_after": record.entropy_after,
        "entropy_before": record.entropy_before,
        "image_id": record.image_id,
        "replicate_id": record.replicate_id,
        "schema": "visual_action_answer_nll_v1",
        "source_id": record.source_id,
        "state_id": record.state_id,
        "target_answer_count": 1,
        "target_answer_index": 0,
        "target_answer_sha256": "target",
        "target_answer_votes": 1,
        "tool_cost": record.tool_cost,
    }


def test_assemble_decar_dataset_builds_registered_sixteen_scalars() -> None:
    records = [_record(None), *[_record(index) for index in range(4)]]
    nll = [
        _nll_row(record, 1.0 if record.action_type == "ANSWER" else 0.8 + 0.1 * index)
        for index, record in enumerate(records)
    ]
    feature_payload = {
        "format_version": 1,
        "metadata": {"outcomes_included": False},
        "decisions": [
            {
                "state_id": "state-0",
                "replicate_id": "replicate-000",
                "source_id": "source-0",
                "image_id": "image-0",
                "action_ids": list(DECAR_ACTION_IDS),
                "question_embedding": torch.arange(8, dtype=torch.float32),
                "global_visual_embedding": torch.arange(8, dtype=torch.float32) + 1,
                "region_embeddings": torch.arange(32, dtype=torch.float32).reshape(
                    4, 8
                ),
                "visual_grid_hw": [4, 6],
            }
        ],
    }
    dataset = assemble_decar_dataset(
        records, nll, feature_payload, {"image-0": (100, 200)}
    )
    assert dataset.decisions == 1
    assert dataset.state_ids == ("state-0",)
    assert dataset.replicate_ids == ("replicate-000",)
    assert dataset.image_ids == ("image-0",)
    assert dataset.question.shape == (1, 8)
    assert dataset.region.shape == (1, 4, 8)
    assert dataset.scalars.shape == (1, 4, len(DECAR_SCALAR_NAMES))
    assert dataset.loss_gaps.tolist()[0] == pytest.approx([0.9, 0.1, 0.0, -0.1])
    assert dataset.task_deltas.tolist()[0] == pytest.approx([0.3, 0.0, -0.2, 0.1])
    assert dataset.scalars[0, 0, -4:].tolist() == pytest.approx([2.0, 0.4, 0.5, -0.3])


def test_source_balanced_weights_give_sources_equal_mass() -> None:
    weights = source_balanced_decision_weights(["a", "a", "b"])
    assert float(weights[:2].sum()) == pytest.approx(0.5)
    assert float(weights[2]) == pytest.approx(0.5)


def test_decar_fit_seed_freezes_refit_and_gate_namespaces() -> None:
    assert decar_fit_seed(0, 0, "decar") == 20_260_917
    assert decar_fit_seed(4, 4, "task_value_only") == 20_261_358
    assert decar_fit_seed(2, 5, "no_harm_head") == 20_261_170
    with pytest.raises(ValueError, match="variant"):
        decar_fit_seed(0, 0, "unknown")


def test_where_and_when_low_epoch_smoke_preserve_shapes() -> None:
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
        device="cpu",
        epochs=2,
    )
    predictions = predict_where(
        where, question, global_visual, region, scalars, device="cpu"
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
    when = fit_when(
        question,
        global_visual,
        selected_region,
        selected_scalars,
        gaps,
        margins,
        selected_deltas,
        sources,
        seed=19,
        device="cpu",
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
        device="cpu",
    )
    scores, eligible = score_when(when, probabilities, predicted_delta)
    assert probabilities.shape == (decisions, 3)
    assert predicted_delta.shape == scores.shape == eligible.shape == (decisions,)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(decisions), atol=1e-6)

    binary = fit_when(
        question,
        global_visual,
        selected_region,
        selected_scalars,
        gaps,
        margins,
        selected_deltas,
        sources,
        seed=20,
        device="cpu",
        binary=True,
        epochs=2,
    )
    binary_probabilities, binary_delta = predict_when(
        binary,
        question,
        global_visual,
        selected_region,
        selected_scalars,
        gaps,
        margins,
        device="cpu",
    )
    binary_scores, binary_eligible = score_when(
        binary, binary_probabilities, binary_delta
    )
    assert binary_probabilities.shape == (decisions, 2)
    assert binary_scores.shape == binary_eligible.shape == (decisions,)
