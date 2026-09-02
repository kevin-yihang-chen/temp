from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from beyond_entropy.infographicvqa_attention_stop_diagnostic import (  # noqa: E402
    ATTENTION_STOP_DIAGNOSTIC_SCHEMA,
    _at_most_positive_top,
    evaluate_attention_stop_factorization,
)
from beyond_entropy.infographicvqa_decar import DECAR_ACTION_IDS  # noqa: E402
from beyond_entropy.schema import ActionRecord, BBox  # noqa: E402


def _decision(index: int) -> list[ActionRecord]:
    key = {
        "state_id": f"state-{index:03d}",
        "image_id": f"image-{index:03d}",
        "source_id": f"source-{index // 2:03d}",
        "question": "What is shown?",
        "original_image": f"image-{index:03d}.png",
        "replicate_id": "replicate-000",
        "generation_seed": index,
        "entropy_before": 1.0 - 0.05 * index,
        "answer_before": "before",
        "correct_before": 0.0,
        "metadata": {},
    }
    baseline = ActionRecord(
        **key,
        action_id="answer-now",
        action_type="ANSWER",
        candidate_bbox=None,
        entropy_after=key["entropy_before"],
        answer_after="before",
        correct_after=0.0,
        tool_cost=0.0,
        pre_action_features={},
    )
    zooms = []
    for action_index, action_id in enumerate(DECAR_ACTION_IDS):
        helpful = action_index == index % 4
        zooms.append(
            ActionRecord(
                **key,
                action_id=action_id,
                action_type="ZOOM",
                candidate_bbox=BBox(
                    0.25 * action_index,
                    0.0,
                    0.25 * (action_index + 1),
                    1.0,
                ),
                entropy_after=0.1,
                answer_after="after",
                correct_after=1.0 if helpful else 0.0,
                tool_cost=1.0,
                pre_action_features={"ug_grid_size": 4.0},
            )
        )
    return [baseline, *zooms]


def _payload(decisions: int) -> dict:
    rows = []
    for index in range(decisions):
        scores = torch.full((4,), 0.1)
        scores[index % 4] = 0.7
        rows.append(
            {
                "state_id": f"state-{index:03d}",
                "replicate_id": "replicate-000",
                "source_id": f"source-{index // 2:03d}",
                "image_id": f"image-{index:03d}",
                "question": "What is shown?",
                "action_ids": list(DECAR_ACTION_IDS),
                "tool_costs": torch.ones(4),
                "bboxes": torch.tensor(
                    [[0.25 * i, 0.0, 0.25 * (i + 1), 1.0] for i in range(4)]
                ),
                "state_signals": torch.tensor([1.0 - 0.05 * index]),
                "question_region_attention": scores,
                "question_image_attention_mass": 1.0,
            }
        )
    return {
        "format_version": 1,
        "metadata": {
            "outcomes_included": False,
            "question_region_attention": {
                "source_features_sha256": "features",
                "source_rollouts_sha256": "rollouts",
                "model_revision": "model",
                "attention_implementation": "eager",
                "top_layers": 4,
                "head_pooling": "mean",
                "question_token_pooling": "mean",
                "candidate_pooling": "ROI mean then normalize across candidates",
                "candidate_actions_executed": False,
                "replace_question_embedding": False,
                "code_revision": "code",
                "completed_decisions": decisions,
                "total_decisions": decisions,
            },
        },
        "decisions": rows,
    }


def test_positive_top_uses_at_most_budget_and_never_selects_negative() -> None:
    values = {("a", "r"): 0.3, ("b", "r"): -0.1, ("c", "r"): 0.2}
    assert _at_most_positive_top(values, budget=2) == {("a", "r"), ("c", "r")}
    assert _at_most_positive_top(values, budget=3) == {("a", "r"), ("c", "r")}


def test_attention_stop_diagnostic_exposes_fixed_action_oracle_headroom() -> None:
    records = [record for index in range(8) for record in _decision(index)]
    indices = np.tile(np.arange(4, dtype=np.int32), (100, 1))
    result = evaluate_attention_stop_factorization(
        records,
        _payload(8),
        expected_attention_code_revision="code",
        expected_model_revision="model",
        expected_source_features_sha256="features",
        expected_rollouts_sha256="rollouts",
        bootstrap_indices=indices,
        expected_decisions=8,
        expected_sources=4,
        expected_bootstrap_resamples=100,
    )
    assert result["schema"] == ATTENTION_STOP_DIAGNOSTIC_SCHEMA
    assert result["valid_for_formal_selection"] is False
    assert result["validation_or_test_inputs_used"] is False
    assert result["raw_action_positive_net"]["positive_net_states"] == 8
    assert result["ceilings"]["raw_action_positive_net_oracle"]["metrics"][
        "source_balanced"
    ]["utility"] == pytest.approx(0.95)
    assert len(result["operating_points"]) == 5
