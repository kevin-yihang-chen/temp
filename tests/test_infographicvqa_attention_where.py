from __future__ import annotations

from copy import deepcopy

import pytest

torch = pytest.importorskip("torch")

from beyond_entropy.infographicvqa_attention_where import (  # noqa: E402
    assemble_attention_where_features,
)
from beyond_entropy.infographicvqa_decar import DECAR_ACTION_IDS  # noqa: E402
from beyond_entropy.schema import ActionRecord, BBox  # noqa: E402


CODE_REVISION = "revision"
MODEL_REVISION = "model-revision"
FEATURES_SHA256 = "features-sha256"
ROLLOUTS_SHA256 = "rollouts-sha256"


def _record(action_index: int | None) -> ActionRecord:
    is_answer = action_index is None
    return ActionRecord(
        state_id="state-0",
        image_id="image-0",
        source_id="source-0",
        question="What is shown?",
        original_image="image.img",
        replicate_id="replicate-000",
        generation_seed=0,
        action_id="answer-now" if is_answer else DECAR_ACTION_IDS[action_index],
        action_type="ANSWER" if is_answer else "ZOOM",
        candidate_bbox=(
            None
            if is_answer
            else BBox(0.1 * action_index, 0.0, 0.4 + 0.1 * action_index, 0.5)
        ),
        entropy_before=0.4,
        entropy_after=0.4,
        answer_before="before",
        answer_after="after",
        correct_before=0.2,
        correct_after=0.2,
        tool_cost=0.0 if is_answer else 1.0,
        pre_action_features={} if is_answer else {"ug_grid_size": 12.0},
        metadata={},
    )


def _payload() -> dict:
    records = [_record(None), *[_record(index) for index in range(4)]]
    zooms = records[1:]
    return {
        "format_version": 1,
        "metadata": {
            "outcomes_included": False,
            "question_region_attention": {
                "source_features_sha256": FEATURES_SHA256,
                "source_rollouts_sha256": ROLLOUTS_SHA256,
                "model_revision": MODEL_REVISION,
                "attention_implementation": "eager",
                "top_layers": 4,
                "head_pooling": "mean",
                "question_token_pooling": "mean",
                "candidate_pooling": "ROI mean then normalize across candidates",
                "candidate_actions_executed": False,
                "replace_question_embedding": False,
                "code_revision": CODE_REVISION,
                "completed_decisions": 1,
                "total_decisions": 1,
            },
        },
        "decisions": [
            {
                "state_id": "state-0",
                "replicate_id": "replicate-000",
                "source_id": "source-0",
                "image_id": "image-0",
                "question": "What is shown?",
                "action_ids": list(DECAR_ACTION_IDS),
                "tool_costs": torch.tensor([1.0, 1.0, 1.0, 1.0]),
                "bboxes": torch.tensor(
                    [record.candidate_bbox.to_list() for record in zooms]
                ),
                "state_signals": torch.tensor([0.4]),
                "question_region_attention": torch.tensor([0.1, 0.2, 0.6, 0.1]),
                "question_image_attention_mass": 2.0,
            }
        ],
    }


def _assemble(payload: dict):
    records = [_record(None), *[_record(index) for index in range(4)]]
    return assemble_attention_where_features(
        records,
        payload,
        expected_code_revision=CODE_REVISION,
        expected_model_revision=MODEL_REVISION,
        expected_source_features_sha256=FEATURES_SHA256,
        expected_rollouts_sha256=ROLLOUTS_SHA256,
    )


def test_attention_where_audit_selects_raw_attention_argmax() -> None:
    features, audit = _assemble(_payload())
    assert features.decisions == 1
    assert features.selected_indices.tolist() == [2]
    assert features.margins.tolist() == pytest.approx([0.4])
    assert audit["passed"] is True
    assert audit["outcomes_included"] is False
    assert audit["selected_action_counts"]["ug-grid-02"] == 1
    assert audit["score_sum_max_absolute_error"] <= 1e-6


def test_attention_where_audit_rejects_unnormalized_scores() -> None:
    payload = deepcopy(_payload())
    payload["decisions"][0]["question_region_attention"] = torch.tensor(
        [0.1, 0.2, 0.3, 0.1]
    )
    with pytest.raises(ValueError, match="unnormalized"):
        _assemble(payload)


def test_attention_where_audit_rejects_metadata_drift() -> None:
    payload = deepcopy(_payload())
    payload["metadata"]["question_region_attention"]["top_layers"] = 3
    with pytest.raises(ValueError, match="top_layers"):
        _assemble(payload)
