from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from beyond_entropy.infographicvqa_decar import DECAR_ACTION_IDS  # noqa: E402
from beyond_entropy.infographicvqa_literature_attention_extraction import (  # noqa: E402
    LITERATURE_ATTENTION_METADATA_KEY,
    _expected_resume_metadata,
)
from beyond_entropy.infographicvqa_literature_attention_where import (  # noqa: E402
    VICROP_QWEN25_LAYER_INDEX,
    assemble_literature_attention_where_features,
    image_attention_entropy,
    laser_all_head_candidate_scores,
    vicrop_relative_candidate_scores,
)
from beyond_entropy.schema import ActionRecord, BBox  # noqa: E402


def _quadrants():
    return torch.tensor(
        [
            [0.0, 0.0, 0.5, 0.5],
            [0.5, 0.0, 1.0, 0.5],
            [0.0, 0.5, 0.5, 1.0],
            [0.5, 0.5, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )


def test_vicrop_relative_projection_uses_official_ratio_and_layer() -> None:
    query = torch.tensor([[8.0, 8.0], [1.0, 1.0]])
    generic = torch.tensor([[2.0, 2.0], [1.0, 1.0]])
    result = vicrop_relative_candidate_scores(query, generic, _quadrants())
    assert result.selected_layer == VICROP_QWEN25_LAYER_INDEX == 22
    assert result.zero_map_fallback is False
    assert result.candidate_scores.tolist() == pytest.approx([0.4, 0.4, 0.1, 0.1])


def test_vicrop_relative_projection_rejects_zero_denominator() -> None:
    with pytest.raises(ValueError, match="zero denominator"):
        vicrop_relative_candidate_scores(
            torch.ones(2, 2), torch.tensor([[1.0, 1.0], [1.0, 0.0]]), _quadrants()
        )


def test_laser_all_head_projection_selects_dynamic_layer() -> None:
    query = torch.ones(3, 2, 2, 2)
    no_query = torch.ones_like(query)
    query[1, :, 1, 1] = 5.0
    query[2, :, 0, 0] = 2.0
    result = laser_all_head_candidate_scores(query, no_query, _quadrants())
    assert result.selected_layer == 1
    assert result.zero_map_fallback is False
    assert result.candidate_scores.tolist() == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert result.layer_scores.tolist() == pytest.approx([0.0, 4.0, 1.0])


def test_laser_all_head_projection_has_registered_zero_fallback() -> None:
    attention = torch.ones(2, 2, 2, 2)
    result = laser_all_head_candidate_scores(attention, attention, _quadrants())
    assert result.selected_layer == 0
    assert result.zero_map_fallback is True
    assert result.candidate_scores.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_image_attention_entropy_normalizes_positive_mass() -> None:
    assert image_attention_entropy(torch.ones(2, 2)) == pytest.approx(
        1.3862943611198906
    )
    with pytest.raises(ValueError, match="positive mass"):
        image_attention_entropy(torch.zeros(2, 2))


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


def _feature_payload() -> dict:
    records = [_record(None), *[_record(index) for index in range(4)]]
    metadata = _expected_resume_metadata(
        source_sha256="features-sha",
        rollouts_sha256="rollouts-sha",
        model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct",
        revision="model-revision",
        device_map="cuda:0",
        dtype="bfloat16",
    )
    metadata.update(
        {
            "code_revision": "revision",
            "completed_decisions": 1,
            "total_decisions": 1,
            "source_features": "features.pt",
            "source_rollouts": "rollouts.jsonl",
        }
    )
    return {
        "format_version": 1,
        "metadata": {
            "outcomes_included": False,
            LITERATURE_ATTENTION_METADATA_KEY: metadata,
        },
        "decisions": [
            {
                "state_id": "state-0",
                "replicate_id": "replicate-000",
                "source_id": "source-0",
                "image_id": "image-0",
                "question": "What is shown?",
                "action_ids": list(DECAR_ACTION_IDS),
                "tool_costs": torch.ones(4),
                "bboxes": torch.tensor(
                    [record.candidate_bbox.to_list() for record in records[1:]]
                ),
                "state_signals": torch.tensor([0.4]),
                "vicrop_relative_region_attention": torch.tensor([0.1, 0.2, 0.6, 0.1]),
                "laser_contrastive_region_attention": torch.tensor(
                    [0.4, 0.3, 0.2, 0.1]
                ),
                "laser_selected_layer": 1,
                "laser_layer_scores": torch.tensor([0.0, 2.0, 1.0] + [0.0] * 21),
                "laser_zero_map_fallback": False,
                "encore_early_entropy": torch.tensor([1.0, 1.1]),
                "literature_attention_image_mass": torch.tensor([0.2, 0.1, 0.15]),
                "literature_attention_grid_thw": torch.tensor([1, 4, 4]),
                "literature_attention_layer_count": 24,
                "literature_attention_head_count": 4,
            }
        ],
    }


def test_literature_feature_audit_covers_both_frozen_variants() -> None:
    records = [_record(None), *[_record(index) for index in range(4)]]
    features, audit = assemble_literature_attention_where_features(
        records,
        _feature_payload(),
        expected_code_revision="revision",
        expected_model_revision="model-revision",
        expected_source_features_sha256="features-sha",
        expected_rollouts_sha256="rollouts-sha",
    )
    assert features.vicrop_selected_indices.tolist() == [2]
    assert features.laser_selected_indices.tolist() == [0]
    assert audit["passed"] is True
    assert audit["outcomes_included"] is False
    assert audit["vicrop_relative_bank"]["selected_action_counts"]["ug-grid-02"] == 1
    assert audit["laser_contrastive_all_head_bank"]["selected_layer_counts"]["1"] == 1


def test_literature_feature_audit_rejects_inconsistent_laser_layer() -> None:
    records = [_record(None), *[_record(index) for index in range(4)]]
    payload = _feature_payload()
    payload["decisions"][0]["laser_selected_layer"] = 2
    with pytest.raises(ValueError, match="layer selection"):
        assemble_literature_attention_where_features(
            records,
            payload,
            expected_code_revision="revision",
            expected_model_revision="model-revision",
            expected_source_features_sha256="features-sha",
            expected_rollouts_sha256="rollouts-sha",
        )


def test_literature_feature_audit_rejects_inconsistent_fallback_flag() -> None:
    records = [_record(None), *[_record(index) for index in range(4)]]
    payload = _feature_payload()
    payload["decisions"][0]["laser_zero_map_fallback"] = True
    with pytest.raises(ValueError, match="fallback flag"):
        assemble_literature_attention_where_features(
            records,
            payload,
            expected_code_revision="revision",
            expected_model_revision="model-revision",
            expected_source_features_sha256="features-sha",
            expected_rollouts_sha256="rollouts-sha",
        )
