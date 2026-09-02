from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from beyond_entropy.dataset import write_jsonl  # noqa: E402
from beyond_entropy.infographicvqa_decar import DECAR_ACTION_IDS  # noqa: E402
from beyond_entropy.infographicvqa_literature_attention_extraction import (  # noqa: E402
    LITERATURE_ATTENTION_METADATA_KEY,
    _expected_resume_metadata,
)
from beyond_entropy.infographicvqa_literature_attention_merge import (  # noqa: E402
    merge_literature_attention_shards,
    sha256_file,
)
from beyond_entropy.qwen_semantic import load_semantic_feature_dataset  # noqa: E402
from beyond_entropy.schema import ActionRecord, BBox  # noqa: E402


def _records(index: int) -> list[ActionRecord]:
    result = []
    for action_index in [None, 0, 1, 2, 3]:
        is_answer = action_index is None
        result.append(
            ActionRecord(
                state_id=f"state-{index}",
                image_id=f"image-{index}",
                source_id=f"source-{index}",
                question=f"Question {index}?",
                original_image=f"image-{index}.png",
                replicate_id="replicate-000",
                generation_seed=0,
                action_id=(
                    "answer-now" if is_answer else DECAR_ACTION_IDS[action_index]
                ),
                action_type="ANSWER" if is_answer else "ZOOM",
                candidate_bbox=(
                    None
                    if is_answer
                    else BBox(
                        0.1 * action_index,
                        0.0,
                        0.4 + 0.1 * action_index,
                        0.5,
                    )
                ),
                entropy_before=0.4,
                entropy_after=0.4,
                answer_before="before",
                answer_after="after",
                correct_before=0.2,
                correct_after=0.2,
                tool_cost=0.0 if is_answer else 1.0,
                pre_action_features={},
                metadata={},
            )
        )
    return result


def _decision(index: int) -> dict:
    return {
        "state_id": f"state-{index}",
        "replicate_id": "replicate-000",
        "source_id": f"source-{index}",
        "image_id": f"image-{index}",
        "question": f"Question {index}?",
        "action_ids": list(DECAR_ACTION_IDS),
        "tool_costs": torch.ones(4),
        "bboxes": torch.tensor(
            [[0.1 * action, 0.0, 0.4 + 0.1 * action, 0.5] for action in range(4)],
            dtype=torch.float32,
        ),
        "state_signals": torch.tensor([0.4]),
        "marker": torch.tensor([index]),
    }


def test_literature_attention_merge_is_canonical_and_source_disjoint(
    tmp_path: Path,
) -> None:
    full_records = _records(0) + _records(1)
    full_rollouts = tmp_path / "full.jsonl"
    write_jsonl(full_records, full_rollouts)
    full_features = tmp_path / "full.pt"
    torch.save(
        {
            "format_version": 1,
            "metadata": {"outcomes_included": False, "base": "canonical"},
            "decisions": [_decision(0), _decision(1)],
        },
        full_features,
    )
    shard_rollouts = []
    shard_features = []
    for index in range(2):
        rollout = tmp_path / f"shard-{index}.jsonl"
        write_jsonl(_records(index), rollout)
        metadata = _expected_resume_metadata(
            source_sha256=f"source-{index}-sha",
            rollouts_sha256=sha256_file(rollout),
            model_name_or_path="model",
            revision="model-revision",
            device_map="cuda:0",
            dtype="bfloat16",
        )
        metadata.update(
            {
                "code_revision": "code-revision",
                "source_features": str((tmp_path / f"base-{index}.pt").resolve()),
                "source_rollouts": str(rollout.resolve()),
                "completed_decisions": 1,
                "total_decisions": 1,
            }
        )
        feature = tmp_path / f"shard-{index}.pt"
        torch.save(
            {
                "format_version": 1,
                "metadata": {
                    "outcomes_included": False,
                    LITERATURE_ATTENTION_METADATA_KEY: metadata,
                },
                "decisions": [_decision(index)],
            },
            feature,
        )
        shard_rollouts.append(rollout)
        shard_features.append(feature)
    output = tmp_path / "merged.pt"
    report_path = tmp_path / "report.json"
    report = merge_literature_attention_shards(
        full_rollouts_path=full_rollouts,
        expected_full_rollouts_sha256=sha256_file(full_rollouts),
        source_features_path=full_features,
        expected_source_features_sha256=sha256_file(full_features),
        shard_rollout_paths=shard_rollouts,
        shard_feature_paths=shard_features,
        expected_code_revision="code-revision",
        output_path=output,
        report_path=report_path,
    )
    merged = load_semantic_feature_dataset(output)
    assert report["passed"] is True
    assert report["source_disjoint"] is True
    assert report["decisions"] == 2
    assert [row["state_id"] for row in merged["decisions"]] == [
        "state-0",
        "state-1",
    ]
    stage = merged["metadata"][LITERATURE_ATTENTION_METADATA_KEY]
    assert stage["source_features"] == str(full_features.resolve())
    assert stage["source_rollouts"] == str(full_rollouts.resolve())
    assert stage["completed_decisions"] == 2
    assert merged["metadata"]["base"] == "canonical"
