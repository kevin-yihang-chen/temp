import json

import pytest

from beyond_entropy.schema import BBox
from beyond_entropy.sequential_rollout_shards import (
    merge_sequential_rollout_shards,
    sha256_file,
    shard_directory_name,
)
from beyond_entropy.sequential_schema import AcquiredObservationSpec, SequentialRolloutRecord
from beyond_entropy.sharding import stable_shard_index


def _record(index):
    return SequentialRolloutRecord(
        state_id=f"s{index}", image_id=f"i{index}", source_id=f"source-{index}",
        question=f"q{index}", original_image=f"/tmp/{index}.png", step_index=1,
        acquired_observations=(AcquiredObservationSpec(
            "crop-a", BBox(0, 0, .5, .5), 1
        ),),
        proposed_action_id="crop-b", proposed_bbox=BBox(.5, .5, 1, 1),
        proposed_visual_cost=1, replicate_id="replicate-000", generation_seed=0,
        stop_answer="no", stop_correct=0, stop_entropy=.5,
        stop_max_probability=.6, stop_top1_top2_margin=.2,
        continue_answer="yes", continue_correct=1, continue_entropy=.2,
        continue_max_probability=.8, continue_top1_top2_margin=.6,
    )


def _materialize(tmp_path, *, corrupt=False):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps({
        "state_id": f"s{i}", "source_id": f"source-{i}",
        "image_id": f"i{i}", "question": f"q{i}",
    }) + "\n" for i in range(6)))
    count = 2
    root = tmp_path / "shards"
    for index in range(count):
        directory = root / shard_directory_name(index, count)
        directory.mkdir(parents=True)
        records = [
            _record(i) for i in range(6)
            if stable_shard_index(
                f"s{i}", count, namespace="sequential-prefix-v1"
            ) == index
        ]
        rollouts = directory / "rollouts.jsonl"
        rollouts.write_text("".join(json.dumps(x.to_dict()) + "\n" for x in records))
        completion = {
            "schema": "sequential_prefix_rollout_completion_v1", "completed": True,
            "test_accessed": False, "dataset_role": "train", "benchmark": "chartqa",
            "manifest_sha256": sha256_file(manifest), "states": len(records),
            "record_count": len(records), "generation_seeds": [0],
            "shard_algorithm": "sha256-state-id-v1", "shard_count": count,
            "shard_index": index, "code_revision": "revision",
            "rollouts_sha256": sha256_file(rollouts),
        }
        if corrupt and index == 1:
            completion["states"] += 1
        (directory / "rollouts.jsonl.complete.json").write_text(json.dumps(completion))
    return manifest, root


def test_merge_sequential_shards_proves_exact_coverage(tmp_path):
    manifest, root = _materialize(tmp_path)
    result = merge_sequential_rollout_shards(
        manifest_path=manifest, expected_manifest_sha256=sha256_file(manifest),
        run_root=root, shard_count=2, output_dir=tmp_path / "merged",
        expected_code_revision="revision", benchmark="chartqa", dataset_role="train",
    )
    assert result["states"] == result["records"] == 6
    assert result["test_accessed"] is False
    assert result["headroom_diagnostic"]["beneficial_count"] == 6
    assert sha256_file(tmp_path / "merged" / "rollouts.jsonl") == result["rollouts_sha256"]


def test_merge_sequential_shards_rejects_completion_drift(tmp_path):
    manifest, root = _materialize(tmp_path, corrupt=True)
    with pytest.raises(ValueError, match="completion mismatch"):
        merge_sequential_rollout_shards(
            manifest_path=manifest, expected_manifest_sha256=sha256_file(manifest),
            run_root=root, shard_count=2, output_dir=tmp_path / "merged",
            expected_code_revision="revision", benchmark="chartqa", dataset_role="train",
        )
