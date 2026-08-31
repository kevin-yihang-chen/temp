from __future__ import annotations

import hashlib
import json

from PIL import Image

from beyond_entropy.dataset import read_jsonl, write_jsonl
from beyond_entropy.rollout_shards import (
    merge_qwen_rollout_shards,
    shard_directory_name,
)
from beyond_entropy.schema import ActionRecord, BBox
from beyond_entropy.sharding import SHARD_ALGORITHM, stable_shard_index


def _record(state_id: str, action_id: str) -> ActionRecord:
    zoom = action_id != "answer-now"
    return ActionRecord(
        state_id=state_id,
        image_id=f"image-{state_id}",
        source_id=f"source-{state_id}",
        question="Question?",
        original_image="image.png",
        replicate_id="replicate-000",
        generation_seed=0,
        action_id=action_id,
        action_type="ZOOM" if zoom else "ANSWER",
        candidate_bbox=BBox(0.0, 0.0, 0.5, 0.5) if zoom else None,
        entropy_before=1.0,
        entropy_after=0.5 if zoom else 1.0,
        answer_before="no",
        answer_after="yes" if zoom else "no",
        correct_before=0.0,
        correct_after=1.0 if zoom else 0.0,
        tool_cost=1.0 if zoom else 0.0,
    )


def test_merge_qwen_rollout_shards_proves_exact_state_coverage(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "white").save(image)
    manifest = tmp_path / "manifest.jsonl"
    states = [f"state-{index:03d}" for index in range(20)]
    with manifest.open("w", encoding="utf-8") as handle:
        for state_id in states:
            handle.write(
                json.dumps(
                    {
                        "state_id": state_id,
                        "image_id": f"image-{state_id}",
                        "source_id": f"source-{state_id}",
                        "image_path": str(image),
                        "question": "Question?",
                        "target": {"answers": ["yes"]},
                    }
                )
                + "\n"
            )
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    run_root = tmp_path / "run"
    shard_count = 3
    for shard_index in range(shard_count):
        shard_states = [
            state_id
            for state_id in states
            if stable_shard_index(state_id, shard_count) == shard_index
        ]
        shard_dir = run_root / shard_directory_name(shard_index, shard_count)
        shard_dir.mkdir(parents=True)
        records = [
            record
            for state_id in shard_states
            for record in (_record(state_id, "answer-now"), _record(state_id, "zoom-0"))
        ]
        rollouts = shard_dir / "rollouts.jsonl"
        write_jsonl(records, rollouts)
        provenance = {
            "code_revision": "revision",
            "manifest_sha256": manifest_sha256,
            "manifest_limit": None,
            "manifest_examples_before_sharding": len(states),
            "shard_algorithm": SHARD_ALGORITHM,
            "shard_count": shard_count,
            "shard_index": shard_index,
            "model": "model",
            "model_revision": "model-revision",
            "ug_framework_revision": "ug-revision",
            "scorer": "screenqa",
            "candidate_count": 1,
            "proposer": "ug-grid",
            "visual_crop_ratio": 2.0,
            "visual_cost": 1.0,
            "generation_seeds": [0],
            "max_new_tokens": 16,
            "min_pixels": 1,
            "max_pixels": 2,
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "system_prompt": "system",
            "local_files_only": True,
            "examples": len(shard_states),
            "completed_examples": len(shard_states),
            "resumed_from_records": len(records),
            "output_sha256": hashlib.sha256(rollouts.read_bytes()).hexdigest(),
        }
        (shard_dir / "rollouts.provenance.json").write_text(
            json.dumps(provenance), encoding="utf-8"
        )
        (shard_dir / "resume.audit.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "rollouts_sha256_before_resume": provenance["output_sha256"],
                    "rollouts_sha256_after_resume": provenance["output_sha256"],
                    "records": len(records),
                    "examples": len(shard_states),
                    "resumed_from_records": len(records),
                }
            ),
            encoding="utf-8",
        )

    output = tmp_path / "merged" / "rollouts.jsonl"
    audit = merge_qwen_rollout_shards(
        manifest_path=manifest,
        expected_manifest_sha256=manifest_sha256,
        run_root=run_root,
        shard_count=shard_count,
        output_path=output,
        expected_code_revision="revision",
        expected_scorer="screenqa",
        require_resume_audit=True,
        bootstrap_resamples=10,
        bootstrap_seed=7,
    )
    merged = read_jsonl(output)
    assert audit["passed"] is True
    assert audit["selected_states"] == len(states)
    assert audit["merged_records"] == len(states) * 2
    assert [record.state_id for record in merged[::2]] == states
