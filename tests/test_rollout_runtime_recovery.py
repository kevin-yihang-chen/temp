from __future__ import annotations

import json
from pathlib import Path

from beyond_entropy.rollout_runtime_recovery import (
    COMPLETION_SCHEMA,
    prepare_runtime_replay_plan,
    repair_runtime_from_replays,
    sha256_file,
)
from beyond_entropy.sharding import stable_shard_index


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _state_for_shard(index: int) -> str:
    candidate = 0
    while True:
        state_id = f"state-{index}-{candidate}"
        if stable_shard_index(state_id, 4) == index:
            return state_id
        candidate += 1


def test_exact_replay_repairs_missing_runtime_without_changing_rollouts(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    image = tmp_path / "image.png"
    image.write_bytes(b"fixture")
    rows: list[dict[str, object]] = []
    rollout_root = tmp_path / "rollout-shards"
    original_hashes: dict[int, str] = {}
    for shard_index in range(4):
        state_id = _state_for_shard(shard_index)
        rows.append(
            {
                "state_id": state_id,
                "source_id": f"source-{shard_index}",
                "image_path": str(image),
                "question": "q",
                "target": {"answers": ["a"]},
                "benchmark": "screenqa",
            }
        )
        shard_dir = rollout_root / f"shard-{shard_index:05d}-of-00004"
        rollouts = shard_dir / "rollouts.jsonl"
        action_rows = [
            {
                "state_id": state_id,
                "source_id": f"source-{shard_index}",
                "action_id": "answer-now" if action == 0 else f"zoom-{action}",
                "value": action,
            }
            for action in range(5)
        ]
        _write_jsonl(rollouts, action_rows)
        original_hashes[shard_index] = sha256_file(rollouts)
        _write_json(
            rollouts.with_suffix(".provenance.json"),
            {
                "output_sha256": original_hashes[shard_index],
                "examples": 1,
                "completed_examples": 1,
                "runtime_measurement": None,
            },
        )
    _write_jsonl(manifest, rows)
    replay_root = tmp_path / "replay"
    plan = prepare_runtime_replay_plan(
        manifest=manifest,
        rollout_root=rollout_root,
        replay_root=replay_root,
        expected_manifest_sha256=sha256_file(manifest),
        shard_count=4,
    )
    runtime = {
        "accelerator_name": "NVIDIA H800",
        "compute_capability": [9, 0],
        "requested_dtype": "bfloat16",
        "parameter_dtype": "torch.bfloat16",
        "attention_implementation": "sdpa",
        "actual_attention_implementation": "sdpa",
        "peak_allocated_bytes": 17,
        "peak_reserved_bytes": 19,
    }
    for entry in plan["entries"]:
        full_rows = [
            row
            for row in json.loads(
                json.dumps(
                    [
                        json.loads(line)
                        for line in Path(entry["full_rollouts"]).read_text().splitlines()
                    ]
                )
            )
            if row["state_id"] == entry["state_id"]
        ]
        probe_rollouts = Path(entry["probe_rollouts"])
        _write_jsonl(probe_rollouts, full_rows)
        _write_json(
            probe_rollouts.with_suffix(".provenance.json"),
            {
                "output_sha256": sha256_file(probe_rollouts),
                "model": "Qwen/Qwen2.5-VL-7B-Instruct",
                "model_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
                "runtime_measurement": runtime,
            },
        )
    completion = repair_runtime_from_replays(
        plan=replay_root / "plan.json",
        code_revision="revision",
        prior_job_ids=["1", "2"],
    )
    assert completion["schema"] == COMPLETION_SCHEMA
    assert completion["passed"] is True
    assert len(completion["repairs"]) == 4
    for shard_index in range(4):
        shard_dir = rollout_root / f"shard-{shard_index:05d}-of-00004"
        rollouts = shard_dir / "rollouts.jsonl"
        assert sha256_file(rollouts) == original_hashes[shard_index]
        provenance = json.loads(
            rollouts.with_suffix(".provenance.json").read_text(encoding="utf-8")
        )
        assert provenance["runtime_measurement"] == runtime
        assert provenance["runtime_measurement_recovery"]["exact_five_record_match"] is True
        assert provenance["runtime_measurement_recovery"]["original_process_peak_reconstructed"] is False


def test_runtime_recovery_workers_are_contract_locked() -> None:
    root = Path(__file__).resolve().parents[1]
    worker = (root / "scripts/slurm_screenqa_backbone_7b_runtime_recovery.sh").read_text()
    submit = (root / "scripts/submit_screenqa_backbone_7b_runtime_recovery.sh").read_text()
    assert "#SBATCH --partition=q-h800" in worker
    assert "#SBATCH --gres=gpu:h800:4" in worker
    assert "#SBATCH --mail-type=ALL" in worker
    assert "deterministic_one_state_h800_replay" in worker
    assert "original_process_peak_reconstructed" in worker
    assert "--bootstrap-resamples 5000" in worker
    assert "--mail-type=ALL" in submit
    assert "show-cpu-gpu-quota" in submit
    assert "-lt 80" in submit
    assert "6512131e7a9bbe55b65f9229a044df4" in submit
    assert "3da107bef0fa8614e6cb088f4e54745c" in submit
