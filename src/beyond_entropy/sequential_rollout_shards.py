"""Validation and merging for deterministic sequential rollout shards."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .sequential_metrics import sequential_diagnostic
from .sequential_schema import SequentialRolloutRecord
from .sharding import SHARD_ALGORITHM, stable_shard_index


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shard_directory_name(index: int, count: int) -> str:
    if count <= 0 or not 0 <= index < count:
        raise ValueError("invalid sequential shard index/count")
    return f"shard-{index:05d}-of-{count:05d}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    result = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row at {path}:{number}")
            result.append(value)
    if not result:
        raise ValueError(f"empty JSONL: {path}")
    return result


def merge_sequential_rollout_shards(
    *, manifest_path: str | Path, expected_manifest_sha256: str,
    run_root: str | Path, shard_count: int, output_dir: str | Path,
    expected_code_revision: str, benchmark: str, dataset_role: str,
    generation_seeds: tuple[int, ...] = (0,),
) -> dict[str, Any]:
    """Prove exact state/replicate coverage before producing one merged bank."""

    manifest = Path(manifest_path).resolve()
    root = Path(run_root).resolve()
    destination = Path(output_dir).resolve()
    if sha256_file(manifest) != expected_manifest_sha256:
        raise ValueError("sequential merge manifest hash mismatch")
    if shard_count <= 0 or not generation_seeds:
        raise ValueError("positive shard count and generation seeds required")
    manifest_rows = _read_jsonl(manifest)
    manifest_by_state = {str(row["state_id"]): row for row in manifest_rows}
    if len(manifest_by_state) != len(manifest_rows):
        raise ValueError("duplicate manifest state")
    expected_decisions = {
        (state_id, f"replicate-{index:03d}")
        for state_id in manifest_by_state
        for index in range(len(generation_seeds))
    }

    records: list[SequentialRolloutRecord] = []
    shard_reports = []
    seen_decisions: set[tuple[str, str]] = set()
    for index in range(shard_count):
        shard_dir = root / shard_directory_name(index, shard_count)
        rollouts = shard_dir / "rollouts.jsonl"
        completion_path = shard_dir / "rollouts.jsonl.complete.json"
        completion = json.loads(completion_path.read_text())
        expected_states = {
            state_id for state_id in manifest_by_state
            if stable_shard_index(
                state_id, shard_count, namespace="sequential-prefix-v1"
            ) == index
        }
        required = {
            "schema": "sequential_prefix_rollout_completion_v1",
            "completed": True,
            "test_accessed": dataset_role == "test",
            "dataset_role": dataset_role,
            "benchmark": benchmark,
            "manifest_sha256": expected_manifest_sha256,
            "states": len(expected_states),
            "record_count": len(expected_states) * len(generation_seeds),
            "generation_seeds": list(generation_seeds),
            "shard_algorithm": SHARD_ALGORITHM,
            "shard_count": shard_count,
            "shard_index": index,
            "code_revision": expected_code_revision,
        }
        for key, expected in required.items():
            if completion.get(key) != expected:
                raise ValueError(
                    f"sequential shard {index} completion mismatch for {key}: "
                    f"{completion.get(key)!r} != {expected!r}"
                )
        if completion.get("rollouts_sha256") != sha256_file(rollouts):
            raise ValueError(f"sequential shard {index} rollout hash mismatch")
        shard_records = [
            SequentialRolloutRecord.from_dict(row) for row in _read_jsonl(rollouts)
        ]
        if {record.state_id for record in shard_records} != expected_states:
            raise ValueError(f"sequential shard {index} state coverage mismatch")
        for record in shard_records:
            if record.decision_id in seen_decisions:
                raise ValueError("duplicate sequential decision across shards")
            row = manifest_by_state[record.state_id]
            if (
                record.source_id != str(row["source_id"])
                or record.image_id != str(row["image_id"])
                or record.question != str(row["question"])
            ):
                raise ValueError("sequential rollout and manifest identity mismatch")
            seen_decisions.add(record.decision_id)
        records.extend(shard_records)
        shard_reports.append({
            "index": index, "states": len(expected_states),
            "records": len(shard_records), "rollouts_sha256": sha256_file(rollouts),
            "completion_sha256": sha256_file(completion_path),
        })
    if seen_decisions != expected_decisions:
        raise ValueError("merged sequential decision coverage mismatch")
    if destination.exists():
        raise FileExistsError("refusing to overwrite merged sequential bank")
    destination.mkdir(parents=True)
    merged_path = destination / "rollouts.jsonl"
    ordered = sorted(records, key=lambda item: item.decision_id)
    payload = "".join(
        json.dumps(record.to_dict(), allow_nan=False, sort_keys=True) + "\n"
        for record in ordered
    )
    merged_path.write_text(payload)
    report = {
        "schema": "merged_sequential_rollout_bank_v1",
        "completed": True,
        "benchmark": benchmark,
        "dataset_role": dataset_role,
        "test_accessed": dataset_role == "test",
        "manifest_path": str(manifest),
        "manifest_sha256": expected_manifest_sha256,
        "code_revision": expected_code_revision,
        "shard_algorithm": SHARD_ALGORITHM,
        "shard_count": shard_count,
        "generation_seeds": list(generation_seeds),
        "states": len(manifest_by_state),
        "records": len(ordered),
        "rollouts_path": str(merged_path),
        "rollouts_sha256": sha256_file(merged_path),
        "headroom_diagnostic": sequential_diagnostic(ordered),
        "shards": shard_reports,
    }
    report_path = destination / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {**report, "report_sha256": sha256_file(report_path)}
