from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .sharding import stable_shard_index


PLAN_SCHEMA = "qwen_rollout_runtime_replay_plan_v1"
REPAIR_SCHEMA = "qwen_rollout_runtime_repair_v1"
COMPLETION_SCHEMA = "qwen_rollout_runtime_replay_completion_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON document: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSONL document: {path}") from exc
    return rows


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"staging file exists: {temporary}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(value, allow_nan=False, indent=2, sort_keys=True))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"staging file exists: {temporary}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, allow_nan=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_runtime_replay_plan(
    *,
    manifest: str | Path,
    rollout_root: str | Path,
    replay_root: str | Path,
    expected_manifest_sha256: str,
    shard_count: int = 4,
) -> dict[str, Any]:
    """Select one already-computed state per rollout shard for exact replay."""

    _require(shard_count > 0, "shard_count must be positive")
    manifest_path = Path(manifest).resolve()
    rollout_path = Path(rollout_root).resolve()
    destination = Path(replay_root).resolve()
    plan_path = destination / "plan.json"
    _require(not plan_path.exists(), "runtime replay plan already exists")
    actual_manifest_sha256 = sha256_file(manifest_path)
    _require(
        actual_manifest_sha256 == expected_manifest_sha256,
        "runtime replay manifest SHA-256 mismatch",
    )
    rows = _read_jsonl(manifest_path)
    _require(rows, "runtime replay manifest is empty")
    selected: dict[int, dict[str, Any]] = {}
    for row in rows:
        state_id = str(row.get("state_id", ""))
        _require(state_id, "runtime replay manifest row lacks state_id")
        index = stable_shard_index(state_id, shard_count)
        selected.setdefault(index, row)
    _require(
        set(selected) == set(range(shard_count)),
        "runtime replay could not select one state per shard",
    )

    entries: list[dict[str, Any]] = []
    for shard_index in range(shard_count):
        row = json.loads(json.dumps(selected[shard_index]))
        state_id = str(row["state_id"])
        image_path = Path(str(row["image_path"]))
        if not image_path.is_absolute():
            image_path = (manifest_path.parent / image_path).resolve()
        _require(image_path.is_file(), f"runtime replay image is missing: {image_path}")
        row["image_path"] = str(image_path)
        probe_dir = destination / f"probe-{shard_index:05d}-of-{shard_count:05d}"
        probe_manifest = probe_dir / "manifest.jsonl"
        _atomic_jsonl(probe_manifest, [row])

        full_rollouts = (
            rollout_path
            / f"shard-{shard_index:05d}-of-{shard_count:05d}"
            / "rollouts.jsonl"
        )
        full_provenance = full_rollouts.with_suffix(".provenance.json")
        _require(full_rollouts.is_file(), f"full rollout shard is missing: {full_rollouts}")
        _require(full_provenance.is_file(), f"full provenance is missing: {full_provenance}")
        main_rows = [
            item for item in _read_jsonl(full_rollouts) if item.get("state_id") == state_id
        ]
        _require(len(main_rows) == 5, "runtime replay state must have five main records")
        entries.append(
            {
                "shard_index": shard_index,
                "state_id": state_id,
                "source_id": str(row.get("source_id", "")),
                "probe_manifest": str(probe_manifest),
                "probe_manifest_sha256": sha256_file(probe_manifest),
                "probe_rollouts": str(probe_dir / "rollouts.jsonl"),
                "full_rollouts": str(full_rollouts),
                "full_rollouts_sha256": sha256_file(full_rollouts),
                "full_provenance": str(full_provenance),
            }
        )
    plan = {
        "schema": PLAN_SCHEMA,
        "manifest": str(manifest_path),
        "manifest_sha256": actual_manifest_sha256,
        "shard_count": shard_count,
        "selection": "first manifest-order state assigned to each sha256-state-id-v1 shard",
        "entries": entries,
        "target_contents_recorded": False,
    }
    _atomic_json(plan_path, plan)
    return plan


def _validate_runtime(value: object) -> dict[str, Any]:
    _require(isinstance(value, dict), "runtime replay measurement is missing")
    runtime = dict(value)
    _require(runtime.get("accelerator_name") == "NVIDIA H800", "replay GPU mismatch")
    _require(runtime.get("compute_capability") == [9, 0], "replay capability mismatch")
    _require(runtime.get("requested_dtype") == "bfloat16", "replay dtype mismatch")
    _require(runtime.get("parameter_dtype") == "torch.bfloat16", "replay parameter dtype mismatch")
    _require(runtime.get("attention_implementation") == "sdpa", "replay attention mismatch")
    _require(runtime.get("actual_attention_implementation") == "sdpa", "replay actual attention mismatch")
    _require(int(runtime.get("peak_allocated_bytes", 0)) > 0, "replay allocated peak missing")
    _require(int(runtime.get("peak_reserved_bytes", 0)) > 0, "replay reserved peak missing")
    return runtime


def repair_runtime_from_replays(
    *,
    plan: str | Path,
    code_revision: str,
    prior_job_ids: Sequence[str],
) -> dict[str, Any]:
    """Attach transparent replay telemetry after exact record equivalence checks."""

    plan_path = Path(plan).resolve()
    payload = _read_json(plan_path)
    _require(payload.get("schema") == PLAN_SCHEMA, "runtime replay plan schema mismatch")
    entries = payload.get("entries")
    _require(isinstance(entries, list) and entries, "runtime replay plan has no entries")
    completion_path = plan_path.with_name("replay.complete.json")
    _require(not completion_path.exists(), "runtime replay completion already exists")
    repairs: list[dict[str, Any]] = []
    for raw_entry in entries:
        _require(isinstance(raw_entry, dict), "runtime replay entry is malformed")
        entry = dict(raw_entry)
        state_id = str(entry["state_id"])
        probe_rollouts = Path(str(entry["probe_rollouts"]))
        probe_provenance = probe_rollouts.with_suffix(".provenance.json")
        full_rollouts = Path(str(entry["full_rollouts"]))
        full_provenance = Path(str(entry["full_provenance"]))
        _require(sha256_file(full_rollouts) == entry["full_rollouts_sha256"], "full rollout changed before replay repair")
        probe_rows = _read_jsonl(probe_rollouts)
        main_rows = [row for row in _read_jsonl(full_rollouts) if row.get("state_id") == state_id]
        _require(len(probe_rows) == 5 and probe_rows == main_rows, "runtime replay action records differ from main rollout")
        _require(
            not any("target" in key.lower() for row in probe_rows for key in row),
            "runtime replay wrote a raw target field",
        )
        probe_meta = _read_json(probe_provenance)
        _require(probe_meta.get("output_sha256") == sha256_file(probe_rollouts), "probe rollout hash mismatch")
        _require(probe_meta.get("model") == "Qwen/Qwen2.5-VL-7B-Instruct", "probe model mismatch")
        _require(probe_meta.get("model_revision") == "cc594898137f460bfe9f0759e9844b3ce807cfb5", "probe revision mismatch")
        runtime = _validate_runtime(probe_meta.get("runtime_measurement"))

        full_meta = _read_json(full_provenance)
        _require(full_meta.get("output_sha256") == sha256_file(full_rollouts), "full provenance output hash mismatch")
        _require(full_meta.get("completed_examples") == full_meta.get("examples"), "full rollout shard is incomplete")
        _require(full_meta.get("runtime_measurement") is None, "full rollout runtime was already populated")
        before_sha256 = sha256_file(full_provenance)
        recovery = {
            "schema": REPAIR_SCHEMA,
            "source": "deterministic_one_state_h800_replay",
            "probe_state_id": state_id,
            "probe_rollouts_sha256": sha256_file(probe_rollouts),
            "probe_provenance_sha256": sha256_file(probe_provenance),
            "exact_five_record_match": True,
            "original_process_peak_reconstructed": False,
            "reason": "original full rollout completed before a positive-bootstrap provenance could be written",
            "recovery_code_revision": code_revision,
            "prior_job_ids": list(prior_job_ids),
        }
        full_meta["runtime_measurement"] = runtime
        full_meta["runtime_measurement_recovery"] = recovery
        _atomic_json(full_provenance, full_meta)
        after_sha256 = sha256_file(full_provenance)
        audit_path = full_provenance.with_name("runtime-recovery.audit.json")
        audit = {
            "schema": REPAIR_SCHEMA,
            "passed": True,
            "state_id": state_id,
            "full_rollouts_sha256": sha256_file(full_rollouts),
            "full_provenance_sha256_before": before_sha256,
            "full_provenance_sha256_after": after_sha256,
            "probe_rollouts_sha256": recovery["probe_rollouts_sha256"],
            "probe_provenance_sha256": recovery["probe_provenance_sha256"],
            "exact_five_record_match": True,
            "runtime_measurement": runtime,
            "recovery": recovery,
        }
        _atomic_json(audit_path, audit)
        repairs.append(
            {
                "shard_index": int(entry["shard_index"]),
                "state_id": state_id,
                "full_rollouts_sha256": audit["full_rollouts_sha256"],
                "full_provenance_sha256": after_sha256,
                "repair_audit": str(audit_path),
                "repair_audit_sha256": sha256_file(audit_path),
                "peak_allocated_bytes": runtime["peak_allocated_bytes"],
                "peak_reserved_bytes": runtime["peak_reserved_bytes"],
            }
        )
    completion = {
        "schema": COMPLETION_SCHEMA,
        "passed": True,
        "plan": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "code_revision": code_revision,
        "prior_job_ids": list(prior_job_ids),
        "repairs": repairs,
        "protected_role_inputs_used": False,
        "target_contents_recorded": False,
        "original_process_peak_reconstructed": False,
    }
    _atomic_json(completion_path, completion)
    return completion
