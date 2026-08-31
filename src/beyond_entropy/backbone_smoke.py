from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from .backbone_diagnostic import sha256_file


SMOKE_SCHEMA = "backbone_diagnostic_engineering_smoke_v1"


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON document is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL row at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            result.append(value)
    return result


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"staging file exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_actions(
    rows: list[Mapping[str, Any]], *, expected_states: set[str], context: str
) -> None:
    actions: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        state_id = str(row.get("state_id", ""))
        action_type = str(row.get("action_type", ""))
        actions[state_id][action_type] += 1
    if set(actions) != expected_states:
        raise ValueError(f"{context} state set mismatch")
    expected = Counter({"ANSWER": 1, "ZOOM": 4})
    bad = [state_id for state_id, counts in actions.items() if counts != expected]
    if bad:
        raise ValueError(f"{context} sibling action completeness failed")


def verify_backbone_engineering_smoke(
    *,
    manifest: str | Path,
    rollouts: str | Path,
    rollout_provenance: str | Path,
    rollout_resume_audit: str | Path,
    answer_nll: str | Path,
    answer_nll_provenance: str | Path,
    output: str | Path,
    expected_manifest_sha256: str,
    expected_decisions: int,
    expected_model: str,
    expected_model_revision: str,
    expected_gpu_name: str,
    expected_code_revision: str,
    rollout_seconds: float,
    answer_nll_seconds: float,
) -> dict[str, Any]:
    """Verify an endpoint-blind rollout plus answer-NLL engineering smoke."""

    paths = {
        "manifest": Path(manifest).resolve(),
        "rollouts": Path(rollouts).resolve(),
        "rollout_provenance": Path(rollout_provenance).resolve(),
        "rollout_resume_audit": Path(rollout_resume_audit).resolve(),
        "answer_nll": Path(answer_nll).resolve(),
        "answer_nll_provenance": Path(answer_nll_provenance).resolve(),
    }
    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError(f"smoke verification output exists: {output_path}")
    if expected_decisions <= 0:
        raise ValueError("expected decisions must be positive")
    if not expected_gpu_name.strip() or not expected_code_revision.strip():
        raise ValueError("expected GPU name and code revision must be non-empty")
    for label, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing smoke {label}: {path}")
    if sha256_file(paths["manifest"]) != expected_manifest_sha256:
        raise ValueError("smoke manifest SHA-256 mismatch")

    manifest_rows = _read_jsonl(paths["manifest"])
    if len(manifest_rows) < expected_decisions:
        raise ValueError("smoke manifest is smaller than the expected prefix")
    selected_manifest = manifest_rows[:expected_decisions]
    state_ids = {str(row.get("state_id", "")) for row in selected_manifest}
    source_ids = {str(row.get("source_id", "")) for row in selected_manifest}
    if "" in state_ids or "" in source_ids:
        raise ValueError("smoke manifest prefix has empty source/state IDs")
    if len(state_ids) != expected_decisions or len(source_ids) != expected_decisions:
        raise ValueError("smoke prefix must contain one state per unique source")

    rollout_rows = _read_jsonl(paths["rollouts"])
    if len(rollout_rows) != expected_decisions * 5:
        raise ValueError("smoke rollout record count mismatch")
    _validate_actions(rollout_rows, expected_states=state_ids, context="rollout")
    for row in rollout_rows:
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("smoke rollout metadata is missing")
        backend_names = ["baseline_backend"]
        if row.get("action_type") == "ZOOM":
            backend_names.append("action_backend")
        for name in backend_names:
            backend = metadata.get(name)
            if not isinstance(backend, Mapping):
                raise ValueError(f"smoke rollout is missing {name}")
            if (
                backend.get("model") != expected_model
                or backend.get("model_revision") != expected_model_revision
            ):
                raise ValueError("smoke rollout backend model contract mismatch")

    rollout_sha256 = sha256_file(paths["rollouts"])
    rollout_provenance_payload = _read_object(paths["rollout_provenance"])
    required_rollout_values = {
        "manifest_sha256": expected_manifest_sha256,
        "manifest_limit": expected_decisions,
        "manifest_examples_before_sharding": expected_decisions,
        "shard_count": 1,
        "shard_index": 0,
        "examples": expected_decisions,
        "completed_examples": expected_decisions,
        "resumed_from_records": expected_decisions * 5,
        "candidate_count": 4,
        "model": expected_model,
        "model_revision": expected_model_revision,
        "code_revision": expected_code_revision,
        "output_sha256": rollout_sha256,
        "scorer": "screenqa",
    }
    for key, expected in required_rollout_values.items():
        if rollout_provenance_payload.get(key) != expected:
            raise ValueError(f"smoke rollout provenance mismatch: {key}")
    resume = _read_object(paths["rollout_resume_audit"])
    if (
        resume.get("passed") is not True
        or resume.get("records") != expected_decisions * 5
        or resume.get("resumed_from_records") != expected_decisions * 5
        or resume.get("rollouts_sha256_before_resume") != rollout_sha256
        or resume.get("rollouts_sha256_after_resume") != rollout_sha256
    ):
        raise ValueError("smoke rollout resume audit mismatch")

    nll_rows = _read_jsonl(paths["answer_nll"])
    if len(nll_rows) != expected_decisions * 5:
        raise ValueError("smoke answer-NLL record count mismatch")
    _validate_actions(nll_rows, expected_states=state_ids, context="answer NLL")
    config_hashes: set[str] = set()
    forbidden_target_fields = {"target", "target_answer", "target_text", "raw_target"}
    for row in nll_rows:
        if forbidden_target_fields.intersection(row):
            raise ValueError("smoke answer-NLL artifact contains a raw target field")
        mean_nll = float(row.get("answer_mean_nll", math.nan))
        token_count = int(row.get("answer_token_count", 0))
        if not math.isfinite(mean_nll) or mean_nll < 0 or token_count <= 0:
            raise ValueError("smoke answer-NLL value is invalid")
        config_hashes.add(str(row.get("config_sha256", "")))
    if "" in config_hashes or len(config_hashes) != 1:
        raise ValueError("smoke answer-NLL configuration is not unique")

    nll_sha256 = sha256_file(paths["answer_nll"])
    nll_provenance = _read_object(paths["answer_nll_provenance"])
    measurement = nll_provenance.get("measurement_config")
    if not isinstance(measurement, Mapping):
        raise ValueError("smoke answer-NLL measurement metadata is missing")
    accelerator = str(measurement.get("accelerator_name", ""))
    if expected_gpu_name.casefold() not in accelerator.casefold():
        raise ValueError("smoke accelerator name mismatch")
    required_nll_values = {
        "manifest_sha256": expected_manifest_sha256,
        "rollouts_sha256": rollout_sha256,
        "output_sha256": nll_sha256,
        "decisions": expected_decisions,
        "records": expected_decisions * 5,
        "sources": expected_decisions,
        "shard_count": 1,
        "shard_index": 0,
        "raw_targets_written": False,
        "model": expected_model,
        "model_revision": expected_model_revision,
        "code_revision": expected_code_revision,
    }
    for key, expected in required_nll_values.items():
        if nll_provenance.get(key) != expected:
            raise ValueError(f"smoke answer-NLL provenance mismatch: {key}")
    if not math.isfinite(rollout_seconds) or rollout_seconds <= 0:
        raise ValueError("rollout time must be finite and positive")
    if not math.isfinite(answer_nll_seconds) or answer_nll_seconds <= 0:
        raise ValueError("answer-NLL time must be finite and positive")

    result = {
        "schema": SMOKE_SCHEMA,
        "passed": True,
        "scientific_status": (
            "endpoint-blind engineering smoke; task metrics were not computed or used"
        ),
        "population": {
            "decisions": expected_decisions,
            "sources": expected_decisions,
            "records_per_artifact": expected_decisions * 5,
        },
        "model": {"name": expected_model, "revision": expected_model_revision},
        "accelerator_name": accelerator,
        "timing_seconds": {
            "rollout_including_resume": rollout_seconds,
            "answer_nll_including_resume": answer_nll_seconds,
            "total": rollout_seconds + answer_nll_seconds,
        },
        "inputs": {
            "manifest_sha256": expected_manifest_sha256,
            "rollouts_sha256": rollout_sha256,
            "answer_nll_sha256": nll_sha256,
            "rollout_provenance_sha256": sha256_file(paths["rollout_provenance"]),
            "rollout_resume_audit_sha256": sha256_file(paths["rollout_resume_audit"]),
            "answer_nll_provenance_sha256": sha256_file(paths["answer_nll_provenance"]),
            "code_revision": expected_code_revision,
            "verifier_module_sha256": sha256_file(Path(__file__).resolve()),
        },
        "outcome_use": {
            "task_endpoints_computed": False,
            "hardware_selected_from_task_outcomes": False,
            "protected_role_inputs_used": False,
        },
    }
    _atomic_json(output_path, result)
    return result
