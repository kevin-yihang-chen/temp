from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .benchmarks import load_manifest
from .dataset import read_jsonl, write_jsonl
from .metrics import (
    bootstrap_entropy_diagnostic,
    diagnostic_to_dict,
    entropy_diagnostic,
)
from .schema import ActionRecord
from .sharding import SHARD_ALGORITHM, stable_shard_index


_INVARIANT_PROVENANCE_KEYS = (
    "code_revision",
    "manifest_sha256",
    "manifest_limit",
    "manifest_examples_before_sharding",
    "shard_algorithm",
    "shard_count",
    "model",
    "model_revision",
    "ug_framework_revision",
    "scorer",
    "candidate_count",
    "proposer",
    "visual_crop_ratio",
    "visual_cost",
    "generation_seeds",
    "max_new_tokens",
    "min_pixels",
    "max_pixels",
    "dtype",
    "attention_implementation",
    "system_prompt",
    "local_files_only",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shard_directory_name(shard_index: int, shard_count: int) -> str:
    return f"shard-{shard_index:05d}-of-{shard_count:05d}"


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def merge_qwen_rollout_shards(
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    run_root: Path,
    shard_count: int,
    shard_key: str = "state_id",
    shard_namespace: str = "",
    output_path: Path,
    limit: int | None = None,
    expected_code_revision: str | None = None,
    expected_scorer: str | None = None,
    require_resume_audit: bool = False,
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Validate and merge a complete deterministic Qwen rollout shard set."""

    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if shard_key not in {"state_id", "source_id"}:
        raise ValueError("rollout shard key must be state_id or source_id")
    if bootstrap_resamples < 0:
        raise ValueError("bootstrap_resamples must be non-negative")
    actual_manifest_sha256 = sha256_file(manifest_path)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "manifest SHA-256 mismatch: "
            f"expected {expected_manifest_sha256}, got {actual_manifest_sha256}"
        )
    output_sidecars = (
        output_path,
        output_path.with_suffix(".diagnostic.json"),
        output_path.with_suffix(".merge.json"),
    )
    existing = [path for path in output_sidecars if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite merged output: {existing[0]}")

    examples = load_manifest(manifest_path, limit=limit)
    expected_state_order = [example.state.state_id for example in examples]
    if len(set(expected_state_order)) != len(expected_state_order):
        raise ValueError("manifest contains duplicate state_id values")
    source_by_state = {
        example.state.state_id: example.state.source_id for example in examples
    }
    expected_by_shard: dict[int, set[str]] = {
        index: {
            state_id
            for state_id in expected_state_order
            if stable_shard_index(
                source_by_state[state_id] if shard_key == "source_id" else state_id,
                shard_count,
                namespace=shard_namespace,
            )
            == index
        }
        for index in range(shard_count)
    }
    empty_shards = [index for index, states in expected_by_shard.items() if not states]
    if empty_shards:
        raise ValueError(
            f"deterministic partition contains empty shards: {empty_shards}"
        )

    records_by_state: dict[str, list[ActionRecord]] = {}
    shard_audits: list[dict[str, Any]] = []
    invariant_provenance: dict[str, Any] | None = None
    candidate_count: int | None = None
    generation_seeds: Sequence[int | None] | None = None
    for shard_index in range(shard_count):
        shard_dir = run_root / shard_directory_name(shard_index, shard_count)
        rollouts_path = shard_dir / "rollouts.jsonl"
        provenance_path = shard_dir / "rollouts.provenance.json"
        if not rollouts_path.is_file() or not provenance_path.is_file():
            raise FileNotFoundError(
                f"shard {shard_index} output is incomplete: {shard_dir}"
            )
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if not isinstance(provenance, dict):
            raise ValueError(f"invalid shard provenance: {provenance_path}")
        actual_rollouts_sha256 = sha256_file(rollouts_path)
        if provenance.get("output_sha256") != actual_rollouts_sha256:
            raise ValueError(f"shard {shard_index} rollout SHA-256 mismatch")
        if provenance.get("shard_index") != shard_index:
            raise ValueError(f"shard {shard_index} provenance index mismatch")
        if provenance.get("examples") != len(expected_by_shard[shard_index]):
            raise ValueError(f"shard {shard_index} provenance example count mismatch")
        if provenance.get("completed_examples") != len(expected_by_shard[shard_index]):
            raise ValueError(f"shard {shard_index} is not complete")
        current_invariants = {
            key: provenance.get(key) for key in _INVARIANT_PROVENANCE_KEYS
        }
        if invariant_provenance is None:
            invariant_provenance = current_invariants
            candidate_count = int(provenance["candidate_count"])
            generation_seeds = list(provenance["generation_seeds"])
        elif current_invariants != invariant_provenance:
            raise ValueError(f"shard {shard_index} provenance settings differ")
        if provenance.get("manifest_sha256") != actual_manifest_sha256:
            raise ValueError(f"shard {shard_index} manifest hash mismatch")
        if provenance.get("manifest_limit") != limit:
            raise ValueError(f"shard {shard_index} manifest limit mismatch")
        if provenance.get("manifest_examples_before_sharding") != len(examples):
            raise ValueError(f"shard {shard_index} pre-shard count mismatch")
        if provenance.get("shard_algorithm") != SHARD_ALGORITHM:
            raise ValueError(f"shard {shard_index} algorithm mismatch")
        recorded_shard_key = provenance.get("shard_key", "state_id")
        if recorded_shard_key != shard_key:
            raise ValueError(f"shard {shard_index} key mismatch")
        recorded_shard_namespace = provenance.get("shard_namespace", "")
        if recorded_shard_namespace != shard_namespace:
            raise ValueError(f"shard {shard_index} namespace mismatch")
        if provenance.get("shard_count") != shard_count:
            raise ValueError(f"shard {shard_index} count mismatch")
        if (
            expected_code_revision is not None
            and provenance.get("code_revision") != expected_code_revision
        ):
            raise ValueError(f"shard {shard_index} code revision mismatch")
        if expected_scorer is not None and provenance.get("scorer") != expected_scorer:
            raise ValueError(f"shard {shard_index} scorer mismatch")

        records = read_jsonl(rollouts_path)
        resume_audit: dict[str, Any] | None = None
        if require_resume_audit:
            resume_audit_path = shard_dir / "resume.audit.json"
            if not resume_audit_path.is_file():
                raise FileNotFoundError(
                    f"shard {shard_index} resume audit is missing: {resume_audit_path}"
                )
            loaded_resume_audit = json.loads(
                resume_audit_path.read_text(encoding="utf-8")
            )
            if not isinstance(loaded_resume_audit, dict):
                raise ValueError(f"shard {shard_index} resume audit is malformed")
            resume_audit = loaded_resume_audit
            if (
                resume_audit.get("passed") is not True
                or resume_audit.get("rollouts_sha256_before_resume")
                != actual_rollouts_sha256
                or resume_audit.get("rollouts_sha256_after_resume")
                != actual_rollouts_sha256
                or resume_audit.get("records") != len(records)
                or resume_audit.get("resumed_from_records") != len(records)
            ):
                raise ValueError(f"shard {shard_index} resume audit mismatch")
        shard_records_by_state: dict[str, list[ActionRecord]] = defaultdict(list)
        for record in records:
            shard_records_by_state[record.state_id].append(record)
        if set(shard_records_by_state) != expected_by_shard[shard_index]:
            missing = sorted(
                expected_by_shard[shard_index] - set(shard_records_by_state)
            )
            extra = sorted(set(shard_records_by_state) - expected_by_shard[shard_index])
            raise ValueError(
                f"shard {shard_index} state coverage mismatch; "
                f"missing={missing[:3]}, extra={extra[:3]}"
            )
        assert candidate_count is not None and generation_seeds is not None
        expected_records_per_state = (candidate_count + 1) * len(generation_seeds)
        for state_id, state_records in shard_records_by_state.items():
            if len(state_records) != expected_records_per_state:
                raise ValueError(
                    f"state {state_id!r} has {len(state_records)} records; "
                    f"expected {expected_records_per_state}"
                )
            if state_id in records_by_state:
                raise ValueError(f"state {state_id!r} occurs in multiple shards")
            records_by_state[state_id] = state_records
        shard_audits.append(
            {
                "shard_index": shard_index,
                "directory": str(shard_dir.resolve()),
                "states": len(shard_records_by_state),
                "records": len(records),
                "rollouts_sha256": actual_rollouts_sha256,
                "provenance_sha256": sha256_file(provenance_path),
                "resumed_from_records": provenance.get("resumed_from_records"),
                "resume_audit_required": require_resume_audit,
                "resume_audit_sha256": (
                    sha256_file(shard_dir / "resume.audit.json")
                    if resume_audit is not None
                    else None
                ),
            }
        )

    if set(records_by_state) != set(expected_state_order):
        raise ValueError(
            "merged shard set does not exactly cover the selected manifest"
        )
    merged_records = [
        record
        for state_id in expected_state_order
        for record in records_by_state[state_id]
    ]
    write_jsonl(merged_records, output_path)
    merged_sha256 = sha256_file(output_path)
    diagnostic = {
        "point_estimate": diagnostic_to_dict(entropy_diagnostic(merged_records)),
        "bootstrap": bootstrap_entropy_diagnostic(
            merged_records,
            n_resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        ),
    }
    diagnostic_path = output_path.with_suffix(".diagnostic.json")
    _write_json_atomic(diagnostic_path, diagnostic)
    assert invariant_provenance is not None
    audit: dict[str, Any] = {
        "passed": True,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": actual_manifest_sha256,
        "manifest_limit": limit,
        "selected_states": len(expected_state_order),
        "merged_records": len(merged_records),
        "merged_rollouts": str(output_path.resolve()),
        "merged_rollouts_sha256": merged_sha256,
        "diagnostic": str(diagnostic_path.resolve()),
        "diagnostic_sha256": sha256_file(diagnostic_path),
        "shard_algorithm": SHARD_ALGORITHM,
        "shard_key": shard_key,
        "shard_namespace": shard_namespace,
        "shard_count": shard_count,
        "resume_audit_required": require_resume_audit,
        "invariant_provenance": invariant_provenance,
        "source_shards": shard_audits,
    }
    _write_json_atomic(output_path.with_suffix(".merge.json"), audit)
    return audit
