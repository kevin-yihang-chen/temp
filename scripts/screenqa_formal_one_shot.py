#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from beyond_entropy.benchmarks import load_manifest
from beyond_entropy.sharding import SHARD_ALGORITHM, stable_shard_index


SCIENTIFIC_STATUS = (
    "ScreenQA one-shot formal sibling bank; frozen calibrated policy and "
    "implementation; no target-derived tuning"
)
LEDGER_NAME = "formal-start.json"
COMPLETION_NAME = "formal-complete.json"
HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
GIT_REVISION = re.compile(r"[0-9a-f]{40,64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")


def _require_hash(value: str, name: str) -> None:
    if HEX_SHA256.fullmatch(value) is None:
        raise ValueError(f"invalid {name} SHA-256")


def build_contract(
    *,
    shard_dir: Path,
    manifest: Path,
    expected_manifest_sha256: str,
    manifest_audit_sha256: str,
    candidate_bundle_sha256: str,
    calibration_bundle_sha256: str,
    code_revision: str,
    shard_index: int,
    shard_count: int,
    expected_total_states: int,
) -> dict[str, Any]:
    for value, name in (
        (expected_manifest_sha256, "manifest"),
        (manifest_audit_sha256, "manifest audit"),
        (candidate_bundle_sha256, "candidate bundle"),
        (calibration_bundle_sha256, "calibration bundle"),
    ):
        _require_hash(value, name)
    if GIT_REVISION.fullmatch(code_revision) is None:
        raise ValueError("invalid formal code revision")
    if shard_count != 4 or not 0 <= shard_index < shard_count:
        raise ValueError("ScreenQA formal collection requires shards 0-3 of four")
    actual_manifest_sha256 = sha256_file(manifest)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError("ScreenQA formal manifest SHA-256 mismatch")
    examples = load_manifest(manifest)
    state_ids = [example.state.state_id for example in examples]
    if len(state_ids) != expected_total_states or len(set(state_ids)) != len(state_ids):
        raise ValueError("ScreenQA formal manifest state coverage mismatch")
    shard_states = sum(
        stable_shard_index(state_id, shard_count) == shard_index
        for state_id in state_ids
    )
    if shard_states <= 0:
        raise ValueError("ScreenQA formal deterministic shard is empty")
    return {
        "schema_version": 1,
        "scientific_status": SCIENTIFIC_STATUS,
        "execution_semantics": (
            "one-shot formal outcome collection with exact-contract checkpoint "
            "recovery only"
        ),
        "formal_outcomes_generation_started": True,
        "formal_outcomes_used_for_tuning": False,
        "resume_policy": (
            "preserve every completed state and resume only missing states under "
            "the identical locked contract"
        ),
        "shard_directory": str(shard_dir.resolve()),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": expected_manifest_sha256,
        "manifest_audit_sha256": manifest_audit_sha256,
        "candidate_bundle_sha256": candidate_bundle_sha256,
        "calibration_bundle_sha256": calibration_bundle_sha256,
        "code_revision": code_revision,
        "expected_total_states": expected_total_states,
        "expected_shard_states": shard_states,
        "expected_shard_records": shard_states * 5,
        "shard_algorithm": SHARD_ALGORITHM,
        "shard_count": shard_count,
        "shard_index": shard_index,
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "model_revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
        "scorer": "screenqa",
        "candidate_count": 4,
        "proposer": "ug-grid",
        "generation_seeds": [0],
        "bootstrap_seed": 20260831,
    }


def open_shard(shard_dir: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    ledger = shard_dir / LEDGER_NAME
    completion = shard_dir / COMPLETION_NAME
    if completion.exists():
        raise FileExistsError(
            f"completed ScreenQA formal shard cannot be reopened: {completion}"
        )
    if ledger.exists():
        if _load_mapping(ledger) != dict(contract):
            raise ValueError("existing ScreenQA formal start ledger changed")
        return {
            "passed": True,
            "ledger_created": False,
            "exact_contract_checkpoint_recovery": True,
            "ledger": str(ledger.resolve()),
            "ledger_sha256": sha256_file(ledger),
        }
    if shard_dir.exists() and any(shard_dir.iterdir()):
        raise ValueError("formal shard has files but no authoritative start ledger")
    shard_dir.mkdir(parents=True, exist_ok=True)
    _write_exclusive(ledger, contract)
    return {
        "passed": True,
        "ledger_created": True,
        "exact_contract_checkpoint_recovery": False,
        "ledger": str(ledger.resolve()),
        "ledger_sha256": sha256_file(ledger),
    }


def _validate_provenance(
    provenance: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    rollouts_sha256: str,
    require_completed_resume: bool,
) -> None:
    expected = {
        "scientific_status": SCIENTIFIC_STATUS,
        "code_revision": contract["code_revision"],
        "manifest_sha256": contract["manifest_sha256"],
        "manifest_limit": None,
        "manifest_examples_before_sharding": contract["expected_total_states"],
        "shard_algorithm": SHARD_ALGORITHM,
        "shard_count": contract["shard_count"],
        "shard_index": contract["shard_index"],
        "model": contract["model"],
        "model_revision": contract["model_revision"],
        "scorer": "screenqa",
        "examples": contract["expected_shard_states"],
        "completed_examples": contract["expected_shard_states"],
        "candidate_count": 4,
        "proposer": "ug-grid",
        "visual_crop_ratio": 2.0,
        "visual_cost": 1.0,
        "generation_seeds": [0],
        "bootstrap_seed": 20260831,
        "max_new_tokens": 32,
        "min_pixels": 200704,
        "max_pixels": 602112,
        "attention_implementation": "sdpa",
        "system_prompt": "You are a helpful assistant.",
        "local_files_only": True,
        "output_sha256": rollouts_sha256,
    }
    for key, expected_value in expected.items():
        if provenance.get(key) != expected_value:
            raise ValueError(f"ScreenQA formal provenance {key} mismatch")
    resumed = provenance.get("resumed_from_records")
    if not isinstance(resumed, int) or not 0 <= resumed <= int(
        contract["expected_shard_records"]
    ):
        raise ValueError("ScreenQA formal provenance resume count is invalid")
    if require_completed_resume and resumed != contract["expected_shard_records"]:
        raise ValueError("ScreenQA formal no-op resume did not start from completion")


def completion_payload(
    shard_dir: Path, contract: Mapping[str, Any]
) -> dict[str, Any]:
    ledger = shard_dir / LEDGER_NAME
    if not ledger.is_file() or _load_mapping(ledger) != dict(contract):
        raise ValueError("ScreenQA formal completion lacks the exact start ledger")
    rollouts = shard_dir / "rollouts.jsonl"
    provenance_path = shard_dir / "rollouts.provenance.json"
    first_provenance_path = shard_dir / "rollouts.first-pass.provenance.json"
    resume_audit_path = shard_dir / "resume.audit.json"
    diagnostic_path = shard_dir / "rollouts.diagnostic.json"
    for path in (
        rollouts,
        provenance_path,
        first_provenance_path,
        resume_audit_path,
        diagnostic_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"incomplete ScreenQA formal shard: {path}")
    records = sum(1 for line in rollouts.open(encoding="utf-8") if line.strip())
    if records != contract["expected_shard_records"]:
        raise ValueError("ScreenQA formal shard record count mismatch")
    rollouts_sha256 = sha256_file(rollouts)
    provenance = _load_mapping(provenance_path)
    first_provenance = _load_mapping(first_provenance_path)
    _validate_provenance(
        provenance,
        contract,
        rollouts_sha256=rollouts_sha256,
        require_completed_resume=True,
    )
    _validate_provenance(
        first_provenance,
        contract,
        rollouts_sha256=rollouts_sha256,
        require_completed_resume=False,
    )
    resume_audit = _load_mapping(resume_audit_path)
    expected_resume = {
        "passed": True,
        "rollouts_sha256_before_resume": rollouts_sha256,
        "rollouts_sha256_after_resume": rollouts_sha256,
        "records": records,
        "examples": contract["expected_shard_states"],
        "resumed_from_records": records,
    }
    for key, expected_value in expected_resume.items():
        if resume_audit.get(key) != expected_value:
            raise ValueError(f"ScreenQA formal resume audit {key} mismatch")
    return {
        "schema_version": 1,
        "passed": True,
        "scientific_status": SCIENTIFIC_STATUS,
        "one_shot_formal_shard_complete": True,
        "formal_outcomes_used_for_tuning": False,
        "exact_contract_checkpoint_recovery_only": True,
        "shard_index": contract["shard_index"],
        "shard_count": contract["shard_count"],
        "states": contract["expected_shard_states"],
        "records": records,
        "ledger_sha256": sha256_file(ledger),
        "rollouts_sha256": rollouts_sha256,
        "provenance_sha256": sha256_file(provenance_path),
        "first_pass_provenance_sha256": sha256_file(first_provenance_path),
        "resume_audit_sha256": sha256_file(resume_audit_path),
        "diagnostic_sha256": sha256_file(diagnostic_path),
        "manifest_sha256": contract["manifest_sha256"],
        "manifest_audit_sha256": contract["manifest_audit_sha256"],
        "candidate_bundle_sha256": contract["candidate_bundle_sha256"],
        "calibration_bundle_sha256": contract["calibration_bundle_sha256"],
        "code_revision": contract["code_revision"],
    }


def complete_shard(shard_dir: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    completion = shard_dir / COMPLETION_NAME
    if completion.exists():
        raise FileExistsError(
            f"refusing to overwrite completed ScreenQA formal shard: {completion}"
        )
    payload = completion_payload(shard_dir, contract)
    _write_exclusive(completion, payload)
    return payload


def verify_shard_completion(
    shard_dir: Path, contract: Mapping[str, Any]
) -> dict[str, Any]:
    completion = shard_dir / COMPLETION_NAME
    if not completion.is_file():
        raise FileNotFoundError(f"ScreenQA formal completion marker missing: {completion}")
    expected = completion_payload(shard_dir, contract)
    if _load_mapping(completion) != expected:
        raise ValueError("ScreenQA formal completion marker changed")
    return expected


def _add_contract_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--manifest-audit-sha256", required=True)
    parser.add_argument("--candidate-bundle-sha256", required=True)
    parser.add_argument("--calibration-bundle-sha256", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--expected-total-states", type=int, required=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Guard one-shot ScreenQA formal shards and exact checkpoint recovery"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("open", "complete", "verify"):
        _add_contract_arguments(subparsers.add_parser(command))
    args = parser.parse_args()
    shard_dir = args.shard_dir.resolve()
    contract = build_contract(
        shard_dir=shard_dir,
        manifest=args.manifest.resolve(),
        expected_manifest_sha256=args.expected_manifest_sha256,
        manifest_audit_sha256=args.manifest_audit_sha256,
        candidate_bundle_sha256=args.candidate_bundle_sha256,
        calibration_bundle_sha256=args.calibration_bundle_sha256,
        code_revision=args.code_revision,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        expected_total_states=args.expected_total_states,
    )
    if args.command == "open":
        result = open_shard(shard_dir, contract)
    elif args.command == "complete":
        result = complete_shard(shard_dir, contract)
    else:
        result = verify_shard_completion(shard_dir, contract)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
