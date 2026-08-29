from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

from beyond_entropy.action_value import factorized_acquisition_calibration_rows
from beyond_entropy.dataset import read_jsonl
from beyond_entropy.docvqa_calibration import (
    EXPECTED_SCIENTIFIC_STATUS,
    MODEL_REVISION,
    calibrate_frozen_candidate_rows,
    validate_calibration_bundle,
    validate_calibration_feature_metadata,
    validate_calibration_preoutcome_gate,
)
from beyond_entropy.docvqa_candidate_freeze import PROTOCOL_SHA256
from beyond_entropy.docvqa_train_allocation import sha256_file
from beyond_entropy.manifest_audit import audit_manifest
from beyond_entropy.qwen_semantic import (
    load_semantic_feature_dataset,
    validate_semantic_feature_dataset,
)
from beyond_entropy.rollout_audit import audit_sibling_rollout_bank


def _load_mapping(path: Path, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _require(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"DocVQA calibration input mismatch for {name}")


def _require_hash(path: Path, expected: str, name: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"DocVQA calibration input is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"DocVQA calibration {name} SHA-256 mismatch")
    return actual


def _tracked_revision(repo_dir: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("tracked worktree must be clean before DocVQA calibration")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _clean_manifest_audit(report: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if not key.startswith("_")}


def _write_bundle_exclusive(
    output_dir: Path,
    calibration: Mapping[str, Any],
    model: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"DocVQA calibration output already exists: {output_dir}")
    staging = output_dir.with_name(output_dir.name + ".tmp")
    if staging.exists():
        raise FileExistsError(f"stale DocVQA calibration staging directory: {staging}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    payloads: dict[str, Mapping[str, Any]] = {
        "calibration.json": calibration,
        "model.json": model,
    }
    for name, payload in payloads.items():
        with (staging / name).open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
    audit_payload = {
        **dict(audit),
        "calibration_sha256": sha256_file(staging / "calibration.json"),
        "model_sha256": sha256_file(staging / "model.json"),
    }
    with (staging / "calibration.audit.json").open("x", encoding="utf-8") as handle:
        json.dump(audit_payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(staging, output_dir)
    return audit_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute the frozen DocVQA-train factorized-v2 fixed sequence"
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-audit", type=Path, required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--expected-candidate-audit-sha256", required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--allocation-audit", type=Path, required=True)
    parser.add_argument("--expected-allocation-sha256", required=True)
    parser.add_argument("--expected-allocation-audit-sha256", required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-manifest-provenance-sha256", required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--expected-rollouts-sha256", required=True)
    parser.add_argument("--rollout-audit", type=Path, required=True)
    parser.add_argument("--expected-rollout-audit-sha256", required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--expected-features-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-code-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    candidate_path = args.candidate.resolve()
    candidate_audit_path = args.candidate_audit.resolve()
    allocation_path = args.allocation.resolve()
    allocation_audit_path = args.allocation_audit.resolve()
    manifest_dir = args.manifest_dir.resolve()
    manifest_path = manifest_dir / "manifest.jsonl"
    manifest_provenance_path = manifest_dir / "manifest.provenance.json"
    rollouts_path = args.rollouts.resolve()
    rollout_audit_path = args.rollout_audit.resolve()
    features_path = args.features.resolve()
    protocol_path = args.protocol.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() or output_dir.with_name(output_dir.name + ".tmp").exists():
        raise FileExistsError("DocVQA calibration output or staging path already exists")

    core_hashes = {
        "candidate": _require_hash(
            candidate_path,
            args.expected_candidate_sha256,
            "candidate",
        ),
        "candidate_audit": _require_hash(
            candidate_audit_path,
            args.expected_candidate_audit_sha256,
            "candidate audit",
        ),
        "allocation": _require_hash(
            allocation_path,
            args.expected_allocation_sha256,
            "allocation",
        ),
        "allocation_audit": _require_hash(
            allocation_audit_path,
            args.expected_allocation_audit_sha256,
            "allocation audit",
        ),
        "protocol": _require_hash(protocol_path, PROTOCOL_SHA256, "protocol"),
    }
    repo_dir = Path(__file__).resolve().parents[1]
    code_revision = _tracked_revision(repo_dir)
    _require(code_revision, args.expected_code_revision, "code revision")
    candidate = _load_mapping(candidate_path, "candidate")
    candidate_audit = _load_mapping(candidate_audit_path, "candidate audit")
    allocation = _load_mapping(allocation_path, "allocation")
    allocation_audit = _load_mapping(allocation_audit_path, "allocation audit")
    validate_calibration_preoutcome_gate(
        candidate,
        candidate_audit,
        allocation,
        allocation_audit,
        candidate_sha256=core_hashes["candidate"],
        allocation_sha256=core_hashes["allocation"],
        code_revision=code_revision,
    )

    outcome_hashes = {
        "manifest": _require_hash(
            manifest_path,
            args.expected_manifest_sha256,
            "manifest",
        ),
        "manifest_provenance": _require_hash(
            manifest_provenance_path,
            args.expected_manifest_provenance_sha256,
            "manifest provenance",
        ),
        "rollouts": _require_hash(
            rollouts_path,
            args.expected_rollouts_sha256,
            "rollouts",
        ),
        "rollout_audit": _require_hash(
            rollout_audit_path,
            args.expected_rollout_audit_sha256,
            "rollout audit",
        ),
        "features": _require_hash(
            features_path,
            args.expected_features_sha256,
            "features",
        ),
    }
    raw_manifest_audit = audit_manifest(manifest_dir)
    manifest_audit = _clean_manifest_audit(raw_manifest_audit)
    rollout_audit = _load_mapping(rollout_audit_path, "rollout audit")
    manifest_count = validate_calibration_bundle(
        candidate,
        candidate_audit,
        allocation,
        allocation_audit,
        manifest_audit,
        rollout_audit,
        candidate_sha256=core_hashes["candidate"],
        candidate_audit_sha256=core_hashes["candidate_audit"],
        allocation_sha256=core_hashes["allocation"],
        allocation_audit_sha256=core_hashes["allocation_audit"],
        manifest_sha256=outcome_hashes["manifest"],
        manifest_provenance_sha256=outcome_hashes["manifest_provenance"],
        rollouts_sha256=outcome_hashes["rollouts"],
        code_revision=code_revision,
    )

    recomputed_rollout_audit = audit_sibling_rollout_bank(
        manifest_path,
        rollouts_path,
        expected_manifest_sha256=outcome_hashes["manifest"],
        expected_states=manifest_count,
        expected_candidate_count=4,
        expected_model_revision=MODEL_REVISION,
        expected_scientific_status=EXPECTED_SCIENTIFIC_STATUS,
    )
    recomputed_rollout_audit.update(
        {
            "candidate_sha256": core_hashes["candidate"],
            "candidate_audit_sha256": core_hashes["candidate_audit"],
            "allocation_sha256": core_hashes["allocation"],
            "allocation_audit_sha256": core_hashes["allocation_audit"],
            "manifest_provenance_sha256": outcome_hashes["manifest_provenance"],
            "protocol_sha256": core_hashes["protocol"],
            "manifest_audit": manifest_audit,
            "formal_outcomes_used": False,
        }
    )
    if recomputed_rollout_audit != rollout_audit:
        raise ValueError("stored DocVQA rollout audit differs from recomputation")

    records = read_jsonl(rollouts_path)
    features = load_semantic_feature_dataset(features_path)
    validate_semantic_feature_dataset(features, records, require_outcomes=False)
    validate_calibration_feature_metadata(
        features,
        rollouts_sha256=outcome_hashes["rollouts"],
        code_revision=code_revision,
        expected_decisions=manifest_count,
    )
    semantic_decisions = {
        (str(decision["state_id"]), str(decision["replicate_id"])): decision
        for decision in features["decisions"]
    }
    rows = factorized_acquisition_calibration_rows(
        candidate,
        records,
        semantic_decisions=semantic_decisions,
    )
    if len(rows) != manifest_count:
        raise RuntimeError("DocVQA calibration rows do not cover every decision")
    if len({row.source_id for row in rows}) != 2500:
        raise RuntimeError("DocVQA calibration rows do not cover 2,500 sources")

    all_hashes = {**core_hashes, **outcome_hashes}
    run = {
        "code_revision": code_revision,
        **{
            key: value
            for name, path in (
                ("candidate", candidate_path),
                ("candidate_audit", candidate_audit_path),
                ("allocation", allocation_path),
                ("allocation_audit", allocation_audit_path),
                ("manifest", manifest_path),
                ("manifest_provenance", manifest_provenance_path),
                ("rollouts", rollouts_path),
                ("rollout_audit", rollout_audit_path),
                ("features", features_path),
                ("protocol", protocol_path),
            )
            for key, value in (
                (name, str(path)),
                (f"{name}_sha256", all_hashes[name]),
            )
        },
    }
    calibration, calibrated_model = calibrate_frozen_candidate_rows(
        candidate,
        rows,
        expected_sources=2500,
        run_provenance=run,
    )

    staging_audit = {
        "passed": True,
        "scientific_status": (
            "DocVQA-train fixed sequence executed once; formal role remains sealed"
        ),
        "selection_status": calibration["selection_status"],
        "selected_threshold": calibration["selected_threshold"],
        "tested_threshold_count": calibration["tested_threshold_count"],
        "stopping_threshold": calibration["stopping_threshold"],
        "n_sources": calibration["n_sources"],
        "n_decisions": calibration["n_decisions"],
        "inputs": {f"{name}_sha256": value for name, value in all_hashes.items()},
        "code_revision": code_revision,
        "ranker_training_outcomes_used": True,
        "calibration_outcomes_used": True,
        "formal_outcomes_used": False,
    }
    final_audit = _write_bundle_exclusive(
        output_dir,
        calibration,
        calibrated_model,
        staging_audit,
    )
    print(json.dumps(final_audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
