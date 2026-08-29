from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from beyond_entropy.docvqa_calibration import (
    EXPECTED_SCIENTIFIC_STATUS,
    MODEL_REVISION,
    validate_calibration_manifest,
    validate_calibration_preoutcome_gate,
)
from beyond_entropy.docvqa_candidate_freeze import PROTOCOL_SHA256
from beyond_entropy.docvqa_train_allocation import sha256_file
from beyond_entropy.manifest_audit import audit_manifest
from beyond_entropy.rollout_audit import audit_sibling_rollout_bank


def _load_mapping(path: Path, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _require_hash(path: Path, expected: str, name: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"DocVQA rollout audit input is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"DocVQA rollout audit {name} SHA-256 mismatch")
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
        raise ValueError("tracked worktree must be clean before DocVQA rollout audit")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _clean_manifest_audit(report: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if not key.startswith("_")}


def _write_frozen(
    path: Path,
    payload: Mapping[str, Any],
    *,
    resume: bool,
) -> None:
    serialized = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if not resume:
            raise FileExistsError(f"refusing to overwrite DocVQA rollout audit: {path}")
        if path.read_text(encoding="utf-8") != serialized:
            raise ValueError("existing DocVQA rollout audit differs from recomputation")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the frozen DocVQA-train calibration sibling bank"
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
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-code-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    candidate_path = args.candidate.resolve()
    candidate_audit_path = args.candidate_audit.resolve()
    allocation_path = args.allocation.resolve()
    allocation_audit_path = args.allocation_audit.resolve()
    manifest_dir = args.manifest_dir.resolve()
    manifest_path = manifest_dir / "manifest.jsonl"
    manifest_provenance_path = manifest_dir / "manifest.provenance.json"
    rollouts_path = args.rollouts.resolve()
    protocol_path = args.protocol.resolve()
    output_path = args.output.resolve()
    if output_path.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite DocVQA rollout audit: {output_path}")

    hashes = {
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
        "protocol": _require_hash(protocol_path, PROTOCOL_SHA256, "protocol"),
    }
    repo_dir = Path(__file__).resolve().parents[1]
    code_revision = _tracked_revision(repo_dir)
    if code_revision != args.expected_code_revision:
        raise ValueError("DocVQA rollout audit code revision mismatch")

    candidate = _load_mapping(candidate_path, "candidate")
    candidate_audit = _load_mapping(candidate_audit_path, "candidate audit")
    allocation = _load_mapping(allocation_path, "allocation")
    allocation_audit = _load_mapping(allocation_audit_path, "allocation audit")
    validate_calibration_preoutcome_gate(
        candidate,
        candidate_audit,
        allocation,
        allocation_audit,
        candidate_sha256=hashes["candidate"],
        allocation_sha256=hashes["allocation"],
        code_revision=code_revision,
    )

    raw_manifest_audit = audit_manifest(manifest_dir)
    manifest_audit = _clean_manifest_audit(raw_manifest_audit)
    manifest_count = validate_calibration_manifest(
        manifest_audit,
        candidate_sha256=hashes["candidate"],
        allocation_sha256=hashes["allocation"],
        manifest_sha256=hashes["manifest"],
    )
    report = audit_sibling_rollout_bank(
        manifest_path,
        rollouts_path,
        expected_manifest_sha256=hashes["manifest"],
        expected_states=manifest_count,
        expected_candidate_count=4,
        expected_model_revision=MODEL_REVISION,
        expected_scientific_status=EXPECTED_SCIENTIFIC_STATUS,
    )
    if report.get("code_revision") != code_revision:
        raise ValueError("DocVQA rollout provenance code revision mismatch")
    report.update(
        {
            "candidate_sha256": hashes["candidate"],
            "candidate_audit_sha256": hashes["candidate_audit"],
            "allocation_sha256": hashes["allocation"],
            "allocation_audit_sha256": hashes["allocation_audit"],
            "manifest_provenance_sha256": hashes["manifest_provenance"],
            "protocol_sha256": hashes["protocol"],
            "manifest_audit": manifest_audit,
            "formal_outcomes_used": False,
        }
    )
    _write_frozen(output_path, report, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
