from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

from beyond_entropy.docvqa_calibration import validate_calibration_preoutcome_gate
from beyond_entropy.docvqa_candidate_freeze import validate_candidate_freeze_gate
from beyond_entropy.docvqa_manifest_export import (
    EXPORTABLE_ROLES,
    validate_exported_docvqa_manifest,
)
from beyond_entropy.docvqa_train_allocation import PROTOCOL_SHA256, sha256_file
from beyond_entropy.manifest_audit import audit_manifest


_HEX_DIGEST = re.compile(r"[0-9a-f]{64}")


def _load_mapping(path: Path, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _require(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"DocVQA manifest verification mismatch for {name}")


def _require_hash(path: Path, expected: str, name: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"DocVQA manifest verification input is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"DocVQA manifest verification {name} SHA-256 mismatch")
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
        raise ValueError("tracked worktree must be clean before manifest verification")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_unmaterialized(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("DocVQA formal output must remain unmaterialized")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a DocVQA development manifest before GPU execution"
    )
    parser.add_argument("--role", choices=sorted(EXPORTABLE_ROLES), required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-manifest-provenance-sha256", required=True)
    parser.add_argument("--manifest-audit", type=Path, required=True)
    parser.add_argument("--expected-manifest-audit-sha256", required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--allocation-audit", type=Path, required=True)
    parser.add_argument("--expected-allocation-sha256", required=True)
    parser.add_argument("--expected-allocation-audit-sha256", required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--candidate-audit", type=Path)
    parser.add_argument("--expected-candidate-sha256")
    parser.add_argument("--expected-candidate-audit-sha256")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-code-revision", required=True)
    parser.add_argument("--formal-output-dir", type=Path, required=True)
    args = parser.parse_args()

    candidate_values = (
        args.candidate,
        args.candidate_audit,
        args.expected_candidate_sha256,
        args.expected_candidate_audit_sha256,
    )
    if args.role == "risk_calibration" and any(
        value is None for value in candidate_values
    ):
        parser.error("risk_calibration requires candidate paths and hashes")
    if args.role == "ranker_training" and any(
        value is not None for value in candidate_values
    ):
        parser.error("ranker_training cannot bind a later candidate")

    manifest_dir = args.manifest_dir.resolve()
    manifest_path = manifest_dir / "manifest.jsonl"
    provenance_path = manifest_dir / "manifest.provenance.json"
    manifest_audit_path = args.manifest_audit.resolve()
    allocation_path = args.allocation.resolve()
    allocation_audit_path = args.allocation_audit.resolve()
    protocol_path = args.protocol.resolve()
    hashes = {
        "manifest": _require_hash(
            manifest_path,
            args.expected_manifest_sha256,
            "manifest",
        ),
        "manifest_provenance": _require_hash(
            provenance_path,
            args.expected_manifest_provenance_sha256,
            "manifest provenance",
        ),
        "manifest_audit": _require_hash(
            manifest_audit_path,
            args.expected_manifest_audit_sha256,
            "manifest audit",
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
    _require_unmaterialized(args.formal_output_dir.resolve())

    allocation = _load_mapping(allocation_path, "allocation")
    allocation_audit = _load_mapping(allocation_audit_path, "allocation audit")
    candidate_sha256: str | None = None
    candidate_audit_sha256: str | None = None
    if args.role == "risk_calibration":
        assert args.candidate is not None
        assert args.candidate_audit is not None
        assert args.expected_candidate_sha256 is not None
        assert args.expected_candidate_audit_sha256 is not None
        candidate_path = args.candidate.resolve()
        candidate_audit_path = args.candidate_audit.resolve()
        candidate_sha256 = _require_hash(
            candidate_path,
            args.expected_candidate_sha256,
            "candidate",
        )
        candidate_audit_sha256 = _require_hash(
            candidate_audit_path,
            args.expected_candidate_audit_sha256,
            "candidate audit",
        )
        validate_calibration_preoutcome_gate(
            _load_mapping(candidate_path, "candidate"),
            _load_mapping(candidate_audit_path, "candidate audit"),
            allocation,
            allocation_audit,
            candidate_sha256=candidate_sha256,
            allocation_sha256=hashes["allocation"],
            code_revision=code_revision,
        )
    else:
        validate_candidate_freeze_gate(
            allocation,
            allocation_audit,
            allocation_sha256=hashes["allocation"],
        )

    expected_audit = validate_exported_docvqa_manifest(
        audit_manifest(manifest_dir),
        allocation,
        args.role,
        allocation_sha256=hashes["allocation"],
        allocation_audit_sha256=hashes["allocation_audit"],
        candidate_sha256=candidate_sha256,
    )
    stored_audit = _load_mapping(manifest_audit_path, "manifest audit")
    required = {
        "passed": True,
        "scientific_status": expected_audit["scientific_status"],
        "role": args.role,
        "allocation_sha256": hashes["allocation"],
        "allocation_audit_sha256": hashes["allocation_audit"],
        "candidate_sha256": candidate_sha256,
        "candidate_audit_sha256": candidate_audit_sha256,
        "manifest_provenance_sha256": hashes["manifest_provenance"],
        "protocol_sha256": hashes["protocol"],
        "manifest": expected_audit["manifest"],
        "code_revision": code_revision,
        "formal_manifest_exported": False,
        "formal_targets_materialized": False,
        "formal_outcomes_collected": False,
    }
    for name, expected in required.items():
        _require(stored_audit.get(name), expected, f"stored audit {name}")
    if args.role == "risk_calibration":
        ranker_hash = str(stored_audit.get("ranker_manifest_sha256", ""))
        if _HEX_DIGEST.fullmatch(ranker_hash) is None:
            raise ValueError("DocVQA calibration audit lacks ranker manifest binding")
    print(
        json.dumps(
            {
                "passed": True,
                "role": args.role,
                "manifest_sha256": hashes["manifest"],
                "manifest_audit_sha256": hashes["manifest_audit"],
                "code_revision": code_revision,
                "formal_manifest_exported": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
