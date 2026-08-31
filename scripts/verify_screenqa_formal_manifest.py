#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any


_export_module = importlib.import_module(
    "scripts.export_screenqa_formal_manifest"
    if __package__
    else "export_screenqa_formal_manifest"
)
verify_formal_gate = _export_module.verify_formal_gate

EXPECTED_STATES = 14672
EXPECTED_IMAGES = 6000
EXPECTED_SOURCES = 1471
EXPECTED_SCIENTIFIC_STATUS = (
    "only frozen formal-test labels opened after successful risk calibration"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def verify_sha256sums(directory: Path) -> None:
    sums = directory / "SHA256SUMS"
    if not sums.is_file():
        raise FileNotFoundError(f"formal checksum bundle is missing: {sums}")
    for line in sums.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = directory / relative.strip()
        if sha256_file(path) != expected:
            raise ValueError(f"ScreenQA formal checksum mismatch: {path}")


def verify_manifest(
    manifest_dir: Path,
    *,
    candidate_dir: Path,
    expected_candidate_bundle_sha256: str,
    calibration_dir: Path,
    expected_calibration_bundle_sha256: str,
    expected_manifest_sha256: str,
    expected_audit_sha256: str,
) -> dict[str, Any]:
    verify_sha256sums(manifest_dir)
    formal_gate = verify_formal_gate(candidate_dir, calibration_dir)
    candidate_bundle_sha256 = sha256_file(candidate_dir / "SHA256SUMS")
    calibration_bundle_sha256 = sha256_file(calibration_dir / "SHA256SUMS")
    if candidate_bundle_sha256 != expected_candidate_bundle_sha256:
        raise ValueError("ScreenQA formal candidate bundle hash mismatch")
    if calibration_bundle_sha256 != expected_calibration_bundle_sha256:
        raise ValueError("ScreenQA formal calibration bundle hash mismatch")
    if formal_gate.get("bundle_sha256") != calibration_bundle_sha256:
        raise ValueError("ScreenQA formal gate calibration binding mismatch")
    manifest_path = manifest_dir / "manifest.jsonl"
    audit_path = manifest_dir / "manifest.audit.json"
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("ScreenQA formal manifest hash mismatch")
    if sha256_file(audit_path) != expected_audit_sha256:
        raise ValueError("ScreenQA formal manifest audit hash mismatch")
    audit = _load_json(audit_path)
    if audit.get("passed") is not True:
        raise ValueError("ScreenQA formal manifest audit did not pass")
    if audit.get("scientific_status") != EXPECTED_SCIENTIFIC_STATUS:
        raise ValueError("ScreenQA formal manifest scientific status mismatch")
    expected_access: dict[str, object] = {
        "ranker_training_outcomes_previously_used": True,
        "risk_calibration_outcomes_previously_used": True,
        "formal_test_opened": True,
        "reserve_opened": False,
        "untouched_opened": False,
        "official_validation_test_opened": False,
        "annotation_objects_deserialized": EXPECTED_STATES,
        "selected_rico_images": EXPECTED_IMAGES,
        "selected_source_components": EXPECTED_SOURCES,
        "unselected_annotation_objects_deserialized": 0,
    }
    for key, expected_value in expected_access.items():
        if audit.get(key) != expected_value:
            raise ValueError(f"ScreenQA formal access audit {key} mismatch")
    if audit.get("formal_gate") != formal_gate:
        raise ValueError("ScreenQA formal audit gate binding mismatch")
    manifest_audit = audit.get("manifest")
    if not isinstance(manifest_audit, dict):
        raise ValueError("ScreenQA formal manifest audit payload is malformed")
    expected_manifest: dict[str, object] = {
        "manifest_sha256": expected_manifest_sha256,
        "count": EXPECTED_STATES,
        "scorer": "screenqa",
    }
    for key, expected_manifest_value in expected_manifest.items():
        if manifest_audit.get(key) != expected_manifest_value:
            raise ValueError(f"ScreenQA formal manifest {key} mismatch")
    provenance = audit.get("export_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("ScreenQA formal export provenance is malformed")
    metadata = provenance.get("selection_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("ScreenQA formal selection metadata is missing")
    if (
        metadata.get("role") != "formal_test"
        or metadata.get("formal_gate") != formal_gate
        or metadata.get("calibration_verified_before_annotation_deserialization")
        is not True
        or metadata.get("reserve_outcomes_opened") is not False
        or metadata.get("untouched_outcomes_opened") is not False
        or metadata.get("unselected_question_or_target_fields_accessed") is not False
    ):
        raise ValueError("ScreenQA formal provenance boundary mismatch")
    return {
        "passed": True,
        "manifest_sha256": expected_manifest_sha256,
        "audit_sha256": expected_audit_sha256,
        "candidate_bundle_sha256": candidate_bundle_sha256,
        "calibration_bundle_sha256": calibration_bundle_sha256,
        "selected_threshold": formal_gate["selected_threshold"],
        "states": EXPECTED_STATES,
        "selected_rico_images": EXPECTED_IMAGES,
        "selected_source_components": EXPECTED_SOURCES,
        "sealed_roles": ["reserve", "untouched", "official_validation_test"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the calibrated-policy-bound ScreenQA formal manifest"
    )
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--expected-candidate-bundle-sha256", required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--expected-calibration-bundle-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-audit-sha256", required=True)
    args = parser.parse_args()
    result = verify_manifest(
        args.manifest_dir.resolve(),
        candidate_dir=args.candidate_dir.resolve(),
        expected_candidate_bundle_sha256=args.expected_candidate_bundle_sha256,
        calibration_dir=args.calibration_dir.resolve(),
        expected_calibration_bundle_sha256=(
            args.expected_calibration_bundle_sha256
        ),
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_audit_sha256=args.expected_audit_sha256,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
