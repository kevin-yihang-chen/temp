#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Callable, cast


_candidate_module = importlib.import_module(
    "scripts.export_screenqa_calibration_manifest"
    if __package__
    else "export_screenqa_calibration_manifest"
)
verify_candidate = cast(
    Callable[[Path], dict[str, Any]], _candidate_module.verify_candidate
)


EXPECTED_STATES = 9951
EXPECTED_IMAGES = 4001
EXPECTED_SOURCES = 1016
EXPECTED_SCIENTIFIC_STATUS = (
    "only frozen risk-calibration labels opened after sole candidate freeze"
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
    sums_path = directory / "SHA256SUMS"
    if not sums_path.is_file():
        raise FileNotFoundError(f"checksum bundle is missing: {sums_path}")
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = directory / relative.strip()
        if sha256_file(path) != expected:
            raise ValueError(f"SHA256SUMS mismatch for {path}")


def verify_manifest(
    manifest_dir: Path,
    *,
    candidate_dir: Path,
    expected_candidate_bundle_sha256: str,
    expected_manifest_sha256: str,
    expected_audit_sha256: str,
) -> dict[str, Any]:
    verify_sha256sums(manifest_dir)
    candidate = verify_candidate(candidate_dir)
    if candidate["bundle_sha256"] != expected_candidate_bundle_sha256:
        raise ValueError("ScreenQA calibration candidate bundle hash mismatch")
    manifest_path = manifest_dir / "manifest.jsonl"
    audit_path = manifest_dir / "manifest.audit.json"
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("ScreenQA calibration manifest hash mismatch")
    if sha256_file(audit_path) != expected_audit_sha256:
        raise ValueError("ScreenQA calibration manifest audit hash mismatch")
    audit = _load_json(audit_path)
    if audit.get("passed") is not True:
        raise ValueError("ScreenQA calibration manifest audit did not pass")
    if audit.get("scientific_status") != EXPECTED_SCIENTIFIC_STATUS:
        raise ValueError("ScreenQA calibration scientific status mismatch")
    expected_access = {
        "ranker_training_outcomes_previously_used": True,
        "risk_calibration_opened": True,
        "formal_test_opened": False,
        "reserve_opened": False,
        "untouched_opened": False,
        "official_validation_test_opened": False,
        "annotation_objects_deserialized": EXPECTED_STATES,
        "selected_rico_images": EXPECTED_IMAGES,
        "selected_source_components": EXPECTED_SOURCES,
        "unselected_annotation_objects_deserialized": 0,
    }
    for key, expected_access_value in expected_access.items():
        if audit.get(key) != expected_access_value:
            raise ValueError(f"ScreenQA calibration access audit {key} mismatch")
    if audit.get("candidate") != candidate:
        raise ValueError("ScreenQA calibration audit candidate binding mismatch")
    manifest_audit = audit.get("manifest")
    if not isinstance(manifest_audit, dict):
        raise ValueError("ScreenQA calibration manifest audit payload is malformed")
    expected_manifest = {
        "manifest_sha256": expected_manifest_sha256,
        "count": EXPECTED_STATES,
        "scorer": "screenqa",
    }
    for key, expected_manifest_value in expected_manifest.items():
        if manifest_audit.get(key) != expected_manifest_value:
            raise ValueError(f"ScreenQA calibration manifest {key} mismatch")
    provenance = audit.get("export_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("ScreenQA calibration export provenance is malformed")
    metadata = provenance.get("selection_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("ScreenQA calibration selection metadata is missing")
    if (
        metadata.get("role") != "risk_calibration"
        or metadata.get("candidate") != candidate
        or metadata.get("candidate_frozen_before_annotation_deserialization") is not True
        or metadata.get("formal_outcomes_opened") is not False
        or metadata.get("reserve_outcomes_opened") is not False
        or metadata.get("untouched_outcomes_opened") is not False
    ):
        raise ValueError("ScreenQA calibration provenance boundary mismatch")
    return {
        "passed": True,
        "manifest_sha256": expected_manifest_sha256,
        "audit_sha256": expected_audit_sha256,
        "candidate_bundle_sha256": expected_candidate_bundle_sha256,
        "states": EXPECTED_STATES,
        "selected_rico_images": EXPECTED_IMAGES,
        "selected_source_components": EXPECTED_SOURCES,
        "sealed_roles": ["formal_test", "reserve", "untouched", "official_validation_test"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the candidate-bound ScreenQA risk-calibration manifest"
    )
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--expected-candidate-bundle-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-audit-sha256", required=True)
    args = parser.parse_args()
    result = verify_manifest(
        args.manifest_dir,
        candidate_dir=args.candidate_dir,
        expected_candidate_bundle_sha256=args.expected_candidate_bundle_sha256,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_audit_sha256=args.expected_audit_sha256,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
