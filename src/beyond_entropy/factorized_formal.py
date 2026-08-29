from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


FORMAL_SOURCES = 5953
CALIBRATION_SOURCES = 3000
CALIBRATION_DECISIONS = 4747
MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
ALLOCATION_SHA256 = (
    "bc0ecb4b6f49a5b0e92b90b4c30620f72246722370d59c8078753d5846f5e9b6"
)
ALLOCATION_AUDIT_SHA256 = (
    "f01f853a7de7774466be55c012b7e174f57f4ac120ed58a0bf3984e71252b5c3"
)
CANDIDATE_SHA256 = (
    "9a6c9d032ebdbc271b7d3c829fbb3d6ff167cac01b54ce75adc8da86e3063342"
)
PROTOCOL_SHA256 = (
    "babf01d4090263d1cfcb28c42f86f7b13ae9de4bb6bab0ca10d6e4707f02e2ca"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mapping(path: Path, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def check_hash(path: Path, expected: str, name: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{name} SHA-256 mismatch")
    return actual


def validate_policy_freeze(
    freeze: Mapping[str, Any],
    *,
    verify_components: bool = True,
) -> None:
    if freeze.get("schema_version") != 1:
        raise ValueError("factorized formal freeze schema mismatch")
    if freeze.get("formal_gate_status") != "ready_for_formal_manifest":
        raise ValueError("factorized calibration has not opened the formal gate")
    if freeze.get("formal_outcomes_used") is not False:
        raise ValueError("factorized policy freeze used formal outcomes")
    calibration = freeze.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("factorized policy freeze is missing calibration metadata")
    if (
        calibration.get("selection_status")
        != "selected_non_degenerate_safe_threshold"
        or calibration.get("n_sources") != CALIBRATION_SOURCES
        or calibration.get("n_decisions") != CALIBRATION_DECISIONS
        or calibration.get("formal_outcomes_used") is not False
    ):
        raise ValueError("factorized policy freeze has an invalid calibration gate")
    formal = freeze.get("formal_test")
    if not isinstance(formal, Mapping):
        raise ValueError("factorized policy freeze is missing formal metadata")
    if (
        formal.get("allocated_sources") != FORMAL_SOURCES
        or formal.get("manifest_materialized") is not False
        or formal.get("rollouts_collected") is not False
        or formal.get("outcomes_used") is not False
    ):
        raise ValueError("factorized policy freeze does not describe a sealed formal role")
    for section_name in ("artifacts", "implementation"):
        section = freeze.get(section_name)
        if not isinstance(section, Mapping) or not section:
            raise ValueError(f"factorized policy freeze is missing {section_name}")
        for name, item in section.items():
            if not isinstance(item, Mapping):
                raise ValueError(f"invalid frozen component {section_name}.{name}")
            path = Path(str(item.get("path", ""))).resolve()
            expected = str(item.get("sha256", ""))
            if not expected:
                raise ValueError(f"missing frozen hash {section_name}.{name}")
            if verify_components:
                check_hash(path, expected, f"{section_name}.{name}")


def validate_materialized_formal_gate(
    *,
    policy_freeze_path: Path,
    expected_policy_freeze_sha256: str,
    model_path: Path,
    expected_model_sha256: str,
    manifest_path: Path,
    expected_manifest_sha256: str,
    manifest_provenance_path: Path,
    expected_manifest_provenance_sha256: str,
    audit_path: Path,
    expected_audit_sha256: str,
) -> dict[str, Any]:
    check_hash(
        policy_freeze_path,
        expected_policy_freeze_sha256,
        "policy freeze",
    )
    freeze = load_mapping(policy_freeze_path, "policy freeze")
    validate_policy_freeze(freeze)
    check_hash(model_path, expected_model_sha256, "calibrated model")
    frozen_model = freeze["artifacts"].get("calibrated_model")
    if not isinstance(frozen_model, Mapping) or (
        str(Path(str(frozen_model.get("path", ""))).resolve())
        != str(model_path.resolve())
        or frozen_model.get("sha256") != expected_model_sha256
    ):
        raise ValueError("formal model is not the policy-freeze model")
    check_hash(manifest_path, expected_manifest_sha256, "formal manifest")
    check_hash(
        manifest_provenance_path,
        expected_manifest_provenance_sha256,
        "formal manifest provenance",
    )
    check_hash(audit_path, expected_audit_sha256, "formal audit")
    provenance = load_mapping(manifest_provenance_path, "manifest provenance")
    selection = provenance.get("selection_metadata")
    if not isinstance(selection, Mapping) or (
        selection.get("policy_freeze_sha256") != expected_policy_freeze_sha256
        or selection.get("allocation_sha256") != ALLOCATION_SHA256
        or selection.get("role") != "formal_test"
        or selection.get("selected_source_group_count") != FORMAL_SOURCES
        or selection.get("selection_uses_targets") is not False
    ):
        raise ValueError("formal manifest is not bound to the frozen factorized policy")
    audit = load_mapping(audit_path, "formal audit")
    formal_audit = audit.get("formal")
    if not isinstance(formal_audit, Mapping) or (
        audit.get("passed") is not True
        or audit.get("policy_freeze_sha256") != expected_policy_freeze_sha256
        or audit.get("allocation_sha256") != ALLOCATION_SHA256
        or formal_audit.get("manifest_sha256") != expected_manifest_sha256
        or formal_audit.get("unique_sources") != FORMAL_SOURCES
    ):
        raise ValueError("formal audit is not bound to the frozen policy and manifest")
    overlap = audit.get("overlap")
    if not isinstance(overlap, Mapping) or any(
        int(value) != 0 for value in overlap.values()
    ):
        raise ValueError("formal audit reports identity overlap")
    return freeze
