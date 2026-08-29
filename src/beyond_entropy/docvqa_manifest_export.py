from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .docvqa_train_allocation import (
    DATASET_ID,
    DATASET_NAME,
    DATASET_REVISION,
    DATASET_SPLIT,
    NAMESPACE,
    PROTOCOL_SHA256,
    ROLE_SPECS,
)
from .manifest_export import image_digest


EXPORTABLE_ROLES = frozenset({"ranker_training", "risk_calibration"})
ROLE_STATE_NAMESPACES = {
    "ranker_training": "docvqa-train-factorized-v2-ranker",
    "risk_calibration": "docvqa-train-factorized-v2-calibration",
}


def _require(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"DocVQA manifest export mismatch for {name}")


def _role_count(role: str) -> int:
    for spec in ROLE_SPECS:
        if spec.name == role:
            return spec.count
    raise ValueError(f"DocVQA role {role!r} is not exportable")


def role_identity_map(
    allocation_document: Mapping[str, Any],
    role: str,
    *,
    expected_count: int | None = None,
) -> dict[str, str]:
    """Return the frozen docId-to-RGB map for one development role."""

    if role not in EXPORTABLE_ROLES:
        raise ValueError(f"DocVQA role {role!r} is not exportable")
    body = allocation_document.get("allocation")
    if not isinstance(body, Mapping):
        raise ValueError("DocVQA allocation body is missing")
    roles = body.get("roles")
    if not isinstance(roles, Mapping):
        raise ValueError("DocVQA allocation roles are missing")
    role_payload = roles.get(role)
    if not isinstance(role_payload, Mapping):
        raise ValueError(f"DocVQA allocation role {role!r} is missing")
    assignments = role_payload.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError(f"DocVQA allocation role {role!r} assignments are invalid")
    identities: dict[str, str] = {}
    for item in assignments:
        if not isinstance(item, Mapping):
            raise ValueError("DocVQA allocation assignment must be a mapping")
        source = str(item.get("source_group_id", "")).strip()
        image = str(item.get("image_id", "")).strip()
        if not source or not image or source in identities:
            raise ValueError("DocVQA allocation role has invalid source identities")
        identities[source] = image
    required_count = _role_count(role) if expected_count is None else expected_count
    if required_count <= 0 or len(identities) != required_count:
        raise ValueError(f"DocVQA allocation role {role!r} count changed")
    if len(set(identities.values())) != required_count:
        raise ValueError(f"DocVQA allocation role {role!r} has RGB overlap")
    return identities


def validate_docvqa_role_rows(
    rows: Sequence[Mapping[str, Any]],
    allocation_document: Mapping[str, Any],
    role: str,
    *,
    expected_count: int | None = None,
) -> dict[str, Any]:
    """Recompute every selected row's docId/RGB identity before export."""

    expected = role_identity_map(
        allocation_document,
        role,
        expected_count=expected_count,
    )
    observed: dict[str, str] = {}
    for index in range(len(rows)):
        row = rows[index]
        source = str(row.get("docId", "")).strip()
        if source not in expected:
            raise ValueError(f"DocVQA export row {index} is outside role {role!r}")
        raw_image = row.get("image")
        convert = getattr(raw_image, "convert", None)
        if not callable(convert):
            raise ValueError(f"DocVQA export row {index} image is not decodable")
        digest = image_digest(convert("RGB"))
        if digest != expected[source]:
            raise ValueError(f"DocVQA export row {index} RGB differs from allocation")
        previous = observed.setdefault(source, digest)
        if previous != digest:
            raise ValueError(f"DocVQA source {source!r} maps to multiple RGB images")
    if set(observed) != set(expected):
        raise ValueError(f"DocVQA export rows do not cover role {role!r} exactly")
    return {
        "role": role,
        "row_count": len(rows),
        "source_group_count": len(observed),
        "unique_image_count": len(set(observed.values())),
        "source_identity_recomputed": True,
        "selection_target_fields_accessed": False,
    }


def validate_exported_docvqa_manifest(
    manifest_audit: Mapping[str, Any],
    allocation_document: Mapping[str, Any],
    role: str,
    *,
    allocation_sha256: str,
    allocation_audit_sha256: str,
    candidate_sha256: str | None = None,
    expected_count: int | None = None,
) -> dict[str, Any]:
    """Bind an exported manifest to its exact allocation role identities."""

    expected = role_identity_map(
        allocation_document,
        role,
        expected_count=expected_count,
    )
    expected_sources = {f"docvqa:{source}" for source in expected}
    expected_images = set(expected.values())
    common = {
        "task": "docvqa",
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "split": DATASET_SPLIT,
        "scorer": "docvqa",
        "unique_sources": len(expected_sources),
        "unique_images": len(expected_images),
    }
    for name, value in common.items():
        _require(manifest_audit.get(name), value, f"manifest {name}")
    count = manifest_audit.get("count")
    if not isinstance(count, int) or count < len(expected_sources):
        raise ValueError("DocVQA manifest has fewer questions than source groups")
    _require(manifest_audit.get("unique_states"), count, "manifest unique states")
    _require(manifest_audit.get("_sources"), expected_sources, "manifest sources")
    _require(manifest_audit.get("_images"), expected_images, "manifest images")
    selection = manifest_audit.get("selection_metadata")
    if not isinstance(selection, Mapping):
        raise ValueError("DocVQA manifest lacks selection metadata")
    expected_selection: dict[str, Any] = {
        "allocation_sha256": allocation_sha256,
        "allocation_audit_sha256": allocation_audit_sha256,
        "protocol_sha256": PROTOCOL_SHA256,
        "namespace": NAMESPACE,
        "role": role,
        "selected_source_group_count": len(expected_sources),
        "selection_uses_targets": False,
    }
    if role == "risk_calibration":
        if not candidate_sha256:
            raise ValueError("DocVQA calibration manifest requires a candidate hash")
        expected_selection["candidate_sha256"] = candidate_sha256
    elif candidate_sha256 is not None:
        raise ValueError("DocVQA ranker manifest cannot bind a later candidate")
    for name, value in expected_selection.items():
        _require(selection.get(name), value, f"selection {name}")

    clean_manifest = {
        key: value for key, value in manifest_audit.items() if not key.startswith("_")
    }
    return {
        "passed": True,
        "scientific_status": (
            f"outcome-order-safe DocVQA {role} manifest identity audit"
        ),
        "role": role,
        "allocation_sha256": allocation_sha256,
        "allocation_audit_sha256": allocation_audit_sha256,
        "candidate_sha256": candidate_sha256,
        "protocol_sha256": PROTOCOL_SHA256,
        "manifest": clean_manifest,
        "formal_manifest_exported": False,
        "formal_outcomes_collected": False,
    }
