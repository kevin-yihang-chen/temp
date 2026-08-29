from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .docvqa_formal import FORMAL_SOURCES
from .docvqa_train_allocation import (
    DATASET_ID,
    DATASET_NAME,
    DATASET_REVISION,
    DATASET_SPLIT,
    NAMESPACE,
    PROTOCOL_SHA256,
    ROLE_SPECS,
    SEED,
)
from .manifest_export import image_digest


FORMAL_ROLE = "formal_test"
FORMAL_STATE_NAMESPACE = "docvqa-train-factorized-v2-formal"


def _require(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"DocVQA formal export mismatch for {name}")


def _role_identity_map(
    allocation_document: Mapping[str, Any],
    role: str,
    expected_count: int,
) -> dict[str, str]:
    body = allocation_document.get("allocation")
    if not isinstance(body, Mapping):
        raise ValueError("DocVQA formal allocation body is missing")
    roles = body.get("roles")
    if not isinstance(roles, Mapping):
        raise ValueError("DocVQA formal allocation roles are missing")
    payload = roles.get(role)
    if not isinstance(payload, Mapping):
        raise ValueError(f"DocVQA formal allocation lacks role {role!r}")
    assignments = payload.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != expected_count:
        raise ValueError(f"DocVQA formal role {role!r} count changed")
    identities: dict[str, str] = {}
    for item in assignments:
        if not isinstance(item, Mapping):
            raise ValueError("DocVQA formal allocation assignment is invalid")
        source = str(item.get("source_group_id", "")).strip()
        image = str(item.get("image_id", "")).strip()
        if not source or not image or source in identities:
            raise ValueError("DocVQA formal allocation has invalid source identities")
        identities[source] = image
    if len(identities) != expected_count or len(set(identities.values())) != expected_count:
        raise ValueError(f"DocVQA formal role {role!r} has duplicate identities")
    return identities


def formal_role_identity_map(
    allocation_document: Mapping[str, Any],
) -> dict[str, str]:
    """Read the sealed formal identities without enabling development export."""

    return _role_identity_map(allocation_document, FORMAL_ROLE, FORMAL_SOURCES)


def validate_sealed_formal_allocation(
    allocation_document: Mapping[str, Any],
    allocation_audit: Mapping[str, Any],
    *,
    allocation_sha256: str,
) -> dict[str, str]:
    """Validate that the original identity allocation still seals formal targets."""

    _require(allocation_document.get("protocol_sha256"), PROTOCOL_SHA256, "protocol")
    dataset = allocation_document.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("DocVQA formal allocation lacks dataset provenance")
    expected_dataset = {
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "split": DATASET_SPLIT,
    }
    for name, expected in expected_dataset.items():
        _require(dataset.get(name), expected, f"dataset {name}")
    body = allocation_document.get("allocation")
    if not isinstance(body, Mapping):
        raise ValueError("DocVQA formal allocation body is missing")
    _require(body.get("namespace"), NAMESPACE, "namespace")
    _require(body.get("seed"), SEED, "seed")

    contract = allocation_document.get("selection_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("DocVQA formal allocation lacks selection contract")
    expected_contract = {
        "selection_target_fields_accessed": False,
        "selection_allowed_fields": ["docId", "image"],
        "formal_manifest_exported": False,
        "formal_outcomes_collected": False,
    }
    for name, expected in expected_contract.items():
        _require(contract.get(name), expected, f"selection contract {name}")

    expected_audit = {
        "passed": True,
        "allocation_sha256": allocation_sha256,
        "protocol_sha256": PROTOCOL_SHA256,
        "formal_outcomes_collected": False,
    }
    for name, expected in expected_audit.items():
        _require(allocation_audit.get(name), expected, f"allocation audit {name}")
    overlap = allocation_audit.get("overlap")
    if not isinstance(overlap, Mapping) or any(
        not isinstance(value, int) or value != 0 for value in overlap.values()
    ):
        raise ValueError("DocVQA formal allocation audit reports identity overlap")

    role_identities: dict[str, dict[str, str]] = {}
    for spec in ROLE_SPECS:
        role_identities[spec.name] = _role_identity_map(
            allocation_document,
            spec.name,
            spec.count,
        )
    formal = role_identities[FORMAL_ROLE]
    formal_sources = set(formal)
    formal_images = set(formal.values())
    for role in ("ranker_training", "risk_calibration"):
        other = role_identities[role]
        if formal_sources.intersection(other) or formal_images.intersection(other.values()):
            raise ValueError(f"DocVQA formal allocation overlaps role {role!r}")
    return formal


def validate_formal_rows(
    rows: Sequence[Mapping[str, Any]],
    allocation_document: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute every formal row docId/RGB identity before target export."""

    expected = formal_role_identity_map(allocation_document)
    observed: dict[str, str] = {}
    for index in range(len(rows)):
        row = rows[index]
        source = str(row.get("docId", "")).strip()
        if source not in expected:
            raise ValueError(f"DocVQA formal row {index} is outside the frozen role")
        raw_image = row.get("image")
        convert = getattr(raw_image, "convert", None)
        if not callable(convert):
            raise ValueError(f"DocVQA formal row {index} image is not decodable")
        digest = image_digest(convert("RGB"))
        if digest != expected[source]:
            raise ValueError(f"DocVQA formal row {index} RGB differs from allocation")
        previous = observed.setdefault(source, digest)
        if previous != digest:
            raise ValueError(f"DocVQA formal source {source!r} maps to multiple images")
    if set(observed) != set(expected):
        raise ValueError("DocVQA formal rows do not cover the frozen role exactly")
    return {
        "role": FORMAL_ROLE,
        "row_count": len(rows),
        "source_group_count": len(observed),
        "unique_image_count": len(set(observed.values())),
        "source_identity_recomputed": True,
        "selection_target_fields_accessed": False,
    }


def validate_formal_manifest_audit(
    manifest_audit: Mapping[str, Any],
    allocation_document: Mapping[str, Any],
    *,
    allocation_sha256: str,
    allocation_audit_sha256: str,
    policy_freeze_sha256: str,
) -> dict[str, Any]:
    """Bind a materialized formal manifest to allocation and policy identities."""

    expected = formal_role_identity_map(allocation_document)
    expected_sources = {f"docvqa:{source}" for source in expected}
    expected_images = set(expected.values())
    common = {
        "task": "docvqa",
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "split": DATASET_SPLIT,
        "scorer": "docvqa",
        "unique_sources": FORMAL_SOURCES,
        "unique_images": FORMAL_SOURCES,
    }
    for name, value in common.items():
        _require(manifest_audit.get(name), value, f"manifest {name}")
    count = manifest_audit.get("count")
    if not isinstance(count, int) or count < FORMAL_SOURCES:
        raise ValueError("DocVQA formal manifest has too few questions")
    _require(manifest_audit.get("unique_states"), count, "manifest states")
    _require(manifest_audit.get("_sources"), expected_sources, "manifest sources")
    _require(manifest_audit.get("_images"), expected_images, "manifest images")
    selection = manifest_audit.get("selection_metadata")
    if not isinstance(selection, Mapping):
        raise ValueError("DocVQA formal manifest lacks selection metadata")
    expected_selection = {
        "allocation_sha256": allocation_sha256,
        "allocation_audit_sha256": allocation_audit_sha256,
        "policy_freeze_sha256": policy_freeze_sha256,
        "protocol_sha256": PROTOCOL_SHA256,
        "namespace": NAMESPACE,
        "role": FORMAL_ROLE,
        "selected_source_group_count": FORMAL_SOURCES,
        "selection_uses_targets": False,
    }
    for name, value in expected_selection.items():
        _require(selection.get(name), value, f"selection {name}")
    return {key: value for key, value in manifest_audit.items() if not key.startswith("_")}
