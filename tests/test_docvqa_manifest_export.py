from __future__ import annotations

from PIL import Image
import pytest

from beyond_entropy.docvqa_manifest_export import (
    validate_docvqa_role_rows,
    validate_exported_docvqa_manifest,
)
from beyond_entropy.docvqa_train_allocation import (
    DATASET_ID,
    DATASET_NAME,
    DATASET_REVISION,
    NAMESPACE,
    PROTOCOL_SHA256,
)
from beyond_entropy.manifest_export import image_digest


def _fixture():
    first = Image.new("RGB", (2, 2), color=(1, 2, 3))
    second = Image.new("RGB", (2, 2), color=(4, 5, 6))
    identities = {"doc-a": image_digest(first), "doc-b": image_digest(second)}
    allocation = {
        "allocation": {
            "roles": {
                "ranker_training": {
                    "assignments": [
                        {"source_group_id": source, "image_id": image}
                        for source, image in identities.items()
                    ]
                },
                "risk_calibration": {"assignments": []},
            }
        }
    }
    rows = [
        {"docId": "doc-a", "image": first},
        {"docId": "doc-a", "image": first.copy()},
        {"docId": "doc-b", "image": second},
    ]
    selection = {
        "allocation_sha256": "a" * 64,
        "allocation_audit_sha256": "b" * 64,
        "protocol_sha256": PROTOCOL_SHA256,
        "namespace": NAMESPACE,
        "role": "ranker_training",
        "selected_source_group_count": 2,
        "selection_uses_targets": False,
    }
    manifest_audit = {
        "task": "docvqa",
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "split": "train",
        "scorer": "docvqa",
        "count": 3,
        "unique_states": 3,
        "unique_sources": 2,
        "unique_images": 2,
        "selection_metadata": selection,
        "_sources": {"docvqa:doc-a", "docvqa:doc-b"},
        "_images": set(identities.values()),
    }
    return rows, allocation, manifest_audit


def test_docvqa_role_export_recomputes_every_row_and_manifest_identity():
    rows, allocation, manifest_audit = _fixture()
    row_audit = validate_docvqa_role_rows(
        rows,
        allocation,
        "ranker_training",
        expected_count=2,
    )
    assert row_audit["row_count"] == 3
    assert row_audit["source_group_count"] == 2
    audit = validate_exported_docvqa_manifest(
        manifest_audit,
        allocation,
        "ranker_training",
        allocation_sha256="a" * 64,
        allocation_audit_sha256="b" * 64,
        expected_count=2,
    )
    assert audit["passed"] is True
    assert audit["formal_manifest_exported"] is False


def test_docvqa_role_export_rejects_rgb_and_source_drift():
    rows, allocation, _ = _fixture()
    rows[0]["image"] = Image.new("RGB", (2, 2), color=(9, 9, 9))
    with pytest.raises(ValueError, match="RGB differs"):
        validate_docvqa_role_rows(
            rows,
            allocation,
            "ranker_training",
            expected_count=2,
        )
    rows, allocation, _ = _fixture()
    rows.pop()
    with pytest.raises(ValueError, match="cover role"):
        validate_docvqa_role_rows(
            rows,
            allocation,
            "ranker_training",
            expected_count=2,
        )


def test_docvqa_manifest_never_exports_formal_or_retrofits_ranker_candidate():
    _, allocation, manifest_audit = _fixture()
    with pytest.raises(ValueError, match="not exportable"):
        validate_docvqa_role_rows([], allocation, "formal_test")
    with pytest.raises(ValueError, match="cannot bind"):
        validate_exported_docvqa_manifest(
            manifest_audit,
            allocation,
            "ranker_training",
            allocation_sha256="a" * 64,
            allocation_audit_sha256="b" * 64,
            candidate_sha256="c" * 64,
            expected_count=2,
        )
