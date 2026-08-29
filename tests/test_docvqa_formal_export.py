from __future__ import annotations

from PIL import Image
import pytest

from beyond_entropy.docvqa_formal_export import (
    validate_formal_manifest_audit,
    validate_formal_rows,
    validate_sealed_formal_allocation,
)
from beyond_entropy.docvqa_train_allocation import (
    DATASET_ID,
    DATASET_NAME,
    DATASET_REVISION,
    DATASET_SPLIT,
    NAMESPACE,
    PROTOCOL_SHA256,
    ROLE_SPECS,
    SEED,
)
from beyond_entropy.manifest_export import image_digest


def _allocation():
    roles = {}
    source = 0
    for spec in ROLE_SPECS:
        assignments = []
        for _ in range(spec.count):
            assignments.append(
                {
                    "source_group_id": f"d{source}",
                    "image_id": f"{source:064x}",
                }
            )
            source += 1
        roles[spec.name] = {"assignments": assignments}
    return {
        "protocol_sha256": PROTOCOL_SHA256,
        "dataset": {
            "dataset_id": DATASET_ID,
            "dataset_name": DATASET_NAME,
            "dataset_revision": DATASET_REVISION,
            "split": DATASET_SPLIT,
        },
        "selection_contract": {
            "selection_target_fields_accessed": False,
            "selection_allowed_fields": ["docId", "image"],
            "formal_manifest_exported": False,
            "formal_outcomes_collected": False,
        },
        "allocation": {"namespace": NAMESPACE, "seed": SEED, "roles": roles},
    }


def test_sealed_docvqa_formal_role_rejects_cross_role_identity_overlap():
    allocation = _allocation()
    audit = {
        "passed": True,
        "allocation_sha256": "a" * 64,
        "protocol_sha256": PROTOCOL_SHA256,
        "formal_outcomes_collected": False,
        "overlap": {"ranker_training_formal_test_sources": 0},
    }
    identities = validate_sealed_formal_allocation(
        allocation,
        audit,
        allocation_sha256="a" * 64,
    )
    assert len(identities) == 3500
    allocation["allocation"]["roles"]["formal_test"]["assignments"][0][
        "image_id"
    ] = allocation["allocation"]["roles"]["ranker_training"]["assignments"][0][
        "image_id"
    ]
    with pytest.raises(ValueError, match="overlaps role"):
        validate_sealed_formal_allocation(
            allocation,
            audit,
            allocation_sha256="a" * 64,
        )


def test_docvqa_formal_rows_recompute_rgb_and_require_complete_role():
    allocation = _allocation()
    formal = allocation["allocation"]["roles"]["formal_test"]["assignments"]
    rows = []
    for index, assignment in enumerate(formal):
        image = Image.new(
            "RGB",
            (1, 1),
            (index % 256, (index // 256) % 256, (index // (256 * 256)) % 256),
        )
        assignment["image_id"] = image_digest(image)
        rows.append({"docId": assignment["source_group_id"], "image": image})
    report = validate_formal_rows(rows, allocation)
    assert report["source_group_count"] == 3500
    rows[0]["image"] = Image.new("RGB", (1, 1), "white")
    with pytest.raises(ValueError, match="RGB differs"):
        validate_formal_rows(rows, allocation)


def test_docvqa_formal_manifest_binds_policy_hash_and_exact_identities():
    allocation = _allocation()
    formal = allocation["allocation"]["roles"]["formal_test"]["assignments"]
    sources = {f"docvqa:{item['source_group_id']}" for item in formal}
    images = {item["image_id"] for item in formal}
    audit = {
        "task": "docvqa",
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "split": DATASET_SPLIT,
        "scorer": "docvqa",
        "count": 4000,
        "unique_states": 4000,
        "unique_sources": 3500,
        "unique_images": 3500,
        "_sources": sources,
        "_images": images,
        "selection_metadata": {
            "allocation_sha256": "a" * 64,
            "allocation_audit_sha256": "b" * 64,
            "policy_freeze_sha256": "c" * 64,
            "protocol_sha256": PROTOCOL_SHA256,
            "namespace": NAMESPACE,
            "role": "formal_test",
            "selected_source_group_count": 3500,
            "selection_uses_targets": False,
        },
    }
    clean = validate_formal_manifest_audit(
        audit,
        allocation,
        allocation_sha256="a" * 64,
        allocation_audit_sha256="b" * 64,
        policy_freeze_sha256="c" * 64,
    )
    assert "_sources" not in clean
    audit["selection_metadata"]["policy_freeze_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="policy_freeze_sha256"):
        validate_formal_manifest_audit(
            audit,
            allocation,
            allocation_sha256="a" * 64,
            allocation_audit_sha256="b" * 64,
            policy_freeze_sha256="c" * 64,
        )
