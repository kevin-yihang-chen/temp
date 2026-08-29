from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from beyond_entropy.docvqa_train_allocation import (
    PROTOCOL_SHA256,
    SELECTED_SOURCE_COUNT,
    build_allocation_audit,
    build_allocation_document,
    load_prior_identities,
    record_source_image_identity,
)
from beyond_entropy.manifest_export import image_digest


def _source_images(count: int = 9510) -> dict[str, str]:
    return {
        f"document-{index:05d}": f"{index:064x}"
        for index in range(count)
    }


def _document(
    tmp_path: Path,
    source_images: dict[str, str],
    *,
    excluded_images: set[str] | None = None,
    excluded_sources: set[str] | None = None,
) -> dict:
    return build_allocation_document(
        source_images,
        excluded_image_ids=excluded_images or set(),
        excluded_source_group_ids=excluded_sources or set(),
        prior_banks=[{"manifest": "prior/manifest.jsonl", "manifest_sha256": "a" * 64}],
        parquet_files=[tmp_path / "train.parquet"],
        parquet_sha256=["b" * 64],
        row_count=39463,
        protocol_path=tmp_path / "protocol.md",
        code_revision="c" * 40,
    )


def _audit(
    tmp_path: Path,
    document: dict,
    *,
    excluded_images: set[str] | None = None,
    excluded_sources: set[str] | None = None,
) -> dict:
    return build_allocation_audit(
        document,
        allocation_path=tmp_path / "allocation.json",
        allocation_sha256="d" * 64,
        excluded_image_ids=excluded_images or set(),
        excluded_source_group_ids=excluded_sources or set(),
    )


def test_docvqa_allocation_is_deterministic_disjoint_and_sealed(tmp_path):
    sources = _source_images()
    first = _document(tmp_path, sources)
    second = _document(tmp_path, sources)
    assert first == second
    assert first["protocol_sha256"] == PROTOCOL_SHA256
    assert first["selection_contract"] == {
        "selection_target_fields_accessed": False,
        "selection_allowed_fields": ["docId", "image"],
        "ranker_manifest_exported": False,
        "calibration_manifest_exported": False,
        "formal_manifest_exported": False,
        "ranker_outcomes_collected": False,
        "calibration_outcomes_collected": False,
        "formal_outcomes_collected": False,
    }
    audit = _audit(tmp_path, first)
    assert audit["passed"] is True
    assert sum(
        role["allocated_source_count"] for role in audit["roles"].values()
    ) == SELECTED_SOURCE_COUNT
    assert not any(audit["overlap"].values())


def test_docvqa_allocation_backfills_prior_and_duplicate_rgb(tmp_path):
    sources = _source_images(9515)
    baseline = _document(tmp_path, sources)
    first_assignments = baseline["allocation"]["roles"]["ranker_training"][
        "assignments"
    ]
    excluded_image = first_assignments[0]["image_id"]
    excluded_source = first_assignments[1]["source_group_id"]
    duplicate_group = first_assignments[2]["source_group_id"]
    sources[duplicate_group] = first_assignments[3]["image_id"]
    excluded_images = {excluded_image}
    excluded_sources = {excluded_source}

    document = _document(
        tmp_path,
        sources,
        excluded_images=excluded_images,
        excluded_sources=excluded_sources,
    )
    audit = _audit(
        tmp_path,
        document,
        excluded_images=excluded_images,
        excluded_sources=excluded_sources,
    )
    allocation = document["allocation"]
    assert allocation["prior_collision_source_group_count"] == 1
    assert allocation["prior_source_group_collision_count"] == 1
    assert allocation["duplicate_rgb_source_group_count"] == 1
    assert sum(
        role["reserve_backfill_count"] for role in audit["roles"].values()
    ) >= 3
    assert not any(audit["overlap"].values())


def test_docvqa_allocation_fails_closed_below_registered_size(tmp_path):
    with pytest.raises(ValueError, match="insufficient eligible reserve"):
        _document(tmp_path, _source_images(SELECTED_SOURCE_COUNT - 1))


def test_docvqa_allocation_rejects_non_digest_image_identity(tmp_path):
    sources = _source_images()
    sources["document-00000"] = "not-a-digest"
    with pytest.raises(ValueError, match="decoded-RGB SHA-256"):
        _document(tmp_path, sources)


def test_docvqa_row_identity_rejects_one_source_with_two_images():
    source_images: dict[str, str] = {}
    assert record_source_image_identity(
        source_images,
        source_group_id="document-1",
        image_id="1" * 64,
    ) is True
    assert record_source_image_identity(
        source_images,
        source_group_id="document-1",
        image_id="1" * 64,
    ) is False
    with pytest.raises(ValueError, match="maps to multiple RGB images"):
        record_source_image_identity(
            source_images,
            source_group_id="document-1",
            image_id="2" * 64,
        )


def test_prior_identity_loader_verifies_rgb_and_docvqa_sources(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "document.png"
    Image.new("RGB", (5, 7), "navy").save(image_path)
    with Image.open(image_path) as raw_image:
        digest = image_digest(raw_image.convert("RGB"))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "image_id": digest,
                "image_path": "images/document.png",
                "source_id": "docvqa:prior-document",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    images, sources, records = load_prior_identities([manifest])
    assert images == {digest}
    assert sources == {"prior-document"}
    assert records[0]["row_count"] == 1
    assert records[0]["docvqa_source_group_count"] == 1


def test_prior_identity_loader_rejects_declared_rgb_mismatch(tmp_path):
    image_path = tmp_path / "document.png"
    Image.new("RGB", (5, 7), "navy").save(image_path)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "image_id": "0" * 64,
                "image_path": "document.png",
                "source_id": "docvqa:prior-document",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="decoded-RGB digest mismatch"):
        load_prior_identities([manifest])
