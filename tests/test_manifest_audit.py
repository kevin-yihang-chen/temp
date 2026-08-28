from __future__ import annotations

import json

import pytest
from PIL import Image

from beyond_entropy.manifest_audit import audit_manifest_pair
from beyond_entropy.manifest_export import export_benchmark_manifest


def _docvqa_row(document_id: str, question_id: str):
    return {
        "image": Image.new("RGB", (8, 8), document_id),
        "questionId": question_id,
        "question": "What is shown?",
        "question_types": ["layout"],
        "docId": document_id,
        "answers": ["text"],
    }


def _export(directory, rows, source_indices):
    return export_benchmark_manifest(
        rows,
        source_indices=source_indices,
        task="docvqa",
        dataset_id="lmms-lab/DocVQA",
        dataset_revision="revision",
        output_dir=directory,
        seed=1,
        selection="test selection",
        selection_metadata={"namespace": "test"},
    )


def test_manifest_pair_audit_binds_files_and_proves_zero_overlap(tmp_path):
    development = tmp_path / "development"
    formal = tmp_path / "formal"
    _export(development, [_docvqa_row("red", "d1")], [1])
    _export(formal, [_docvqa_row("blue", "f1")], [2])
    report = audit_manifest_pair(
        development,
        formal,
        task="docvqa",
        expected_revision="revision",
    )
    assert report["passed"] is True
    assert report["overlap"] == {"states": 0, "sources": 0, "images": 0}
    assert report["development"]["count"] == 1
    assert len(report["formal"]["image_bundle_sha256"]) == 64


def test_manifest_pair_audit_rejects_source_leakage(tmp_path):
    development = tmp_path / "development"
    formal = tmp_path / "formal"
    _export(development, [_docvqa_row("red", "d1")], [1])
    _export(formal, [_docvqa_row("red", "f1")], [2])
    with pytest.raises(ValueError, match="leakage"):
        audit_manifest_pair(
            development,
            formal,
            task="docvqa",
            expected_revision="revision",
        )


def test_manifest_audit_rejects_changed_manifest_bytes(tmp_path):
    development = tmp_path / "development"
    formal = tmp_path / "formal"
    _export(development, [_docvqa_row("red", "d1")], [1])
    _export(formal, [_docvqa_row("blue", "f1")], [2])
    manifest = development / "manifest.jsonl"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["question"] = "changed"
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        audit_manifest_pair(
            development,
            formal,
            task="docvqa",
            expected_revision="revision",
        )
