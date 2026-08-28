import hashlib
import json

import pytest

from scripts.export_textvqa_train_scale_formal import (
    _prior_identities,
    _verify_freeze_components,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_formal_export_gate_verifies_frozen_components(tmp_path):
    artifact = tmp_path / "model.json"
    implementation = tmp_path / "evaluator.py"
    artifact.write_text("{}\n", encoding="utf-8")
    implementation.write_text("pass\n", encoding="utf-8")
    freeze = {
        "formal_gate_status": "ready_for_formal_manifest",
        "formal_test": {
            "allocated_sources": 5000,
            "manifest_materialized": False,
            "rollouts_collected": False,
        },
        "artifacts": {
            "model": {"path": str(artifact), "sha256": _sha256(artifact)}
        },
        "implementation": {
            "evaluator": {
                "path": str(implementation),
                "sha256": _sha256(implementation),
            }
        },
    }
    _verify_freeze_components(freeze)
    implementation.write_text("raise RuntimeError\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        _verify_freeze_components(freeze)


def test_prior_identity_reader_checks_manifest_hashes(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"image_id": "image-a", "source_id": "textvqa:source-a"})
        + "\n",
        encoding="utf-8",
    )
    banks = [
        {
            "manifest": str(manifest),
            "manifest_sha256": _sha256(manifest),
        }
    ]
    images, groups = _prior_identities(banks)
    assert images == {"image-a"}
    assert groups == {"source-a"}
