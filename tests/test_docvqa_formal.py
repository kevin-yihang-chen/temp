from __future__ import annotations

import json

import pytest

from beyond_entropy.docvqa_formal import (
    FORMAL_SOURCES,
    POLICY_FREEZE_SCIENTIFIC_STATUS,
    REQUIRED_ARTIFACTS,
    REQUIRED_IMPLEMENTATION,
    validate_materialized_formal_gate,
    validate_policy_freeze,
)
from beyond_entropy.docvqa_train_allocation import PROTOCOL_SHA256, sha256_file


def _write_json(path, payload):
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _freeze(component):
    frozen = {"path": str(component.resolve()), "sha256": sha256_file(component)}
    return {
        "schema_version": 1,
        "scientific_status": POLICY_FREEZE_SCIENTIFIC_STATUS,
        "formal_gate_status": "ready_for_formal_manifest",
        "formal_outcomes_used": False,
        "code_revision": "a" * 40,
        "implementation_commit_time_unix": 1,
        "implementation_committed_before_calibration_output": True,
        "calibration": {
            "selection_status": "selected_non_degenerate_safe_threshold",
            "selected_threshold": 0.1,
            "n_sources": 2500,
            "n_decisions": 3000,
            "formal_outcomes_used": False,
        },
        "formal_test": {
            "allocated_sources": FORMAL_SOURCES,
            "manifest_materialized": False,
            "rollouts_collected": False,
            "outcomes_used": False,
            "bootstrap_resamples": 20000,
            "bootstrap_confidence": 0.975,
            "bootstrap_seed": 20260829,
        },
        "artifacts": {name: frozen for name in REQUIRED_ARTIFACTS},
        "implementation": {name: frozen for name in REQUIRED_IMPLEMENTATION},
    }


def test_docvqa_policy_freeze_rejects_component_or_bootstrap_drift(tmp_path):
    component = tmp_path / "component.json"
    _write_json(component, {"threshold": 0.1})
    freeze = _freeze(component)
    validate_policy_freeze(freeze)
    freeze["formal_test"]["bootstrap_seed"] = 0
    with pytest.raises(ValueError, match="bootstrap_seed"):
        validate_policy_freeze(freeze)
    freeze = _freeze(component)
    _write_json(component, {"threshold": 0.2})
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_policy_freeze(freeze)


def test_docvqa_materialized_gate_binds_policy_manifest_and_identities(tmp_path):
    model = tmp_path / "model.json"
    allocation = tmp_path / "allocation.json"
    manifest = tmp_path / "manifest.jsonl"
    provenance = tmp_path / "manifest.provenance.json"
    audit = tmp_path / "formal.audit.json"
    freeze_path = tmp_path / "policy-freeze.json"
    _write_json(model, {"threshold": 0.1})
    _write_json(allocation, {"allocation": True})
    manifest.write_text('{"state_id":"s0"}\n', encoding="utf-8")
    freeze = _freeze(model)
    freeze["artifacts"]["allocation"] = {
        "path": str(allocation.resolve()),
        "sha256": sha256_file(allocation),
    }
    _write_json(freeze_path, freeze)
    freeze_hash = sha256_file(freeze_path)
    manifest_hash = sha256_file(manifest)
    _write_json(
        provenance,
        {
            "code_revision": "a" * 40,
            "formal_outcomes_collected": False,
            "selection_metadata": {
                "policy_freeze_sha256": freeze_hash,
                "allocation_sha256": sha256_file(allocation),
                "protocol_sha256": PROTOCOL_SHA256,
                "role": "formal_test",
                "selected_source_group_count": FORMAL_SOURCES,
                "selection_uses_targets": False,
            }
        },
    )
    _write_json(
        audit,
        {
            "passed": True,
            "policy_freeze_sha256": freeze_hash,
            "allocation_sha256": sha256_file(allocation),
            "formal_outcomes_collected": False,
            "formal": {
                "manifest_sha256": manifest_hash,
                "count": FORMAL_SOURCES,
                "unique_states": FORMAL_SOURCES,
                "unique_sources": FORMAL_SOURCES,
                "unique_images": FORMAL_SOURCES,
            },
            "overlap": {"formal_calibration_sources": 0},
        },
    )
    validated = validate_materialized_formal_gate(
        policy_freeze_path=freeze_path,
        expected_policy_freeze_sha256=freeze_hash,
        model_path=model,
        expected_model_sha256=sha256_file(model),
        manifest_path=manifest,
        expected_manifest_sha256=manifest_hash,
        manifest_provenance_path=provenance,
        expected_manifest_provenance_sha256=sha256_file(provenance),
        audit_path=audit,
        expected_audit_sha256=sha256_file(audit),
    )
    assert validated["formal_test"]["allocated_sources"] == FORMAL_SOURCES
    payload = json.loads(audit.read_text())
    payload["overlap"]["formal_calibration_sources"] = 1
    _write_json(audit, payload)
    with pytest.raises(ValueError, match="identity overlap"):
        validate_materialized_formal_gate(
            policy_freeze_path=freeze_path,
            expected_policy_freeze_sha256=freeze_hash,
            model_path=model,
            expected_model_sha256=sha256_file(model),
            manifest_path=manifest,
            expected_manifest_sha256=manifest_hash,
            manifest_provenance_path=provenance,
            expected_manifest_provenance_sha256=sha256_file(provenance),
            audit_path=audit,
            expected_audit_sha256=sha256_file(audit),
        )


def test_docvqa_policy_freeze_requires_complete_artifact_inventory(tmp_path):
    component = tmp_path / "component.json"
    _write_json(component, {"threshold": 0.1})
    freeze = _freeze(component)
    del freeze["implementation"]["formal_evaluator_script"]
    with pytest.raises(ValueError, match="formal_evaluator_script"):
        validate_policy_freeze(freeze)
