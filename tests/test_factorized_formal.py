from __future__ import annotations

import json

import pytest

from beyond_entropy.factorized_formal import (
    ALLOCATION_SHA256,
    validate_materialized_formal_gate,
    validate_policy_freeze,
    sha256_file,
)


def _write_json(path, payload):
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _freeze_payload(component):
    frozen = {"path": str(component.resolve()), "sha256": sha256_file(component)}
    return {
        "schema_version": 1,
        "formal_gate_status": "ready_for_formal_manifest",
        "formal_outcomes_used": False,
        "calibration": {
            "selection_status": "selected_non_degenerate_safe_threshold",
            "n_sources": 3000,
            "n_decisions": 4747,
            "formal_outcomes_used": False,
        },
        "formal_test": {
            "allocated_sources": 5953,
            "manifest_materialized": False,
            "rollouts_collected": False,
            "outcomes_used": False,
        },
        "artifacts": {"calibrated_model": frozen},
        "implementation": {"evaluator": frozen},
    }


def test_factorized_policy_freeze_rejects_component_tampering(tmp_path):
    component = tmp_path / "model.json"
    _write_json(component, {"threshold": 0.1})
    freeze = _freeze_payload(component)
    validate_policy_freeze(freeze)
    _write_json(component, {"threshold": 0.2})
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_policy_freeze(freeze)


def test_materialized_gate_binds_manifest_audit_and_model(tmp_path):
    model = tmp_path / "model.json"
    manifest = tmp_path / "manifest.jsonl"
    provenance = tmp_path / "manifest.provenance.json"
    audit = tmp_path / "formal.audit.json"
    freeze_path = tmp_path / "policy-freeze.json"
    _write_json(model, {"threshold": 0.1})
    manifest.write_text('{"state_id":"s0"}\n', encoding="utf-8")
    model_hash = sha256_file(model)
    manifest_hash = sha256_file(manifest)
    freeze = _freeze_payload(model)
    _write_json(freeze_path, freeze)
    freeze_hash = sha256_file(freeze_path)
    _write_json(
        provenance,
        {
            "selection_metadata": {
                "policy_freeze_sha256": freeze_hash,
                "allocation_sha256": ALLOCATION_SHA256,
                "role": "formal_test",
                "selected_source_group_count": 5953,
                "selection_uses_targets": False,
            }
        },
    )
    _write_json(
        audit,
        {
            "passed": True,
            "policy_freeze_sha256": freeze_hash,
            "allocation_sha256": ALLOCATION_SHA256,
            "formal": {
                "manifest_sha256": manifest_hash,
                "unique_sources": 5953,
            },
            "overlap": {"formal_calibration_sources": 0},
        },
    )
    validated = validate_materialized_formal_gate(
        policy_freeze_path=freeze_path,
        expected_policy_freeze_sha256=freeze_hash,
        model_path=model,
        expected_model_sha256=model_hash,
        manifest_path=manifest,
        expected_manifest_sha256=manifest_hash,
        manifest_provenance_path=provenance,
        expected_manifest_provenance_sha256=sha256_file(provenance),
        audit_path=audit,
        expected_audit_sha256=sha256_file(audit),
    )
    assert validated["formal_test"]["allocated_sources"] == 5953


def test_materialized_gate_rejects_identity_overlap(tmp_path):
    model = tmp_path / "model.json"
    manifest = tmp_path / "manifest.jsonl"
    provenance = tmp_path / "manifest.provenance.json"
    audit = tmp_path / "formal.audit.json"
    freeze_path = tmp_path / "policy-freeze.json"
    _write_json(model, {"threshold": 0.1})
    manifest.write_text('{"state_id":"s0"}\n', encoding="utf-8")
    freeze = _freeze_payload(model)
    _write_json(freeze_path, freeze)
    freeze_hash = sha256_file(freeze_path)
    manifest_hash = sha256_file(manifest)
    _write_json(
        provenance,
        {
            "selection_metadata": {
                "policy_freeze_sha256": freeze_hash,
                "allocation_sha256": ALLOCATION_SHA256,
                "role": "formal_test",
                "selected_source_group_count": 5953,
                "selection_uses_targets": False,
            }
        },
    )
    _write_json(
        audit,
        {
            "passed": True,
            "policy_freeze_sha256": freeze_hash,
            "allocation_sha256": ALLOCATION_SHA256,
            "formal": {
                "manifest_sha256": manifest_hash,
                "unique_sources": 5953,
            },
            "overlap": {"formal_calibration_sources": 1},
        },
    )
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
