from __future__ import annotations

import json

import pytest

from beyond_entropy.docvqa_candidate_freeze import (
    PROTOCOL_SHA256,
    build_frozen_candidate,
    serialized_sha256,
)


def _raw_inputs():
    common = {
        "model_type": "multidomain_factorized_action_value",
        "training_protocol": "source_grouped_oof_domain_source_balanced_v2",
        "sample_weighting": "equal_domain_then_equal_source_then_equal_row",
        "feature_mode": "hybrid-context-semantic",
        "seed": 20260829,
        "n_folds": 5,
        "lambda_cost": 0.05,
        "selected_alpha": 1.0,
        "domains": ["docvqa"],
    }
    model = {
        **common,
        "state_feature_count": 27,
        "action_feature_count": 46,
        "threshold": 0.25,
        "decision_rule": "development-only rule",
    }
    report = {
        **common,
        "refit": {"state_feature_count": 27, "action_feature_count": 46},
        "run": {
            "formal_outcomes_used": False,
            "development_inputs": {"docvqa": {"path": "rollouts.jsonl"}},
            "semantic_features": {"docvqa": {"path": "features.pt"}},
        },
    }
    provenance = {
        "raw_model": "model.json",
        "raw_model_sha256": "1" * 64,
        "development_report": "report.json",
        "development_report_sha256": "2" * 64,
        "development_rollouts": "rollouts.jsonl",
        "development_rollouts_sha256": "3" * 64,
        "development_features": "features.pt",
        "development_features_sha256": "4" * 64,
        "allocation": "allocation.json",
        "allocation_sha256": "5" * 64,
        "allocation_audit": "allocation.audit.json",
        "allocation_audit_sha256": "6" * 64,
        "protocol": "protocol.md",
        "protocol_sha256": PROTOCOL_SHA256,
    }
    return model, report, provenance


def test_docvqa_candidate_freeze_is_deterministic_and_outcome_sealed():
    model, report, provenance = _raw_inputs()
    scores = {
        ("state-1", "replicate-000"): 0.9,
        ("state-2", "replicate-000"): 0.8,
        ("state-3", "replicate-000"): 0.7,
        ("state-4", "replicate-000"): 0.6,
    }
    sources = {
        ("state-1", "replicate-000"): "document-many",
        ("state-2", "replicate-000"): "document-many",
        ("state-3", "replicate-000"): "document-many",
        ("state-4", "replicate-000"): "document-one",
    }
    first_candidate, first_audit = build_frozen_candidate(
        model,
        report,
        scores_by_key=scores,
        source_by_key=sources,
        provenance=provenance,
        code_revision="7" * 40,
        expected_sources=2,
    )
    second_candidate, second_audit = build_frozen_candidate(
        model,
        report,
        scores_by_key=scores,
        source_by_key=sources,
        provenance=provenance,
        code_revision="7" * 40,
        expected_sources=2,
    )
    assert first_candidate == second_candidate
    assert first_audit == second_audit
    assert first_candidate["threshold"] is None
    assert first_candidate["development_oof_threshold"] == 0.25
    assert first_candidate["calibration_contract"]["calibration_sources"] == 2500
    assert first_candidate["calibration_contract"]["formal_sources"] == 3500
    assert first_candidate["candidate_freeze"]["calibration_outcomes_used"] is False
    assert first_candidate["candidate_freeze"]["formal_outcomes_used"] is False
    assert first_audit["candidate_sha256"] == serialized_sha256(first_candidate)
    assert json.dumps(first_candidate, allow_nan=False)


def test_docvqa_candidate_freeze_rejects_contract_or_source_change():
    model, report, provenance = _raw_inputs()
    scores = {("state-1", "replicate-000"): 0.5}
    sources = {("state-1", "replicate-000"): "document-1"}
    changed_model = dict(model)
    changed_model["selected_alpha"] = 10.0
    with pytest.raises(ValueError, match="selected_alpha"):
        build_frozen_candidate(
            changed_model,
            report,
            scores_by_key=scores,
            source_by_key=sources,
            provenance=provenance,
            code_revision="7" * 40,
            expected_sources=1,
        )
    with pytest.raises(ValueError, match="2 ranker-training sources"):
        build_frozen_candidate(
            model,
            report,
            scores_by_key=scores,
            source_by_key=sources,
            provenance=provenance,
            code_revision="7" * 40,
            expected_sources=2,
        )


def test_docvqa_candidate_freeze_rejects_nonfinite_artifact_value():
    model, report, provenance = _raw_inputs()
    model["unused_numeric_payload"] = float("nan")
    with pytest.raises(ValueError, match="Out of range float values"):
        build_frozen_candidate(
            model,
            report,
            scores_by_key={("state-1", "replicate-000"): 0.5},
            source_by_key={("state-1", "replicate-000"): "document-1"},
            provenance=provenance,
            code_revision="7" * 40,
            expected_sources=1,
        )
