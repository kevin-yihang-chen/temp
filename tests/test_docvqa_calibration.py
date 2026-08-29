from __future__ import annotations

import copy

import pytest

from beyond_entropy.docvqa_calibration import (
    MODEL_REVISION,
    calibrate_frozen_candidate_rows,
    validate_calibration_bundle,
    validate_calibration_feature_metadata,
)
from beyond_entropy.docvqa_candidate_freeze import PROTOCOL_SHA256
from beyond_entropy.risk_control import AcquisitionCalibrationRow


def _candidate():
    thresholds = [2.0, 1.0, 0.0]
    return {
        "model_type": "multidomain_factorized_action_value",
        "training_protocol": "source_grouped_oof_domain_source_balanced_v2",
        "sample_weighting": "equal_domain_then_equal_source_then_equal_row",
        "feature_mode": "hybrid-context-semantic",
        "seed": 20260829,
        "n_folds": 5,
        "lambda_cost": 0.05,
        "selected_alpha": 1.0,
        "domains": ["docvqa"],
        "state_feature_count": 27,
        "action_feature_count": 46,
        "threshold": None,
        "threshold_grid": thresholds,
        "calibration_contract": {
            "method": "fixed_sequence_bounded_mean_kl_ltt_v1",
            "threshold_order": "strict_to_permissive_descending",
            "threshold_rate_weighting": "equal_source_then_equal_question",
            "target_source_balanced_development_call_rates": [0.01],
            "threshold_summaries": [
                {
                    "threshold": threshold,
                    "source_balanced_development_call_rate": 0.0,
                    "pooled_development_call_rate": 0.0,
                }
                for threshold in thresholds
            ],
            "constraints": [
                {"kind": "induced_harm", "limit": 0.005},
                {"kind": "net_negative_call_mass", "limit": 0.02},
            ],
            "family_error": 0.05,
            "per_step_p_cutoff": 0.025,
            "min_source_call_rate": 0.01,
            "min_source_utility": 0.001,
            "calibration_sources": 2500,
            "formal_sources": 3500,
        },
        "candidate_freeze": {
            "ranker_training_outcomes_used": True,
            "calibration_outcomes_used": False,
            "formal_outcomes_used": False,
        },
    }


def _positive_rows(count: int = 2500):
    return [
        AcquisitionCalibrationRow(
            source_id=f"document-{index:04d}",
            score=1.0 if index < 500 else 0.0,
            gain=0.2 if index < 500 else -0.2,
        )
        for index in range(count)
    ]


def test_docvqa_fixed_sequence_selects_safe_nondegenerate_threshold():
    calibration, model = calibrate_frozen_candidate_rows(
        _candidate(),
        _positive_rows(),
        expected_sources=2500,
        run_provenance={"candidate_sha256": "a" * 64},
    )
    assert calibration["selection_status"] == "selected_non_degenerate_safe_threshold"
    assert calibration["selected_threshold"] == 1.0
    assert calibration["tested_threshold_count"] == 3
    assert calibration["stopping_threshold"] == 0.0
    assert calibration["run"]["ranker_training_outcomes_used"] is True
    assert calibration["run"]["calibration_outcomes_used"] is True
    assert calibration["run"]["formal_outcomes_used"] is False
    assert model["threshold"] == 1.0
    assert model["risk_calibration"]["selected_threshold"] == 1.0


def test_docvqa_fixed_sequence_failure_returns_answer_now():
    rows = [
        AcquisitionCalibrationRow(
            source_id=f"document-{index:04d}",
            score=1.0,
            gain=-0.5,
        )
        for index in range(2500)
    ]
    calibration, model = calibrate_frozen_candidate_rows(
        _candidate(),
        rows,
        expected_sources=2500,
    )
    assert calibration["selection_status"] == "no_non_degenerate_safe_threshold"
    assert calibration["selected_threshold"] is None
    assert calibration["stopping_threshold"] == 1.0
    assert model["threshold"] is None


def test_docvqa_calibration_rejects_source_or_candidate_drift():
    with pytest.raises(ValueError, match="2501 source groups"):
        calibrate_frozen_candidate_rows(
            _candidate(),
            _positive_rows(),
            expected_sources=2501,
        )
    candidate = _candidate()
    candidate["calibration_contract"]["family_error"] = 0.1
    with pytest.raises(ValueError, match="family_error"):
        calibrate_frozen_candidate_rows(
            candidate,
            _positive_rows(),
            expected_sources=2500,
        )


def _calibration_bundle():
    candidate = _candidate()
    candidate["candidate_freeze"].update(
        {
            "protocol_sha256": PROTOCOL_SHA256,
            "allocation_sha256": "c" * 64,
            "code_revision": "b" * 40,
        }
    )
    candidate_audit = {
        "passed": True,
        "scientific_status": (
            "sole DocVQA factorized-v2 candidate frozen before calibration export"
        ),
        "candidate_sha256": "a" * 64,
        "protocol_sha256": PROTOCOL_SHA256,
        "development_sources": 3500,
        "threshold_count": 3,
        "ranker_training_outcomes_used": True,
        "calibration_outcomes_used": False,
        "formal_outcomes_used": False,
        "code_revision": "b" * 40,
    }
    allocation = {
        "protocol_sha256": PROTOCOL_SHA256,
        "selection_contract": {
            "selection_target_fields_accessed": False,
            "selection_allowed_fields": ["docId", "image"],
            "ranker_manifest_exported": False,
            "calibration_manifest_exported": False,
            "formal_manifest_exported": False,
            "ranker_outcomes_collected": False,
            "calibration_outcomes_collected": False,
            "formal_outcomes_collected": False,
        },
        "allocation": {
            "roles": {
                "ranker_training": {"count": 3500},
                "risk_calibration": {"count": 2500},
                "formal_test": {"count": 3500},
            }
        },
    }
    allocation_audit = {
        "passed": True,
        "allocation_sha256": "c" * 64,
        "protocol_sha256": PROTOCOL_SHA256,
        "ranker_outcomes_collected": False,
        "calibration_outcomes_collected": False,
        "formal_outcomes_collected": False,
    }
    manifest_audit = {
        "task": "docvqa",
        "dataset_id": "lmms-lab/DocVQA",
        "dataset_name": "DocVQA",
        "dataset_revision": "539088ef8a8ada01ac8e2e6d4e372586748a265e",
        "split": "train",
        "scorer": "docvqa",
        "count": 10000,
        "unique_states": 10000,
        "unique_sources": 2500,
        "unique_images": 2500,
        "manifest_sha256": "d" * 64,
        "selection_metadata": {
            "allocation_sha256": "c" * 64,
            "candidate_sha256": "a" * 64,
            "protocol_sha256": PROTOCOL_SHA256,
            "namespace": "beyond-entropy-docvqa-train-factorized-v2",
            "role": "risk_calibration",
            "selected_source_group_count": 2500,
            "selection_uses_targets": False,
        },
    }
    rollout_audit = {
        "passed": True,
        "manifest_sha256": "d" * 64,
        "manifest_provenance_sha256": "e" * 64,
        "rollouts_sha256": "f" * 64,
        "model_revision": MODEL_REVISION,
        "code_revision": "b" * 40,
        "scientific_status": (
            "fresh DocVQA-train factorized-v2 calibration sibling bank; outcomes "
            "may calibrate the sole frozen candidate only"
        ),
        "candidate_sha256": "a" * 64,
        "candidate_audit_sha256": "1" * 64,
        "allocation_sha256": "c" * 64,
        "allocation_audit_sha256": "2" * 64,
        "protocol_sha256": PROTOCOL_SHA256,
        "states": 10000,
        "records": 50000,
        "unique_sources": 2500,
        "unique_images": 2500,
        "candidate_count": 4,
        "answer_records": 10000,
        "zoom_records": 40000,
    }
    return (
        candidate,
        candidate_audit,
        allocation,
        allocation_audit,
        manifest_audit,
        rollout_audit,
    )


def test_docvqa_calibration_bundle_binds_every_gate():
    bundle = _calibration_bundle()
    assert validate_calibration_bundle(
        *bundle,
        candidate_sha256="a" * 64,
        candidate_audit_sha256="1" * 64,
        allocation_sha256="c" * 64,
        allocation_audit_sha256="2" * 64,
        manifest_sha256="d" * 64,
        manifest_provenance_sha256="e" * 64,
        rollouts_sha256="f" * 64,
        code_revision="b" * 40,
    ) == 10000
    bundle[-1]["model_revision"] = "changed"
    with pytest.raises(ValueError, match="model_revision"):
        validate_calibration_bundle(
            *bundle,
            candidate_sha256="a" * 64,
            candidate_audit_sha256="1" * 64,
            allocation_sha256="c" * 64,
            allocation_audit_sha256="2" * 64,
            manifest_sha256="d" * 64,
            manifest_provenance_sha256="e" * 64,
            rollouts_sha256="f" * 64,
            code_revision="b" * 40,
        )


def test_docvqa_calibration_feature_metadata_binds_all_three_stages():
    metadata = {
        "source_rollouts_sha256": "f" * 64,
        "model_revision": MODEL_REVISION,
        "code_revision": "b" * 40,
        "outcomes_included": False,
        "question_reembedding": {
            "source_rollouts_sha256": "f" * 64,
            "model_revision": MODEL_REVISION,
            "code_revision": "b" * 40,
            "completed_decisions": 2,
            "total_decisions": 2,
        },
        "question_region_attention": {
            "source_rollouts_sha256": "f" * 64,
            "model_revision": MODEL_REVISION,
            "code_revision": "b" * 40,
            "completed_decisions": 2,
            "total_decisions": 2,
        },
    }
    features = {"metadata": metadata, "decisions": [{}, {}]}
    validate_calibration_feature_metadata(
        features,
        rollouts_sha256="f" * 64,
        code_revision="b" * 40,
        expected_decisions=2,
    )
    metadata["question_region_attention"]["completed_decisions"] = 1
    with pytest.raises(ValueError, match="completed_decisions"):
        validate_calibration_feature_metadata(
            features,
            rollouts_sha256="f" * 64,
            code_revision="b" * 40,
            expected_decisions=2,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("candidate_allocation", "candidate allocation"),
        ("manifest_states", "manifest unique states"),
        ("selection_targets", "selection_uses_targets"),
        ("rollout_candidate_audit", "candidate_audit_sha256"),
    ],
)
def test_docvqa_calibration_bundle_rejects_cross_artifact_drift(
    mutation: str,
    message: str,
):
    bundle = list(copy.deepcopy(_calibration_bundle()))
    if mutation == "candidate_allocation":
        bundle[0]["candidate_freeze"]["allocation_sha256"] = "changed"
    elif mutation == "manifest_states":
        bundle[4]["unique_states"] = 9999
    elif mutation == "selection_targets":
        bundle[4]["selection_metadata"]["selection_uses_targets"] = True
    elif mutation == "rollout_candidate_audit":
        bundle[5]["candidate_audit_sha256"] = "changed"
    with pytest.raises(ValueError, match=message):
        validate_calibration_bundle(
            *bundle,
            candidate_sha256="a" * 64,
            candidate_audit_sha256="1" * 64,
            allocation_sha256="c" * 64,
            allocation_audit_sha256="2" * 64,
            manifest_sha256="d" * 64,
            manifest_provenance_sha256="e" * 64,
            rollouts_sha256="f" * 64,
            code_revision="b" * 40,
        )
