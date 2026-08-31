from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.freeze_screenqa_semantic_candidate import freeze_candidate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, eligible: bool = True) -> dict[str, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    rollouts_sha256 = "1" * 64
    protocol = tmp_path / "protocol.md"
    protocol.write_text("semantic v2 frozen\n", encoding="utf-8")
    protocol_sha256 = _sha256(protocol)
    activation = tmp_path / "activation.audit.json"
    activation.write_text(
        json.dumps(
            {
                "passed": True,
                "semantic_escalation_activated": True,
                "ranker_rollouts_sha256": rollouts_sha256,
                "v2_protocol_sha256": protocol_sha256,
                "records": 72_555,
                "decisions": 14_511,
                "sources": 1_510,
                "semantic_code_revision": "2" * 40,
                "calibration_outcomes_opened": False,
                "formal_outcomes_opened": False,
                "reserve_outcomes_opened": False,
            }
        ),
        encoding="utf-8",
    )
    features = tmp_path / "features.pt"
    features.write_bytes(b"label-free semantic fixture")
    feature_sha256 = _sha256(features)
    label_free = tmp_path / "label-free-audit.json"
    label_free.write_text(
        json.dumps(
            {
                "features_sha256": feature_sha256,
                "rollouts_sha256": rollouts_sha256,
                "decisions": 14_511,
                "outcome_fields_present": [],
                "outcomes_included_metadata": False,
            }
        ),
        encoding="utf-8",
    )
    thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    selected = {
        "answer_now_only": False,
        "risk_accepted": True,
        "source_call_rate": 0.015,
        "source_utility": 0.002,
        "threshold": 0.7,
    }
    report = {
        "feature_mode": "hybrid-context-semantic",
        "model_type": "multidomain_factorized_action_value",
        "training_protocol": "source_grouped_oof_domain_source_balanced_v2",
        "sample_weighting": "equal_domain_then_equal_source_then_equal_row",
        "seed": 20260831,
        "n_folds": 5,
        "lambda_cost": 0.05,
        "domains": ["screenqa"],
        "development_decisions": 14_511,
        "selected_alpha": 1.0,
        "selected_threshold": 0.25,
        "candidate_oof_metrics": [
            {"alpha": alpha} for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]
        ],
        "run": {
            "code_revision": "3" * 40,
            "formal_outcomes_used": False,
            "development_inputs": {
                "screenqa": {"records": 72_555, "sha256": rollouts_sha256}
            },
            "semantic_features": {
                "screenqa": {"path": str(features), "sha256": feature_sha256}
            },
        },
        "development_tail_risk_diagnostic": {
            "family_error": 0.05,
            "lambda_cost": 0.05,
            "min_source_call_rate": 0.01,
            "min_source_utility": 0.001,
            "n_decisions": 14_511,
            "n_sources": 1_510,
            "selection_objective": "source_utility",
            "valid_for_formal_selection": False,
            "constraints": [
                {"kind": "induced_harm", "limit": 0.005},
                {"kind": "net_negative_call_mass", "limit": 0.02},
            ],
            "requested_thresholds": [
                {"target_pooled_call_rate": rate, "threshold": threshold}
                for rate, threshold in zip(
                    [0.005, 0.01, 0.015, 0.02, 0.03, 0.05], thresholds
                )
            ],
            "selection_status": (
                "selected_non_degenerate_safe_threshold"
                if eligible
                else "no_non_degenerate_safe_threshold"
            ),
            "selected": selected if eligible else None,
        },
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    model = {
        key: report[key]
        for key in (
            "feature_mode",
            "model_type",
            "training_protocol",
            "sample_weighting",
            "seed",
            "n_folds",
            "lambda_cost",
            "domains",
        )
    }
    model["selected_alpha"] = report["selected_alpha"]
    model["threshold"] = report["selected_threshold"]
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(model), encoding="utf-8")
    return {
        "report_path": report_path,
        "model_path": model_path,
        "features_path": features,
        "expected_features_sha256": feature_sha256,
        "label_free_audit_path": label_free,
        "expected_label_free_audit_sha256": _sha256(label_free),
        "activation_path": activation,
        "expected_activation_sha256": _sha256(activation),
        "protocol_path": protocol,
        "expected_protocol_sha256": protocol_sha256,
        "expected_rollouts_sha256": rollouts_sha256,
        "output_dir": tmp_path / "candidate-v2",
    }


def test_screenqa_semantic_candidate_freezes_only_eligible_model(tmp_path):
    audit = freeze_candidate(**_fixture(tmp_path / "eligible"))
    output = Path(audit["selected_model"]).parent
    assert audit["candidate_frozen"] is True
    assert audit["semantic_escalation_required"] is False
    assert audit["further_ranker_search_allowed"] is False
    model = json.loads((output / "model.json").read_text(encoding="utf-8"))
    assert model["threshold"] is None
    assert model["development_oof_threshold"] == 0.25
    assert model["candidate_selection"]["calibration_outcomes_used"] is False


def test_screenqa_semantic_candidate_stops_after_sole_candidate_failure(tmp_path):
    args = _fixture(tmp_path, eligible=False)
    output = Path(args["output_dir"])
    audit = freeze_candidate(**args)
    assert audit["candidate_frozen"] is False
    assert audit["ranker_development_stopped"] is True
    assert audit["further_ranker_search_allowed"] is False
    assert not (output / "model.json").exists()
    assert (output / "candidate.audit.json").is_file()


def test_screenqa_semantic_candidate_rejects_outcome_bearing_feature_audit(tmp_path):
    args = _fixture(tmp_path)
    audit_path = Path(args["label_free_audit_path"])
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    payload["outcome_fields_present"] = ["success_after"]
    audit_path.write_text(json.dumps(payload), encoding="utf-8")
    args["expected_label_free_audit_sha256"] = _sha256(audit_path)
    with pytest.raises(ValueError, match="outcome_fields_present mismatch"):
        freeze_candidate(**args)


def test_screenqa_semantic_candidate_rejects_protocol_hash_drift(tmp_path):
    args = _fixture(tmp_path)
    args["expected_protocol_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="v2 protocol SHA-256 mismatch"):
        freeze_candidate(**args)
