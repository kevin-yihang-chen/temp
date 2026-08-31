from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.select_screenqa_ranker_candidate import select_candidate


def _write_candidate(
    root: Path,
    *,
    feature_mode: str,
    source_utility: float,
    eligible: bool = True,
) -> tuple[Path, Path]:
    root.mkdir()
    selected = {
        "answer_now_only": False,
        "risk_accepted": eligible,
        "source_call_rate": 0.015 if eligible else 0.0,
        "source_utility": source_utility,
        "threshold": 0.9,
    }
    report = {
        "feature_mode": feature_mode,
        "model_type": "multidomain_factorized_action_value",
        "training_protocol": "source_grouped_oof_domain_source_balanced_v2",
        "sample_weighting": "equal_domain_then_equal_source_then_equal_row",
        "seed": 20260831,
        "n_folds": 5,
        "lambda_cost": 0.05,
        "domains": ["screenqa"],
        "development_decisions": 14511,
        "selected_alpha": 1.0,
        "selected_threshold": 0.25,
        "candidate_oof_metrics": [
            {"alpha": alpha}
            for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]
        ],
        "run": {
            "code_revision": "1" * 40,
            "formal_outcomes_used": False,
            "development_inputs": {
                "screenqa": {"records": 72555, "sha256": "2" * 64}
            },
        },
        "development_tail_risk_diagnostic": {
            "family_error": 0.05,
            "lambda_cost": 0.05,
            "min_source_call_rate": 0.01,
            "min_source_utility": 0.001,
            "n_decisions": 14511,
            "n_sources": 1510,
            "selection_objective": "source_utility",
            "valid_for_formal_selection": False,
            "selection_status": (
                "selected_non_degenerate_safe_threshold"
                if eligible
                else "no_non_degenerate_safe_threshold"
            ),
            "selected": selected if eligible else None,
            "requested_thresholds": [
                {"target_pooled_call_rate": rate, "threshold": 1.0 - index / 10}
                for index, rate in enumerate(
                    [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
                )
            ],
            "constraints": [
                {"kind": "induced_harm", "limit": 0.005},
                {"kind": "net_negative_call_mass", "limit": 0.02},
            ],
        },
    }
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
    report_path = root / "report.json"
    model_path = root / "model.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    model_path.write_text(json.dumps(model), encoding="utf-8")
    return report_path, model_path


def _run_selection(tmp_path: Path, context_utility: float, spatial_utility: float):
    context_report, context_model = _write_candidate(
        tmp_path / "context",
        feature_mode="context-geometry",
        source_utility=context_utility,
    )
    spatial_report, spatial_model = _write_candidate(
        tmp_path / "spatial",
        feature_mode="spatial-context-geometry",
        source_utility=spatial_utility,
    )
    protocol = tmp_path / "protocol.md"
    protocol.write_text("frozen", encoding="utf-8")
    protocol_sha = hashlib.sha256(protocol.read_bytes()).hexdigest()
    return select_candidate(
        context_report=context_report,
        context_model=context_model,
        spatial_report=spatial_report,
        spatial_model=spatial_model,
        protocol=protocol,
        expected_protocol_sha256=protocol_sha,
        output_dir=tmp_path / "selected",
    )


def test_screenqa_selection_prefers_lower_capacity_within_tie_margin(tmp_path):
    audit = _run_selection(tmp_path, 0.0020, 0.0022)
    assert audit["candidate_frozen"] is True
    assert audit["selected_feature_mode"] == "context-geometry"
    assert audit["selection_reason"] == "utility_tie_within_margin_choose_lower_capacity"
    model = json.loads((tmp_path / "selected" / "model.json").read_text())
    assert model["development_oof_threshold"] == 0.25
    assert model["threshold"] is None
    assert model["threshold_grid"] == [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    assert model["calibration_contract"]["calibration_sources"] == 1016
    assert model["calibration_contract"]["formal_sources"] == 1471
    assert model["candidate_selection"]["calibration_outcomes_used"] is False


def test_screenqa_selection_uses_larger_registered_source_utility(tmp_path):
    audit = _run_selection(tmp_path, 0.0020, 0.0023)
    assert audit["selected_feature_mode"] == "spatial-context-geometry"
    assert audit["selection_reason"] == "largest_registered_source_utility"


def test_screenqa_selection_stops_when_no_candidate_is_eligible(tmp_path):
    context_report, context_model = _write_candidate(
        tmp_path / "context",
        feature_mode="context-geometry",
        source_utility=0.0,
        eligible=False,
    )
    spatial_report, spatial_model = _write_candidate(
        tmp_path / "spatial",
        feature_mode="spatial-context-geometry",
        source_utility=0.0,
        eligible=False,
    )
    protocol = tmp_path / "protocol.md"
    protocol.write_text("frozen", encoding="utf-8")
    audit = select_candidate(
        context_report=context_report,
        context_model=context_model,
        spatial_report=spatial_report,
        spatial_model=spatial_model,
        protocol=protocol,
        expected_protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        output_dir=tmp_path / "selected",
    )
    assert audit["candidate_frozen"] is False
    assert audit["semantic_escalation_required"] is True
    assert not (tmp_path / "selected" / "model.json").exists()


def test_screenqa_selection_rejects_protocol_hash_drift(tmp_path):
    context_report, context_model = _write_candidate(
        tmp_path / "context",
        feature_mode="context-geometry",
        source_utility=0.002,
    )
    spatial_report, spatial_model = _write_candidate(
        tmp_path / "spatial",
        feature_mode="spatial-context-geometry",
        source_utility=0.002,
    )
    protocol = tmp_path / "protocol.md"
    protocol.write_text("frozen", encoding="utf-8")
    with pytest.raises(ValueError, match="protocol hash mismatch"):
        select_candidate(
            context_report=context_report,
            context_model=context_model,
            spatial_report=spatial_report,
            spatial_model=spatial_model,
            protocol=protocol,
            expected_protocol_sha256="0" * 64,
            output_dir=tmp_path / "selected",
        )
