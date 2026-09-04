from __future__ import annotations

import pytest

from beyond_entropy.predictability_audit import (
    AUDIT_BENCHMARKS,
    PREDICTOR_LEVELS,
    TARGET_FAMILIES,
    AuditVerdict,
)
from beyond_entropy.predictability_verdict import (
    classify_formal_report,
    render_predictability_audit,
    validate_completed_formal_report,
    write_predictability_audit,
)


def _policy(utility: float = 0.01) -> dict[str, float | int]:
    return {
        "decisions": 10,
        "calls": 2,
        "accuracy": 0.7,
        "cost": 0.2,
        "call_rate": 0.2,
        "incremental_utility": utility,
        "rescue_precision": 0.5,
        "harm_rate_per_call": 0.1,
        "marginal_gain_per_call": 0.4,
    }


def _paired(lower: float = 0.001, upper: float = 0.02) -> dict[str, float | int | str]:
    return {
        "point": 0.01,
        "candidate_utility": 0.01,
        "baseline_utility": 0.0,
        "lower": lower,
        "upper": upper,
        "confidence_level": 0.95,
        "resamples": 20_000,
        "seed": 1,
        "resampling_unit": "source_id",
        "sources": 10,
    }


def _formal_report() -> dict:
    baseline = _policy(0.0)
    baseline.update(
        {
            "accuracy": 0.69,
            "cost": 0.3,
            "rescue_precision": 0.4,
            "harm_rate_per_call": 0.1,
        }
    )
    split_audits = {}
    strong_baselines = {}
    headrooms = {}
    primaries = {}
    post_actions = {}
    representations = {}
    for benchmark in AUDIT_BENCHMARKS:
        split_audits[benchmark] = {
            "passed": True,
            "test_role_validation": {"passed": True},
        }
        strong_baselines[benchmark] = {
            "strongest_baseline": "entropy_gate_fixed_visual_tool",
            "test": {"entropy_gate_fixed_visual_tool": baseline},
        }
        headrooms[benchmark] = {
            "always_call": {"utility": 0.01},
            "privileged_binary_oracle": {
                "utility": 0.02,
                "paired_vs_answer_now": _paired(),
            },
            "raw_targets": {"rescue_rate": 0.1, "harm_rate": 0.03},
        }
        primaries[benchmark] = {
            "strongest_baseline": "entropy_gate_fixed_visual_tool",
            "selected_cell_keys": [
                {
                    "level": "l0_uncertainty",
                    "target": "direct_gain",
                    "seed": seed,
                }
                for seed in (17, 29, 47)
            ],
            "test_policy": _policy(),
            "paired_vs_strongest_baseline": _paired(-0.001, 0.02),
            "maximum_lower_ci_across_all_deployable_cells_and_primary": -0.001,
            "operating_point_vs_strongest_baseline": {
                "candidate_accuracy": 0.7,
                "baseline_accuracy": 0.69,
                "candidate_cost": 0.2,
                "baseline_cost": 0.3,
                "accuracy_cost_pareto": True,
                "candidate_rescue_precision": 0.5,
                "baseline_rescue_precision": 0.4,
                "rescue_precision_higher": True,
                "candidate_harm_rate_per_call": 0.1,
                "baseline_harm_rate_per_call": 0.1,
                "harm_rate_not_higher": True,
            },
        }
        post_actions[benchmark] = {
            "ensemble": {"paired_vs_answer_now": _paired(0.001, 0.03)}
        }
        representations[benchmark] = {
            "validation_paired_vs_strongest_baseline": _paired(-0.001, 0.02),
            "test_paired_vs_strongest_baseline": _paired(-0.01, 0.01),
        }
    prediction = {
        "auroc": 0.7,
        "auprc": 0.3,
        "brier": 0.2,
        "calibration_error": 0.1,
        "rescue_auprc": 0.25,
        "harm_auprc": 0.2,
    }
    cells = [
        {
            "benchmark": benchmark,
            "predictor_level": level,
            "target": target,
            "seeds": [
                {
                    "seed": seed,
                    "test_policy": _policy(),
                    "test_prediction": prediction,
                    "paired_vs_strongest_baseline": _paired(),
                    "test_curve": [
                        {
                            "requested_call_rate": rate,
                            **_policy(utility=0.01 * rate),
                        }
                        for rate in (0.0, 1.0)
                    ],
                }
                for seed in (17, 29, 47)
            ],
        }
        for benchmark in AUDIT_BENCHMARKS
        for level in PREDICTOR_LEVELS
        for target in TARGET_FAMILIES
    ]
    return {
        "schema": "predictability_matrix_report_v3",
        "formal_claim_eligible": True,
        "frozen_before_test": True,
        "seeds": [17, 29, 47],
        "matrix": {
            "complete": True,
            "expected_cells": 36,
            "completed_cells": 36,
            "missing": [],
        },
        "split_audits": split_audits,
        "strong_baselines": strong_baselines,
        "oracle_headroom": headrooms,
        "primary_deployable": primaries,
        "post_action_probe": post_actions,
        "representation_diagnostic": representations,
        "cells": cells,
        "one_shot_test_access": {
            "ledger": "/tmp/access.json",
            "ledger_sha256": "1" * 64,
            "frozen_model_sha256": "2" * 64,
            "frozen_report_sha256": "3" * 64,
            "protocol_sha256": "4" * 64,
            "allocation_report_sha256": "6" * 64,
            "code_revision": "5" * 40,
            "test_artifacts": {},
        },
    }


def test_formal_report_renders_only_a_registered_terminal_verdict(tmp_path) -> None:
    report = _formal_report()
    validate_completed_formal_report(report)
    assert classify_formal_report(report) == AuditVerdict.PIVOT
    rendered = render_predictability_audit(report, report_sha256="a" * 64)
    assert "**Frozen verdict: PIVOT.**" in rendered
    assert "## Predictor ladder" in rendered
    assert "## Accuracy-cost frontier" in rendered
    assert "20,000 resamples" in rendered
    output = tmp_path / "PREDICTABILITY_AUDIT.md"
    write_predictability_audit(output, rendered)
    assert output.read_text() == rendered
    with pytest.raises(FileExistsError):
        write_predictability_audit(output, rendered)


def test_final_audit_rejects_wrong_filename_and_incomplete_matrix(tmp_path) -> None:
    with pytest.raises(ValueError, match="filename"):
        write_predictability_audit(tmp_path / "other.md", "text")
    report = _formal_report()
    report["matrix"]["completed_cells"] = 35
    with pytest.raises(ValueError, match="36-cell"):
        classify_formal_report(report)


def test_inconclusive_formal_report_does_not_manufacture_a_verdict() -> None:
    report = _formal_report()
    for benchmark in AUDIT_BENCHMARKS:
        report["post_action_probe"][benchmark]["ensemble"]["paired_vs_answer_now"][
            "lower"
        ] = -0.001
    with pytest.raises(ValueError, match="does not support"):
        render_predictability_audit(report, report_sha256="a" * 64)
