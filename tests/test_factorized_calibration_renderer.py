from __future__ import annotations

from copy import deepcopy

import pytest

from beyond_entropy.factorized_calibration_contract import (
    FAILURE,
    SUCCESS,
    validate_factorized_v2_calibration_result,
)
from scripts.render_factorized_v2_calibration_result import render_calibration_markdown
from scripts.freeze_factorized_v2_formal_policy import (
    _validate_successful_calibration,
)


def _candidate(threshold, *, call, utility, harm_p=0.001, negative_p=0.001):
    risks = {
        "induced_harm": {
            "limit": 0.005,
            "upper_bound": 1.0,
            "source_balanced_mean": 0.001,
            "p_value": harm_p,
            "passed": harm_p <= 0.025,
        },
        "net_negative_call_mass": {
            "limit": 0.02,
            "upper_bound": 1.0,
            "source_balanced_mean": 0.01,
            "p_value": negative_p,
            "passed": negative_p <= 0.025,
        },
    }
    return {
        "threshold": threshold,
        "source_call_rate": call,
        "source_utility": utility,
        "risks": risks,
        "risk_accepted": all(risk["passed"] for risk in risks.values()),
    }


def _payload(*, success):
    grid = [1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    first = _candidate(1.1, call=0.0, utility=0.0)
    second = _candidate(1.0, call=0.02, utility=0.002)
    if success:
        stopped = _candidate(0.9, call=0.04, utility=0.003, harm_p=0.5)
        candidates = [first, second, stopped]
        untested = grid[3:]
        selected = second
        stopping = 0.9
    else:
        failed = _candidate(
            0.9,
            call=0.03,
            utility=0.0005,
            harm_p=0.5,
        )
        candidates = [first, failed]
        selected = None
        stopping = 0.9
        grid = [1.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
        untested = grid[2:]
    status = SUCCESS if success else FAILURE
    threshold = selected["threshold"] if selected is not None else None
    run = {
        "code_revision": "d85c8d57db2b0c663f760e1fc43a0a9920297422",
        "candidate_sha256": "9a6c9d032ebdbc271b7d3c829fbb3d6ff167cac01b54ce75adc8da86e3063342",
        "allocation_sha256": "bc0ecb4b6f49a5b0e92b90b4c30620f72246722370d59c8078753d5846f5e9b6",
        "allocation_audit_sha256": "f01f853a7de7774466be55c012b7e174f57f4ac120ed58a0bf3984e71252b5c3",
        "manifest_sha256": "0db79580d7bb96794901703a6ec0bfc0ae14e31159ddde5664762aa0351b323a",
        "protocol_sha256": "babf01d4090263d1cfcb28c42f86f7b13ae9de4bb6bab0ca10d6e4707f02e2ca",
        "rollouts_sha256": "1" * 64,
        "rollout_audit_sha256": "2" * 64,
        "features_sha256": "3" * 64,
        "formal_outcomes_used": False,
    }
    calibration = {
        "scientific_status": (
            "source-level fixed-sequence risk calibration; nested thresholds "
            "are frozen before calibration outcomes"
        ),
        "method": "fixed_sequence_bounded_mean_kl_ltt_v1",
        "threshold_order": "strict_to_permissive_descending",
        "lambda_cost": 0.05,
        "max_tool_cost": 1.0,
        "family_error": 0.05,
        "per_step_hypothesis_count": 2,
        "adjusted_p_cutoff": 0.025,
        "n_sources": 3000,
        "n_decisions": 4747,
        "constraints": [
            {"kind": "induced_harm", "limit": 0.005},
            {"kind": "net_negative_call_mass", "limit": 0.02},
        ],
        "min_source_call_rate": 0.01,
        "min_source_utility": 0.001,
        "selection_objective": "most_permissive_pre_failure_with_non_degeneracy",
        "selection_status": status,
        "selected_threshold": threshold,
        "selected": selected,
        "answer_now": {
            "threshold": None,
            "answer_now_only": True,
            "source_call_rate": 0.0,
            "source_utility": 0.0,
        },
        "tested_threshold_count": len(candidates),
        "stopping_threshold": stopping,
        "candidates": candidates,
        "untested_thresholds": untested,
        "run": run,
    }
    risk_keys = (
        "selection_status",
        "selected_threshold",
        "method",
        "threshold_order",
        "constraints",
        "family_error",
        "per_step_hypothesis_count",
        "adjusted_p_cutoff",
        "min_source_call_rate",
        "min_source_utility",
        "selection_objective",
        "tested_threshold_count",
        "stopping_threshold",
        "untested_thresholds",
    )
    model = {
        "threshold": threshold,
        "threshold_grid": grid,
        "risk_calibration": {key: calibration[key] for key in risk_keys},
    }
    model["risk_calibration"]["provenance"] = run
    return calibration, model


def test_renderer_uses_same_frozen_template_for_success():
    calibration, model = _payload(success=True)
    rendered = render_calibration_markdown(
        calibration,
        model,
        calibration_sha256="a" * 64,
        model_sha256="b" * 64,
    )
    assert validate_factorized_v2_calibration_result(calibration, model) == SUCCESS
    assert "Calibration decision: **PASS**" in rendered
    assert "not a formal scientific success" in rendered
    assert "| 2 | 1.000000" in rendered


def test_renderer_closes_failed_branch_without_opening_formal():
    calibration, model = _payload(success=False)
    rendered = render_calibration_markdown(
        calibration,
        model,
        calibration_sha256="a" * 64,
        model_sha256="b" * 64,
    )
    assert validate_factorized_v2_calibration_result(calibration, model) == FAILURE
    assert "Calibration decision: **FAIL**" in rendered
    assert "formal split must not be materialized" in rendered
    assert "Fixed-sequence stopping threshold: `0.900000`" in rendered


def test_renderer_rejects_model_threshold_mismatch():
    calibration, model = _payload(success=True)
    model["threshold"] = 0.9
    with pytest.raises(ValueError, match="model threshold"):
        validate_factorized_v2_calibration_result(calibration, model)


def test_renderer_rejects_relabelled_risk_decision():
    calibration, model = _payload(success=False)
    corrupted = deepcopy(calibration)
    corrupted["candidates"][-1]["risks"]["induced_harm"]["passed"] = True
    with pytest.raises(ValueError, match="induced_harm pass decision"):
        validate_factorized_v2_calibration_result(corrupted, model)


def test_policy_freeze_recomputes_calibration_instead_of_trusting_status():
    calibration, model = _payload(success=True)
    candidate = {
        "threshold": None,
        "threshold_grid": model["threshold_grid"],
    }
    assert (
        _validate_successful_calibration(
            candidate=candidate,
            calibration=calibration,
            model=model,
        )
        == 1.0
    )
    calibration["candidates"][1]["risks"]["induced_harm"]["p_value"] = 0.5
    with pytest.raises(ValueError, match="induced_harm pass decision"):
        _validate_successful_calibration(
            candidate=candidate,
            calibration=calibration,
            model=model,
        )
