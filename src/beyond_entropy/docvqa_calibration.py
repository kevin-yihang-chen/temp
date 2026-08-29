from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .risk_control import (
    AcquisitionCalibrationRow,
    RiskConstraint,
    calibrate_source_risk_threshold_fixed_sequence,
)


EXPECTED_CONSTRAINTS = [
    {"kind": "induced_harm", "limit": 0.005},
    {"kind": "net_negative_call_mass", "limit": 0.02},
]


def _require(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"DocVQA calibration contract mismatch for {name}")


def validate_frozen_candidate(candidate: Mapping[str, Any]) -> list[float]:
    expected = {
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
    }
    for name, value in expected.items():
        _require(candidate.get(name), value, f"candidate {name}")
    _require(candidate.get("threshold"), None, "uncalibrated threshold")
    raw_thresholds = candidate.get("threshold_grid")
    if not isinstance(raw_thresholds, list) or any(
        not isinstance(value, (int, float)) for value in raw_thresholds
    ):
        raise ValueError("DocVQA candidate threshold grid is invalid")
    thresholds = [float(value) for value in raw_thresholds]
    if (
        not 2 <= len(thresholds) <= 11
        or any(not math.isfinite(value) for value in thresholds)
        or any(left <= right for left, right in zip(thresholds, thresholds[1:]))
    ):
        raise ValueError("DocVQA candidate threshold grid is invalid")
    contract = candidate.get("calibration_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("DocVQA candidate is missing its calibration contract")
    expected_contract = {
        "method": "fixed_sequence_bounded_mean_kl_ltt_v1",
        "threshold_order": "strict_to_permissive_descending",
        "threshold_rate_weighting": "equal_source_then_equal_question",
        "constraints": EXPECTED_CONSTRAINTS,
        "family_error": 0.05,
        "per_step_p_cutoff": 0.025,
        "min_source_call_rate": 0.01,
        "min_source_utility": 0.001,
        "calibration_sources": 2500,
        "formal_sources": 3500,
    }
    for name, value in expected_contract.items():
        _require(contract.get(name), value, f"candidate calibration {name}")
    summaries = contract.get("threshold_summaries")
    if not isinstance(summaries, list) or len(summaries) != len(thresholds):
        raise ValueError("DocVQA threshold summaries do not match the grid")
    summary_thresholds: list[float] = []
    for summary in summaries:
        if not isinstance(summary, Mapping):
            raise ValueError("DocVQA threshold summary must be a mapping")
        value = summary.get("threshold")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("DocVQA threshold summary has an invalid threshold")
        summary_thresholds.append(float(value))
    if summary_thresholds != thresholds:
        raise ValueError("DocVQA threshold summaries changed order or values")
    freeze = candidate.get("candidate_freeze")
    if not isinstance(freeze, Mapping):
        raise ValueError("DocVQA candidate is missing freeze provenance")
    _require(
        freeze.get("ranker_training_outcomes_used"),
        True,
        "ranker outcome disclosure",
    )
    _require(
        freeze.get("calibration_outcomes_used"),
        False,
        "calibration outcome exclusion",
    )
    _require(
        freeze.get("formal_outcomes_used"),
        False,
        "formal outcome exclusion",
    )
    return thresholds


def calibrate_frozen_candidate_rows(
    candidate: Mapping[str, Any],
    rows: Sequence[AcquisitionCalibrationRow],
    *,
    expected_sources: int = 2500,
    run_provenance: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the preregistered DocVQA fixed sequence to labeled source rows."""

    thresholds = validate_frozen_candidate(candidate)
    if expected_sources <= 0:
        raise ValueError("expected calibration source count must be positive")
    materialized_rows = list(rows)
    if not materialized_rows:
        raise ValueError("DocVQA calibration rows must not be empty")
    source_count = len({row.source_id for row in materialized_rows})
    if source_count != expected_sources:
        raise ValueError(
            f"DocVQA calibration requires {expected_sources} source groups"
        )
    calibration = calibrate_source_risk_threshold_fixed_sequence(
        materialized_rows,
        thresholds,
        constraints=[
            RiskConstraint("induced_harm", 0.005),
            RiskConstraint("net_negative_call_mass", 0.02),
        ],
        lambda_cost=0.05,
        max_tool_cost=1.0,
        family_error=0.05,
        min_source_call_rate=0.01,
        min_source_utility=0.001,
    )
    provenance = {} if run_provenance is None else dict(run_provenance)
    provenance["formal_outcomes_used"] = False
    calibration["run"] = provenance

    calibrated_model = dict(candidate)
    calibrated_model["threshold"] = calibration["selected_threshold"]
    calibrated_model["risk_calibration"] = {
        key: calibration[key]
        for key in (
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
    }
    calibrated_model["risk_calibration"]["provenance"] = provenance
    return calibration, calibrated_model
