from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


EXPECTED_CODE_REVISION = "d85c8d57db2b0c663f760e1fc43a0a9920297422"
EXPECTED_CANDIDATE_SHA256 = (
    "9a6c9d032ebdbc271b7d3c829fbb3d6ff167cac01b54ce75adc8da86e3063342"
)
EXPECTED_ALLOCATION_SHA256 = (
    "bc0ecb4b6f49a5b0e92b90b4c30620f72246722370d59c8078753d5846f5e9b6"
)
EXPECTED_ALLOCATION_AUDIT_SHA256 = (
    "f01f853a7de7774466be55c012b7e174f57f4ac120ed58a0bf3984e71252b5c3"
)
EXPECTED_MANIFEST_SHA256 = (
    "0db79580d7bb96794901703a6ec0bfc0ae14e31159ddde5664762aa0351b323a"
)
EXPECTED_PROTOCOL_SHA256 = (
    "babf01d4090263d1cfcb28c42f86f7b13ae9de4bb6bab0ca10d6e4707f02e2ca"
)
EXPECTED_CONSTRAINTS = [
    {"kind": "induced_harm", "limit": 0.005},
    {"kind": "net_negative_call_mass", "limit": 0.02},
]
RISK_NAMES = tuple(item["kind"] for item in EXPECTED_CONSTRAINTS)
SUCCESS = "selected_non_degenerate_safe_threshold"
FAILURE = "no_non_degenerate_safe_threshold"
RISK_KEYS = (
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


def _require(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"factorized-v2 calibration result mismatch for {name}")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"factorized-v2 calibration result has invalid {name}")
    return value


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"factorized-v2 calibration result has non-finite {name}")
    return number


def _sha256(value: Any, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"factorized-v2 calibration result has invalid {name}")
    return digest


def _validate_candidate(
    candidate: Mapping[str, Any],
    *,
    adjusted_p_cutoff: float,
) -> tuple[bool, bool]:
    _finite(candidate.get("threshold"), "candidate threshold")
    call_rate = _finite(candidate.get("source_call_rate"), "source call rate")
    utility = _finite(candidate.get("source_utility"), "source utility")
    if not 0.0 <= call_rate <= 1.0:
        raise ValueError("factorized-v2 calibration result has invalid call rate")
    risks = _mapping(candidate.get("risks"), "candidate risks")
    _require(set(risks), set(RISK_NAMES), "candidate risk family")
    risk_passes: list[bool] = []
    for expected in EXPECTED_CONSTRAINTS:
        name = str(expected["kind"])
        risk = _mapping(risks[name], f"{name} risk")
        _require(risk.get("limit"), expected["limit"], f"{name} limit")
        upper_bound = _finite(risk.get("upper_bound"), f"{name} upper bound")
        if upper_bound <= 0.0:
            raise ValueError(
                f"factorized-v2 calibration result has invalid {name} upper bound"
            )
        p_value = _finite(risk.get("p_value"), f"{name} p-value")
        if not 0.0 <= p_value <= 1.0:
            raise ValueError(
                f"factorized-v2 calibration result has invalid {name} p-value"
            )
        risk_mean = _finite(
            risk.get("source_balanced_mean"), f"{name} source-balanced mean"
        )
        if not 0.0 <= risk_mean <= upper_bound:
            raise ValueError(f"factorized-v2 calibration result has invalid {name} mean")
        passed = p_value <= adjusted_p_cutoff
        _require(risk.get("passed"), passed, f"{name} pass decision")
        risk_passes.append(passed)
    accepted = all(risk_passes)
    _require(candidate.get("risk_accepted"), accepted, "joint risk decision")
    nondegenerate = accepted and call_rate >= 0.01 and utility >= 0.001
    return accepted, nondegenerate


def validate_factorized_v2_calibration_result(
    calibration: Mapping[str, Any],
    model: Mapping[str, Any],
) -> str:
    """Recompute the frozen calibration decision and model embedding contract."""

    expected_scalars = {
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
        "constraints": EXPECTED_CONSTRAINTS,
        "min_source_call_rate": 0.01,
        "min_source_utility": 0.001,
        "selection_objective": "most_permissive_pre_failure_with_non_degeneracy",
    }
    for name, expected in expected_scalars.items():
        _require(calibration.get(name), expected, name)
    status = str(calibration.get("selection_status"))
    if status not in {SUCCESS, FAILURE}:
        raise ValueError("factorized-v2 calibration result has invalid selection status")

    run = _mapping(calibration.get("run"), "run provenance")
    expected_run = {
        "code_revision": EXPECTED_CODE_REVISION,
        "candidate_sha256": EXPECTED_CANDIDATE_SHA256,
        "allocation_sha256": EXPECTED_ALLOCATION_SHA256,
        "allocation_audit_sha256": EXPECTED_ALLOCATION_AUDIT_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "formal_outcomes_used": False,
    }
    for name, expected in expected_run.items():
        _require(run.get(name), expected, f"run {name}")
    for name in ("rollouts_sha256", "rollout_audit_sha256", "features_sha256"):
        _sha256(run.get(name), f"run {name}")

    candidates_value = calibration.get("candidates")
    if not isinstance(candidates_value, Sequence) or isinstance(
        candidates_value, (str, bytes)
    ):
        raise ValueError("factorized-v2 calibration result has invalid candidates")
    candidates = [_mapping(value, "candidate") for value in candidates_value]
    _require(
        calibration.get("tested_threshold_count"),
        len(candidates),
        "tested threshold count",
    )
    if not candidates:
        raise ValueError("factorized-v2 calibration result tested no thresholds")
    untested_value = calibration.get("untested_thresholds")
    if not isinstance(untested_value, list):
        raise ValueError("factorized-v2 calibration result has invalid untested thresholds")
    untested = [_finite(value, "untested threshold") for value in untested_value]
    tested_thresholds = [
        _finite(candidate.get("threshold"), "tested threshold")
        for candidate in candidates
    ]
    threshold_grid = [
        _finite(value, "model threshold grid")
        for value in model.get("threshold_grid", [])
    ]
    _require(tested_thresholds + untested, threshold_grid, "frozen threshold sequence")
    if len(threshold_grid) != 11 or any(
        left <= right for left, right in zip(threshold_grid, threshold_grid[1:])
    ):
        raise ValueError("factorized-v2 model threshold grid is not frozen descending")

    decisions = [
        _validate_candidate(candidate, adjusted_p_cutoff=0.025)
        for candidate in candidates
    ]
    first_failure = next(
        (index for index, (accepted, _) in enumerate(decisions) if not accepted), None
    )
    if first_failure is None:
        _require(calibration.get("stopping_threshold"), None, "stopping threshold")
        _require(untested, [], "thresholds after complete fixed sequence")
    else:
        _require(first_failure, len(candidates) - 1, "fixed-sequence stopping index")
        _require(
            calibration.get("stopping_threshold"),
            tested_thresholds[-1],
            "stopping threshold",
        )
        _require(
            untested,
            threshold_grid[len(candidates) :],
            "untested thresholds after stopping",
        )

    eligible = [
        candidate
        for candidate, (_, nondegenerate) in zip(candidates, decisions)
        if nondegenerate
    ]
    expected_selected = eligible[-1] if eligible else None
    expected_status = SUCCESS if expected_selected is not None else FAILURE
    expected_threshold = (
        float(expected_selected["threshold"])
        if expected_selected is not None
        else None
    )
    _require(status, expected_status, "recomputed selection status")
    _require(calibration.get("selected"), expected_selected, "selected candidate")
    _require(
        calibration.get("selected_threshold"), expected_threshold, "selected threshold"
    )

    answer_now = _mapping(calibration.get("answer_now"), "answer-now baseline")
    _require(answer_now.get("threshold"), None, "answer-now threshold")
    _require(answer_now.get("answer_now_only"), True, "answer-now marker")
    _require(answer_now.get("source_call_rate"), 0.0, "answer-now call rate")
    _require(answer_now.get("source_utility"), 0.0, "answer-now utility")

    _require(model.get("threshold"), expected_threshold, "model threshold")
    risk = _mapping(model.get("risk_calibration"), "model risk calibration")
    expected_risk = {name: calibration[name] for name in RISK_KEYS}
    expected_risk["provenance"] = calibration["run"]
    _require(dict(risk), expected_risk, "embedded model calibration")
    return status
