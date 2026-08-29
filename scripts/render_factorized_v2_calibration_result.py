from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from beyond_entropy.factorized_formal import check_hash, load_mapping


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
    threshold = _finite(candidate.get("threshold"), "candidate threshold")
    call_rate = _finite(candidate.get("source_call_rate"), "source call rate")
    utility = _finite(candidate.get("source_utility"), "source utility")
    risks = _mapping(candidate.get("risks"), "candidate risks")
    _require(set(risks), set(RISK_NAMES), "candidate risk family")
    risk_passes: list[bool] = []
    for expected in EXPECTED_CONSTRAINTS:
        name = str(expected["kind"])
        risk = _mapping(risks[name], f"{name} risk")
        _require(risk.get("limit"), expected["limit"], f"{name} limit")
        p_value = _finite(risk.get("p_value"), f"{name} p-value")
        if not 0.0 <= p_value <= 1.0:
            raise ValueError(
                f"factorized-v2 calibration result has invalid {name} p-value"
            )
        risk_mean = _finite(
            risk.get("source_balanced_mean"), f"{name} source-balanced mean"
        )
        if not 0.0 <= risk_mean <= float(risk.get("upper_bound", 1.0)):
            raise ValueError(f"factorized-v2 calibration result has invalid {name} mean")
        passed = p_value <= adjusted_p_cutoff
        _require(risk.get("passed"), passed, f"{name} pass decision")
        risk_passes.append(passed)
    accepted = all(risk_passes)
    _require(candidate.get("risk_accepted"), accepted, "joint risk decision")
    nondegenerate = accepted and call_rate >= 0.01 and utility >= 0.001
    return accepted, nondegenerate


def validate_calibration_result(
    calibration: Mapping[str, Any],
    model: Mapping[str, Any],
) -> str:
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
        float(expected_selected["threshold"]) if expected_selected is not None else None
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


def _number(value: Any) -> str:
    return f"{float(value):.6f}"


def render_calibration_markdown(
    calibration: Mapping[str, Any],
    model: Mapping[str, Any],
    *,
    calibration_sha256: str,
    model_sha256: str,
) -> str:
    status = validate_calibration_result(calibration, model)
    passed = status == SUCCESS
    disposition = (
        "This independent calibration opens eligibility to freeze the selected policy "
        "and materialize the sealed formal split. It is not a formal scientific success."
        if passed
        else (
            "This branch is closed as a negative independent calibration. The sealed "
            "formal split must not be materialized or evaluated for this candidate."
        )
    )
    selected = calibration.get("selected")
    selected_threshold = calibration.get("selected_threshold")
    lines = [
        "# Factorized-v2 TextVQA independent calibration result",
        "",
        f"Calibration decision: **{'PASS' if passed else 'FAIL'}**.",
        "",
        disposition,
        "",
        "The threshold order, safety constraints, non-degeneracy floors, source "
        "allocation, and reporting template were frozen before calibration output. "
        "No formal outcome was used.",
        "",
        "## Decision summary",
        "",
        f"- Selection status: `{status}`",
        (
            "- Selected threshold: "
            + (
                "none (answer now)"
                if selected_threshold is None
                else f"`{_number(selected_threshold)}`"
            )
        ),
        f"- Tested thresholds: {int(calibration['tested_threshold_count'])} / 11",
        (
            "- Fixed-sequence stopping threshold: "
            + (
                "none"
                if calibration.get("stopping_threshold") is None
                else f"`{_number(calibration['stopping_threshold'])}`"
            )
        ),
        "- Source count / decision count: 3,000 / 4,747",
        "- Safety family error / per-step cutoff: 0.05 / 0.025",
        "- Non-degeneracy floors: source call rate >= 0.01 and source utility >= 0.001",
        "",
    ]
    if isinstance(selected, Mapping):
        lines.extend(
            [
                "## Selected policy",
                "",
                f"- Source-balanced call rate: {_number(selected['source_call_rate'])}",
                f"- Source-balanced utility: {_number(selected['source_utility'])}",
                "",
            ]
        )
    lines.extend(
        [
            "## Frozen threshold sequence",
            "",
            "| Step | Threshold | Source call | Source utility | Harm mean | Harm p "
            "| Negative-call mean | Negative-call p | Risk | Non-degenerate | Selected |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: "
            "| :---: | :---: |",
        ]
    )
    for index, candidate_value in enumerate(calibration["candidates"], start=1):
        candidate = _mapping(candidate_value, "candidate")
        risks = _mapping(candidate["risks"], "candidate risks")
        harm = _mapping(risks["induced_harm"], "induced-harm risk")
        negative = _mapping(risks["net_negative_call_mass"], "negative-call risk")
        nondegenerate = (
            bool(candidate["risk_accepted"])
            and float(candidate["source_call_rate"]) >= 0.01
            and float(candidate["source_utility"]) >= 0.001
        )
        is_selected = candidate == selected
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                index,
                _number(candidate["threshold"]),
                _number(candidate["source_call_rate"]),
                _number(candidate["source_utility"]),
                _number(harm["source_balanced_mean"]),
                _number(harm["p_value"]),
                _number(negative["source_balanced_mean"]),
                _number(negative["p_value"]),
                "pass" if candidate["risk_accepted"] else "fail",
                "yes" if nondegenerate else "no",
                "yes" if is_selected else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Integrity and provenance",
            "",
            f"- Calibration JSON SHA-256: `{calibration_sha256}`",
            f"- Calibrated model SHA-256: `{model_sha256}`",
            f"- Candidate SHA-256: `{calibration['run']['candidate_sha256']}`",
            f"- Allocation SHA-256: `{calibration['run']['allocation_sha256']}`",
            f"- Rollouts SHA-256: `{calibration['run']['rollouts_sha256']}`",
            f"- Rollout audit SHA-256: `{calibration['run']['rollout_audit_sha256']}`",
            f"- Label-free features SHA-256: `{calibration['run']['features_sha256']}`",
            f"- Protocol SHA-256: `{calibration['run']['protocol_sha256']}`",
            f"- Code revision: `{calibration['run']['code_revision']}`",
            "- Formal outcomes used: `false`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and render the frozen factorized-v2 calibration result"
    )
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--expected-calibration-sha256", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calibration_hash = check_hash(
        args.calibration,
        args.expected_calibration_sha256,
        "calibration report",
    )
    model_hash = check_hash(args.model, args.expected_model_sha256, "calibrated model")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite calibration report: {args.output}")
    calibration = load_mapping(args.calibration, "calibration report")
    model = load_mapping(args.model, "calibrated model")
    rendered = render_calibration_markdown(
        calibration,
        model,
        calibration_sha256=calibration_hash,
        model_sha256=model_hash,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
