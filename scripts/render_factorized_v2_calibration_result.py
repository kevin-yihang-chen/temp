from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from beyond_entropy.factorized_calibration_contract import (
    SUCCESS,
    validate_factorized_v2_calibration_result,
)
from beyond_entropy.factorized_formal import check_hash, load_mapping


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"factorized-v2 calibration result has invalid {name}")
    return value


def _number(value: Any) -> str:
    return f"{float(value):.6f}"


def render_calibration_markdown(
    calibration: Mapping[str, Any],
    model: Mapping[str, Any],
    *,
    calibration_sha256: str,
    model_sha256: str,
) -> str:
    status = validate_factorized_v2_calibration_result(calibration, model)
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
