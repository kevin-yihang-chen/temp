from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from beyond_entropy.docvqa_calibration import (
    SUCCESS,
    validate_docvqa_calibration_artifact_bundle,
)
from beyond_entropy.docvqa_train_allocation import sha256_file


def _load_mapping(path: Path, name: str) -> dict[str, Any]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _check_hash(path: Path, expected: str, name: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"DocVQA calibration artifact is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"DocVQA calibration {name} SHA-256 mismatch")
    return actual


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"DocVQA calibration renderer has invalid {name}")
    return value


def _number(value: Any) -> str:
    return f"{float(value):.6f}"


def render_docvqa_calibration_markdown(
    calibration: Mapping[str, Any],
    model: Mapping[str, Any],
    audit: Mapping[str, Any],
    *,
    calibration_sha256: str,
    model_sha256: str,
    audit_sha256: str,
) -> str:
    status = validate_docvqa_calibration_artifact_bundle(
        calibration,
        model,
        audit,
        calibration_sha256=calibration_sha256,
        model_sha256=model_sha256,
    )
    passed = status == SUCCESS
    disposition = (
        "This independent calibration permits a separate formal-policy freeze. "
        "It is not a formal scientific success, and the formal identities remain sealed."
        if passed
        else (
            "This preregistered branch is closed as a negative calibration. The "
            "formal identities and outcomes must remain unmaterialized."
        )
    )
    selected = calibration.get("selected")
    selected_threshold = calibration.get("selected_threshold")
    thresholds = model["threshold_grid"]
    lines = [
        "# DocVQA-train factorized-v2 independent calibration result",
        "",
        f"Calibration decision: **{'PASS' if passed else 'FAIL'}**.",
        "",
        disposition,
        "",
        "The source allocation, sole candidate, threshold order, safety tests, "
        "non-degeneracy floors, and renderer were frozen before calibration. No "
        "formal outcome was accessed.",
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
        (
            f"- Tested thresholds: {int(calibration['tested_threshold_count'])} "
            f"/ {len(thresholds)}"
        ),
        (
            "- Fixed-sequence stopping threshold: "
            + (
                "none"
                if calibration.get("stopping_threshold") is None
                else f"`{_number(calibration['stopping_threshold'])}`"
            )
        ),
        (
            f"- Source count / decision count: {int(calibration['n_sources']):,} / "
            f"{int(calibration['n_decisions']):,}"
        ),
        "- Safety family error / per-step cutoff: 0.05 / 0.025",
        "- Risk bounds: induced harm <= 0.005; negative-call mass <= 0.02",
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
    for index, raw_candidate in enumerate(calibration["candidates"], start=1):
        candidate = _mapping(raw_candidate, "candidate")
        risks = _mapping(candidate["risks"], "candidate risks")
        harm = _mapping(risks["induced_harm"], "induced-harm risk")
        negative = _mapping(
            risks["net_negative_call_mass"],
            "negative-call risk",
        )
        nondegenerate = (
            bool(candidate["risk_accepted"])
            and float(candidate["source_call_rate"]) >= 0.01
            and float(candidate["source_utility"]) >= 0.001
        )
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
                "yes" if candidate == selected else "no",
            )
        )
    run = _mapping(calibration["run"], "run provenance")
    lines.extend(
        [
            "",
            "## Integrity and provenance",
            "",
            f"- Calibration JSON SHA-256: `{calibration_sha256}`",
            f"- Calibrated model SHA-256: `{model_sha256}`",
            f"- Calibration audit SHA-256: `{audit_sha256}`",
            f"- Candidate SHA-256: `{run['candidate_sha256']}`",
            f"- Candidate audit SHA-256: `{run['candidate_audit_sha256']}`",
            f"- Allocation SHA-256: `{run['allocation_sha256']}`",
            f"- Allocation audit SHA-256: `{run['allocation_audit_sha256']}`",
            f"- Manifest SHA-256: `{run['manifest_sha256']}`",
            f"- Rollouts SHA-256: `{run['rollouts_sha256']}`",
            f"- Rollout audit SHA-256: `{run['rollout_audit_sha256']}`",
            f"- Label-free features SHA-256: `{run['features_sha256']}`",
            f"- Protocol SHA-256: `{run['protocol_sha256']}`",
            f"- Code revision: `{run['code_revision']}`",
            "- Formal outcomes used: `false`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and render the frozen DocVQA calibration result"
    )
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--expected-calibration-sha256", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--expected-audit-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calibration_sha256 = _check_hash(
        args.calibration,
        args.expected_calibration_sha256,
        "report",
    )
    model_sha256 = _check_hash(args.model, args.expected_model_sha256, "model")
    audit_sha256 = _check_hash(args.audit, args.expected_audit_sha256, "audit")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite DocVQA report: {args.output}")
    rendered = render_docvqa_calibration_markdown(
        _load_mapping(args.calibration, "calibration report"),
        _load_mapping(args.model, "calibrated model"),
        _load_mapping(args.audit, "calibration audit"),
        calibration_sha256=calibration_sha256,
        model_sha256=model_sha256,
        audit_sha256=audit_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(rendered)
    print(args.output)


if __name__ == "__main__":
    main()
