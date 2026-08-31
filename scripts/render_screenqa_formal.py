#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value: Any, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def render(report: Mapping[str, Any]) -> str:
    source = report["source_balanced"]
    question = report["question_weighted"]
    interval = report["source_bootstrap"]["metrics"]["utility"]
    baselines = report["baselines"]
    pass_rule = report["pass_rule"]
    outcome = "PASS" if report.get("passed") is True else "FAIL"
    rows = [
        ("Frozen policy", source["utility"]),
        ("No call", 0.0),
        ("Random crop, always call", source["random_always_call_utility"]),
        (
            "Post-action entropy crop, always call (idealized)",
            source["post_action_entropy_always_call_utility"],
        ),
        (
            "Exhaustive UG entropy, four calls charged",
            source["ug_style_exhaustive_entropy_utility"],
        ),
        (
            "Matched-budget entropy gate, learned crop",
            baselines["matched_budget_entropy_gate_source_utility_learned_crop"],
        ),
        (
            "Matched-budget entropy gate, random crop",
            baselines["matched_budget_entropy_gate_source_utility_random_crop"],
        ),
        (
            "Matched-budget random gate, random crop (expected)",
            baselines[
                "matched_budget_random_gate_source_utility_random_crop_expected"
            ],
        ),
    ]
    lines = [
        "# ScreenQA one-shot formal result",
        "",
        f"Registered decision: **{outcome}**.",
        "",
        "The report evaluates the exact independently calibrated threshold once; "
        "formal outcomes were not used for tuning.",
        "",
        "## Primary result",
        "",
        f"- Source-balanced utility: {_number(source['utility'])}",
        (
            "- Registered 97.5% whole-source interval: "
            f"[{_number(interval['ci_low'])}, {_number(interval['ci_high'])}]"
        ),
        f"- Question-weighted utility: {_number(question['utility'])}",
        f"- Source-balanced call rate: {_number(source['call'])}",
        f"- Threshold: {_number(report['threshold'])}",
        f"- Sources / decisions: {report['n_sources']} / {report['n_decisions']}",
        "",
        "## Registered pass clauses",
        "",
        "| Clause | Passed |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {name.replace('_', ' ')} | {str(bool(value)).lower()} |"
        for name, value in pass_rule.items()
    )
    lines.extend(
        [
            "",
            "## Cost-faithful source-balanced utility controls",
            "",
            "| Policy | Utility |",
            "|---|---:|",
        ]
    )
    lines.extend(f"| {name} | {_number(value)} |" for name, value in rows)
    fixed = baselines["fixed_crop_source_utility_always_call"]
    lines.extend(
        [
            "",
            "Fixed-crop always-call utilities: "
            + ", ".join(
                f"{action_id}={_number(value)}"
                for action_id, value in sorted(fixed.items())
            )
            + ".",
            "",
            "Post-action entropy selection is explicitly idealized because it "
            "observes acquired outcomes; exhaustive UG is charged for all four "
            "candidate executions. These secondary controls do not alter the "
            "registered formal PASS/FAIL decision.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the ScreenQA formal report")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.report) != args.expected_report_sha256:
        raise ValueError("ScreenQA formal report SHA-256 mismatch")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite rendered formal report: {args.output}")
    payload = json.loads(args.report.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ScreenQA formal report must be a JSON object")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(render(payload))
    print(args.output)


if __name__ == "__main__":
    main()
