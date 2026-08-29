from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from beyond_entropy.factorized_formal import check_hash, load_mapping


def _number(value: Any) -> str:
    return f"{float(value):.6f}"


def render_report(report: Mapping[str, Any]) -> str:
    source = report["source_balanced"]
    question = report["question_weighted"]
    bootstrap = report["source_bootstrap"]["metrics"]["utility"]
    selection = report["selection"]
    ranking = report["ranking"]
    risk = report["risk_diagnostics"]
    pass_rule = report["pass_rule"]
    status = "PASS" if report["passed"] else "FAIL"
    disposition = (
        "The frozen branch passes its one-shot formal decision."
        if report["passed"]
        else (
            "The frozen branch fails its one-shot formal decision and is retained "
            "as a negative confirmation; this formal bank must not select a replacement."
        )
    )
    lines = [
        "# Factorized-v2 TextVQA one-shot formal result",
        "",
        f"Decision: **{status}**.",
        "",
        disposition,
        "",
        "The model, threshold, risk family, evaluator, source allocation, and "
        "implementation were frozen before this 5,953-source manifest was exported. "
        "No formal outcome tuned the policy.",
        "",
        "## Primary result",
        "",
        "| Metric | Estimate | 97.5% interval |",
        "| --- | ---: | ---: |",
        (
            "| Source-balanced utility | {} | [{}, {}] |".format(
                _number(source["utility"]),
                _number(bootstrap["ci_low"]),
                _number(bootstrap["ci_high"]),
            )
        ),
        f"| Question-weighted utility | {_number(question['utility'])} | -- |",
        f"| Source-balanced call rate | {_number(source['call'])} | -- |",
        f"| Accuracy gain | {_number(source['gain'])} | -- |",
        "",
        "## Frozen pass rule",
        "",
    ]
    for name, passed in pass_rule.items():
        lines.append(f"- `{name}`: {'pass' if passed else 'fail'}")
    lines.extend(
        [
            "",
            "## Safety and selection diagnostics",
            "",
            f"- Induced-harm mass: {_number(risk['source_balanced_induced_harm_mass'])}",
            (
                "- Net-negative-call mass: "
                f"{_number(risk['source_balanced_net_negative_call_mass'])}"
            ),
            f"- Unnecessary-call rate: {_number(selection['unnecessary_call_rate'])}",
            (
                "- Positive-utility precision among calls: "
                f"{_number(selection['positive_utility_call_precision'])}"
            ),
            f"- Correct stopping rate: {_number(selection['correct_stopping_rate'])}",
            f"- Oracle regret: {_number(report['oracle_regret'])}",
            "",
            "## Crop-ranking diagnostics",
            "",
            (
                "- Learned top-1 rescue rate within helpful states: "
                f"{_number(ranking['top1_rescue_rate_within_helpful_states'])}"
            ),
            (
                "- Random rescue rate within helpful states: "
                f"{_number(ranking['random_rescue_rate_within_helpful_states'])}"
            ),
            "",
            "Random, fixed-crop, same-gate, post-action entropy, and oracle utilities "
            "are retained in the JSON report. The post-action entropy comparator is "
            "diagnostic rather than deployable.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the frozen factorized-v2 formal JSON result"
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check_hash(args.report, args.expected_report_sha256, "formal report")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite rendered report: {args.output}")
    report = load_mapping(args.report, "formal report")
    required = {
        "passed",
        "source_balanced",
        "question_weighted",
        "source_bootstrap",
        "selection",
        "ranking",
        "risk_diagnostics",
        "pass_rule",
        "oracle_regret",
        "run",
    }
    if not required.issubset(report):
        raise ValueError("formal report is missing mandatory diagnostics")
    run = report["run"]
    if not isinstance(run, Mapping) or (
        run.get("formal_outcomes_used") is not True
        or run.get("no_target_derived_tuning") is not True
    ):
        raise ValueError("formal report provenance is invalid")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(report), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
