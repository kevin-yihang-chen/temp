from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value: Any) -> str:
    return f"{float(value):.6f}"


def render_scaled_formal_markdown(
    evaluation: Mapping[str, Any],
    calibration: Mapping[str, Any],
    *,
    evaluation_sha256: str,
    calibration_sha256: str,
    policy_freeze_sha256: str,
) -> str:
    source = evaluation["source_balanced"]
    question = evaluation["question_weighted"]
    interval = evaluation["source_bootstrap"]["metrics"]["utility"]
    selection = evaluation["selection"]
    ranking = evaluation["ranking"]
    selected = calibration["selected"]
    passed = bool(evaluation["passed"])
    verdict = "PASS" if passed else "FAIL"
    interpretation = (
        "The frozen policy passes every preregistered one-shot criterion. This "
        "supports positive in-family cost-sensitive visual-acquisition utility on "
        "the reserved TextVQA train-source population."
        if passed
        else "The frozen policy fails at least one preregistered one-shot criterion. "
        "The result is retained as a negative confirmation and this formal bank "
        "must not be reused to select a replacement policy."
    )
    lines = [
        "# Scaled TextVQA risk-controlled formal result",
        "",
        f"**Preregistered verdict: {verdict}.**",
        "",
        "> One-shot evaluation of the exact policy selected on an independent "
        "3,000-source risk-calibration bank. No formal outcome tuned the model, "
        "feature contract, crop ranker, call-value head, threshold, or evaluator.",
        "",
        "## Primary result",
        "",
        "| Criterion | Formal value | Passed |",
        "|---|---:|:---:|",
        (
            "| Source-balanced utility | "
            f"{_number(source['utility'])} | "
            f"{'yes' if evaluation['pass_rule']['source_utility_positive'] else 'no'} |"
        ),
        (
            "| Two-sided 97.5% whole-source bootstrap interval | "
            f"[{_number(interval['ci_low'])}, {_number(interval['ci_high'])}] | "
            f"{'yes' if evaluation['pass_rule']['source_utility_97_5pct_ci_low_positive'] else 'no'} |"
        ),
        (
            "| Question-weighted utility | "
            f"{_number(question['utility'])} | "
            f"{'yes' if evaluation['pass_rule']['question_weighted_utility_positive'] else 'no'} |"
        ),
        (
            "| Source-balanced call rate | "
            f"{_number(source['call'])} | "
            f"{'yes' if evaluation['pass_rule']['source_call_rate_at_least_0_01'] else 'no'} |"
        ),
        "",
        f"The evaluation contains {evaluation['n_decisions']:,} questions from "
        f"{evaluation['n_sources']:,} source images. The interval uses "
        f"{evaluation['source_bootstrap']['n_resamples']:,} iid whole-source "
        "bootstrap resamples.",
        "",
        "## Frozen calibration decision",
        "",
        f"The selected threshold is `{_number(evaluation['threshold'])}`. On the "
        "independent calibration bank it had source-balanced call rate "
        f"`{_number(selected['source_call_rate'])}` and utility "
        f"`{_number(selected['source_utility'])}`.",
        "",
        "| Calibration risk | Mean | Limit | Accepted |",
        "|---|---:|---:|:---:|",
    ]
    for name in ("induced_harm", "net_negative_call_mass"):
        risk = selected["risks"][name]
        lines.append(
            f"| `{name}` | {_number(risk['source_balanced_mean'])} | "
            f"{_number(risk['limit'])} | {'yes' if risk['passed'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "These finite-sample risk tests apply to the calibration population. "
            "Their transfer to the formal population is evaluated empirically and "
            "is not claimed as a guarantee under arbitrary distribution shift.",
            "",
            "## Mandatory formal diagnostics",
            "",
            "| Diagnostic | Source-balanced | Question-weighted |",
            "|---|---:|---:|",
        ]
    )
    for name in (
        "gain",
        "induced_harm",
        "net_negative_call",
        "negative_net_value",
        "oracle_utility",
        "random_utility",
        "entropy_search_utility",
    ):
        lines.append(
            f"| `{name}` | {_number(source[name])} | {_number(question[name])} |"
        )
    lines.extend(
        [
            f"| `oracle_regret` | {_number(evaluation['oracle_regret'])} | -- |",
            "",
            "| Selection/ranking diagnostic | Value |",
            "|---|---:|",
            f"| Positive-utility precision among calls | {_number(selection['positive_utility_call_precision'])} |",
            f"| Unnecessary-call rate | {_number(selection['unnecessary_call_rate'])} |",
            f"| Correct stopping rate | {_number(selection['correct_stopping_rate'])} |",
            f"| Source-balanced raw gain per call | {_number(selection['source_balanced_raw_gain_per_call'])} |",
            f"| Top-1 rescue within helpful states | {_number(ranking['top1_rescue_rate_within_helpful_states'])} |",
            f"| Random rescue within helpful states | {_number(ranking['random_rescue_rate_within_helpful_states'])} |",
            "",
            "Fixed-crop source-balanced utilities:",
            "",
        ]
    )
    for action_id, utility in sorted(ranking["fixed_crop_source_utilities"].items()):
        lines.append(f"- `{action_id}`: `{_number(utility)}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            interpretation,
            "",
            "Earlier TextVQA and DocVQA formal failures remain part of the record; "
            "this scaled experiment neither replaces nor erases them.",
            "",
            "## Integrity",
            "",
            f"- Formal evaluation SHA-256: `{evaluation_sha256}`",
            f"- Calibration report SHA-256: `{calibration_sha256}`",
            f"- Policy freeze SHA-256: `{policy_freeze_sha256}`",
            f"- Frozen model SHA-256: `{evaluation['run']['model_sha256']}`",
            f"- Formal manifest SHA-256: `{evaluation['run']['manifest_sha256']}`",
            f"- Formal rollouts SHA-256: `{evaluation['run']['rollouts_sha256']}`",
            f"- Label-free features SHA-256: `{evaluation['run']['features_sha256']}`",
            f"- Protocol SHA-256: `{evaluation['run']['protocol_sha256']}`",
            f"- Evaluator module SHA-256: `{evaluation['run']['evaluator_module_sha256']}`",
            f"- Evaluator script SHA-256: `{evaluation['run']['evaluator_script_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the frozen scaled TextVQA evaluation without recomputing metrics"
    )
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--expected-evaluation-sha256", required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--expected-calibration-sha256", required=True)
    parser.add_argument("--policy-freeze", type=Path, required=True)
    parser.add_argument("--expected-policy-freeze-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path, expected, name in (
        (args.evaluation, args.expected_evaluation_sha256, "evaluation"),
        (args.calibration, args.expected_calibration_sha256, "calibration"),
        (args.policy_freeze, args.expected_policy_freeze_sha256, "policy freeze"),
    ):
        if _sha256(path) != expected:
            raise ValueError(f"{name} SHA-256 mismatch")
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    rendered = render_scaled_formal_markdown(
        evaluation,
        calibration,
        evaluation_sha256=args.expected_evaluation_sha256,
        calibration_sha256=args.expected_calibration_sha256,
        policy_freeze_sha256=args.expected_policy_freeze_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
