from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from beyond_entropy.docvqa_formal import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    FORMAL_SOURCES,
    check_hash,
)


PASS_RULE_KEYS = frozenset(
    {
        "source_utility_positive",
        "source_utility_97_5pct_ci_low_positive",
        "question_weighted_utility_positive",
        "source_call_rate_at_least_0_01",
        "threshold_matches_calibration_choice",
        "all_frozen_hashes_and_identity_audits_match",
    }
)
MANDATORY_BASELINES = frozenset(
    {
        "ug_style_exhaustive_candidate_count",
        "ug_style_exhaustive_search_charged_all_candidate_costs",
        "ug_style_exhaustive_entropy_source_gain",
        "ug_style_exhaustive_entropy_source_utility",
        "matched_budget_call_count",
        "matched_budget_entropy_gate_source_utility_learned_crop",
        "matched_budget_entropy_gate_source_utility_random_crop",
        "matched_budget_random_gate_source_utility_random_crop_expected",
        "fixed_crop_source_utility_entropy_gate",
        "fixed_crop_source_utility_always_call",
        "fixed_crop_source_utility_same_gate",
    }
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"DocVQA formal report has invalid {name}")
    return value


def _number(value: Any) -> str:
    return f"{float(value):.6f}"


def validate_formal_report(report: Mapping[str, Any]) -> None:
    required = {
        "passed",
        "threshold",
        "lambda_cost",
        "n_sources",
        "n_decisions",
        "source_balanced",
        "question_weighted",
        "source_bootstrap",
        "risk_diagnostics",
        "ranking",
        "baselines",
        "selection",
        "oracle_regret",
        "pass_rule",
        "run",
    }
    missing = required.difference(report)
    if missing:
        raise ValueError(
            "DocVQA formal report lacks: " + ", ".join(sorted(missing))
        )
    if report.get("n_sources") != FORMAL_SOURCES:
        raise ValueError("DocVQA formal report source population changed")
    if float(report.get("lambda_cost", -1.0)) != 0.05:
        raise ValueError("DocVQA formal report cost changed")
    pass_rule = _mapping(report["pass_rule"], "pass rule")
    if set(pass_rule) != PASS_RULE_KEYS:
        raise ValueError("DocVQA formal report pass rule changed")
    if bool(report["passed"]) != all(bool(value) for value in pass_rule.values()):
        raise ValueError("DocVQA formal report decision differs from pass rule")
    bootstrap = _mapping(report["source_bootstrap"], "source bootstrap")
    if (
        bootstrap.get("n_resamples") != BOOTSTRAP_RESAMPLES
        or float(bootstrap.get("confidence_level", -1.0)) != BOOTSTRAP_CONFIDENCE
        or bootstrap.get("seed") != BOOTSTRAP_SEED
    ):
        raise ValueError("DocVQA formal bootstrap contract changed")
    metrics = _mapping(bootstrap.get("metrics"), "bootstrap metrics")
    utility_interval = _mapping(metrics.get("utility"), "utility interval")
    for name in ("ci_low", "ci_high"):
        float(utility_interval[name])
    baselines = _mapping(report["baselines"], "baselines")
    missing_baselines = MANDATORY_BASELINES.difference(baselines)
    if missing_baselines:
        raise ValueError(
            "DocVQA formal report lacks baselines: "
            + ", ".join(sorted(missing_baselines))
        )
    if (
        baselines.get("ug_style_exhaustive_candidate_count") != 4
        or baselines.get("ug_style_exhaustive_search_charged_all_candidate_costs")
        is not True
    ):
        raise ValueError("DocVQA exhaustive entropy accounting changed")
    run = _mapping(report["run"], "run provenance")
    expected_run = {
        "formal_outcomes_used": True,
        "no_target_derived_tuning": True,
        "feature_outcomes_included": False,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_confidence": BOOTSTRAP_CONFIDENCE,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    for name, expected in expected_run.items():
        if run.get(name) != expected:
            raise ValueError(f"DocVQA formal run provenance changed for {name}")
    for name in (
        "policy_freeze_sha256",
        "model_sha256",
        "manifest_sha256",
        "manifest_provenance_sha256",
        "formal_audit_sha256",
        "rollouts_sha256",
        "rollout_audit_sha256",
        "features_sha256",
    ):
        value = str(run.get(name, ""))
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"DocVQA formal run has invalid {name}")


def render_report(report: Mapping[str, Any]) -> str:
    validate_formal_report(report)
    source = _mapping(report["source_balanced"], "source metrics")
    question = _mapping(report["question_weighted"], "question metrics")
    bootstrap = _mapping(
        _mapping(report["source_bootstrap"], "bootstrap")["metrics"],
        "bootstrap metrics",
    )
    utility_interval = _mapping(bootstrap["utility"], "utility interval")
    selection = _mapping(report["selection"], "selection diagnostics")
    ranking = _mapping(report["ranking"], "ranking diagnostics")
    risk = _mapping(report["risk_diagnostics"], "risk diagnostics")
    baselines = _mapping(report["baselines"], "baselines")
    pass_rule = _mapping(report["pass_rule"], "pass rule")
    run = _mapping(report["run"], "run provenance")
    passed = bool(report["passed"])
    disposition = (
        "The frozen DocVQA branch passes its one-shot benchmark-specific decision."
        if passed
        else (
            "The frozen DocVQA branch fails its one-shot decision and is retained "
            "without selecting any replacement on this formal bank."
        )
    )
    lines = [
        "# DocVQA-train factorized-v2 one-shot formal result",
        "",
        f"Decision: **{'PASS' if passed else 'FAIL'}**.",
        "",
        disposition,
        "",
        "This result is benchmark-specific. A cross-benchmark claim still requires "
        "a separate successful prospective TextVQA decision and is not implied here.",
        "",
        "The model, calibrated threshold, source allocation, evaluator, bootstrap, "
        "baselines, and implementation were frozen before formal target export. No "
        "formal outcome tuned the policy.",
        "",
        "## Primary result",
        "",
        "| Metric | Estimate | 97.5% source interval |",
        "| --- | ---: | ---: |",
        (
            "| Source-balanced utility | {} | [{}, {}] |".format(
                _number(source["utility"]),
                _number(utility_interval["ci_low"]),
                _number(utility_interval["ci_high"]),
            )
        ),
        f"| Question-weighted utility | {_number(question['utility'])} | -- |",
        f"| Source-balanced call rate | {_number(source['call'])} | -- |",
        f"| Source-balanced raw ANLS gain | {_number(source['gain'])} | -- |",
        f"| Baseline ANLS | {_number(source['baseline_accuracy'])} | -- |",
        f"| Policy ANLS | {_number(source['policy_accuracy'])} | -- |",
        "",
        "## Frozen pass rule",
        "",
    ]
    for name in sorted(pass_rule):
        lines.append(f"- `{name}`: {'pass' if pass_rule[name] else 'fail'}")
    lines.extend(
        [
            "",
            "## Safety, stopping, and ranking",
            "",
            f"- Executed calls: {int(selection['calls'])}",
            (
                "- Raw gain per call: "
                f"{_number(selection['source_balanced_raw_gain_per_call'])}"
            ),
            (
                "- Induced-harm mass: "
                f"{_number(risk['source_balanced_induced_harm_mass'])}"
            ),
            (
                "- Net-negative-call mass: "
                f"{_number(risk['source_balanced_net_negative_call_mass'])}"
            ),
            f"- Unnecessary-call rate: {_number(selection['unnecessary_call_rate'])}",
            (
                "- Positive-utility precision among calls: "
                f"{_number(selection['positive_utility_call_precision'])}"
            ),
            f"- Correct-stopping rate: {_number(selection['correct_stopping_rate'])}",
            (
                "- Learned crop rescue rate within helpful states: "
                f"{_number(ranking['top1_rescue_rate_within_helpful_states'])}"
            ),
            (
                "- Random crop rescue rate within helpful states: "
                f"{_number(ranking['random_rescue_rate_within_helpful_states'])}"
            ),
            f"- Oracle utility: {_number(source['oracle_utility'])}",
            f"- Oracle regret: {_number(report['oracle_regret'])}",
            "",
            "## Matched-call and exhaustive baselines",
            "",
            (
                "- Learned gate + learned crop utility: "
                f"{_number(source['utility'])}"
            ),
            (
                "- Entropy gate + learned crop utility: "
                f"{_number(baselines['matched_budget_entropy_gate_source_utility_learned_crop'])}"
            ),
            (
                "- Entropy gate + random crop utility: "
                f"{_number(baselines['matched_budget_entropy_gate_source_utility_random_crop'])}"
            ),
            (
                "- Random gate + random crop expected utility: "
                f"{_number(baselines['matched_budget_random_gate_source_utility_random_crop_expected'])}"
            ),
            (
                "- UG-style exhaustive entropy utility, charging all four calls: "
                f"{_number(baselines['ug_style_exhaustive_entropy_source_utility'])}"
            ),
            "",
            "Fixed-crop, random-crop, same-gate, post-action entropy, and all-call "
            "controls remain in the bound JSON report.",
            "",
            "## Integrity",
            "",
            f"- Policy freeze SHA-256: `{run['policy_freeze_sha256']}`",
            f"- Model SHA-256: `{run['model_sha256']}`",
            f"- Manifest SHA-256: `{run['manifest_sha256']}`",
            f"- Formal audit SHA-256: `{run['formal_audit_sha256']}`",
            f"- Rollouts SHA-256: `{run['rollouts_sha256']}`",
            f"- Features SHA-256: `{run['features_sha256']}`",
            f"- Code revision: `{run['code_revision']}`",
            "- Formal outcomes used for tuning: `false`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the frozen DocVQA-train formal JSON result"
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.report.resolve()
    output_path = args.output.resolve()
    check_hash(report_path, args.expected_report_sha256, "formal report")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite DocVQA formal rendering: {output_path}")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("DocVQA formal report must be a JSON object")
    rendered = render_report(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        handle.write(rendered)
    print(output_path)


if __name__ == "__main__":
    main()
