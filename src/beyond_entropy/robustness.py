from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence


ROBUSTNESS_METRICS = (
    "accuracy",
    "accuracy_gain",
    "tool_use_rate",
    "mean_policy_utility",
    "mean_oracle_regret",
)


def _lambda_result(report: Mapping[str, Any], lambda_cost: float) -> Mapping[str, Any]:
    matches = [
        sweep
        for sweep in report["lambda_sweep"]
        if abs(float(sweep["lambda_cost"]) - lambda_cost) <= 1e-12
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one lambda={lambda_cost} sweep, found {len(matches)}")
    return matches[0]


def aggregate_semantic_reports(
    report_paths: Sequence[str | Path],
    *,
    lambda_cost: float = 0.05,
) -> dict[str, Any]:
    """Summarize repeated grouped splits without treating them as independent CIs."""

    if not report_paths:
        raise ValueError("at least one semantic report is required")
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in report_paths]
    seeds = [int(report["run"]["seed"]) for report in reports]
    if len(set(seeds)) != len(seeds):
        raise ValueError("semantic robustness reports contain duplicate seeds")
    by_policy: dict[str, dict[str, list[float]]] = {}
    expected_policies: set[str] | None = None
    for report in reports:
        sweep = _lambda_result(report, lambda_cost)
        policies = {
            str(result["policy"]): result for result in sweep["policy_results"]
        }
        if expected_policies is None:
            expected_policies = set(policies)
        elif set(policies) != expected_policies:
            raise ValueError("semantic robustness reports contain different policies")
        for policy_name, result in policies.items():
            metrics = by_policy.setdefault(
                policy_name,
                {metric: [] for metric in ROBUSTNESS_METRICS},
            )
            for metric in ROBUSTNESS_METRICS:
                metrics[metric].append(float(result[metric]))
    policy_summary: dict[str, Any] = {}
    for policy_name, metrics in sorted(by_policy.items()):
        summary: dict[str, Any] = {
            metric: {
                "mean": mean(values),
                "median": median(values),
                "min": min(values),
                "max": max(values),
            }
            for metric, values in metrics.items()
        }
        utilities = metrics["mean_policy_utility"]
        summary["positive_utility_splits"] = sum(value > 0.0 for value in utilities)
        summary["nonnegative_utility_splits"] = sum(value >= 0.0 for value in utilities)
        summary["n_splits"] = len(utilities)
        policy_summary[policy_name] = summary
    return {
        "scientific_status": (
            "repeated grouped-split robustness diagnostic; split estimates overlap "
            "and are not an independent confidence interval"
        ),
        "lambda_cost": lambda_cost,
        "seeds": sorted(seeds),
        "reports": [str(Path(path).resolve()) for path in report_paths],
        "policies": policy_summary,
    }


def build_robustness_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Semantic grouped-split robustness",
        "",
        "> Repeated split diagnostic only; overlapping test sets are not independent.",
        "",
        "| Policy | Mean accuracy gain [min, max] | Mean tool rate | Mean utility [min, max] | Positive splits |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy, summary in report["policies"].items():
        gain = summary["accuracy_gain"]
        tool = summary["tool_use_rate"]
        utility = summary["mean_policy_utility"]
        lines.append(
            "| {} | {:.4f} [{:.4f}, {:.4f}] | {:.4f} | {:.4f} [{:.4f}, {:.4f}] | {}/{} |".format(
                policy,
                gain["mean"],
                gain["min"],
                gain["max"],
                tool["mean"],
                utility["mean"],
                utility["min"],
                utility["max"],
                summary["positive_utility_splits"],
                summary["n_splits"],
            )
        )
    lines.append("")
    return "\n".join(lines)
