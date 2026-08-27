from __future__ import annotations

from typing import Mapping, Sequence


def _format(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_markdown_report(report: Mapping[str, object]) -> str:
    diagnostic = report["entropy_diagnostic"]
    if not isinstance(diagnostic, Mapping):
        raise ValueError("report entropy_diagnostic must be a mapping")
    policy_results = report["policy_results"]
    if not isinstance(policy_results, Sequence):
        raise ValueError("report policy_results must be a sequence")
    lines = [
        "# Beyond Entropy MVP Report",
        "",
        "> This report uses synthetic counterfactual rollouts to validate the pipeline. "
        "It is not an empirical paper result.",
        "",
        "## Entropy diagnostic",
        "",
        f"- Zoom actions: {_format(diagnostic['n_zoom_actions'])}",
        f"- Paired decisions: {_format(diagnostic['n_decisions'])}",
        f"- Confidence-gain rate: {_format(diagnostic['confidence_gain_rate'])}",
        f"- Task-improvement rate: {_format(diagnostic['task_improvement_rate'])}",
        "- Strict SCGR (confidence gain and harm): "
        f"{_format(diagnostic['spurious_confidence_gain_rate'])}",
        "- Non-beneficial confidence-gain rate: "
        f"{_format(diagnostic['nonbeneficial_confidence_gain_rate'])}",
        f"- Confidence-gain precision: {_format(diagnostic['confidence_gain_precision'])}",
        f"- Pearson corr(Delta-H, Delta-success): {_format(diagnostic['entropy_success_pearson'])}",
        f"- Entropy Top-1 mismatch: {_format(diagnostic['entropy_top1_mismatch_rate'])}",
        "- Mean entropy-selection success regret: "
        f"{_format(diagnostic['mean_entropy_selection_regret'])}",
        "",
        "## Policy comparison",
        "",
        "| Policy | Accuracy | Accuracy gain | Tool calls | Visual cost | Tool use | Correct stop | Policy utility | Oracle regret | Gain/call |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in policy_results:
        if not isinstance(item, Mapping):
            raise ValueError("policy result must be a mapping")
        lines.append(
            "| {policy} | {accuracy} | {accuracy_gain} | {avg_tool_calls} | "
            "{avg_visual_cost} | {tool_use_rate} | {correct_stopping_rate} | "
            "{mean_policy_utility} | {mean_oracle_regret} | "
            "{marginal_accuracy_gain_per_tool_call} |".format(
                **{key: _format(value) for key, value in item.items()}
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- `entropy_search` observes every candidate's post-action entropy, so its selection "
            "cost and policy utility count all candidate tool calls.",
            "- Entropy thresholds are tuned on training data and held fixed on the test split.",
            "- `learned_voi` scores candidates using only fields available before execution, then "
            "subtracts runtime cost and either performs one zoom or stops.",
            "- `oracle_voi` reads counterfactual labels and is only a diagnostic upper bound.",
            "- Replace synthetic rows with real paired VLM rollouts before making research claims.",
            "",
        ]
    )
    return "\n".join(lines)
