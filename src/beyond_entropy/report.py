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
        f"- Confidence-gain rate: {_format(diagnostic['confidence_gain_rate'])}",
        f"- Task-improvement rate: {_format(diagnostic['task_improvement_rate'])}",
        "- Spurious Confidence Gain Rate (SCGR): "
        f"{_format(diagnostic['spurious_confidence_gain_rate'])}",
        f"- Pearson corr(Delta-H, Delta-success): {_format(diagnostic['entropy_success_pearson'])}",
        "",
        "## Policy comparison",
        "",
        "| Policy | Accuracy | Tool calls | Visual cost | Zoom rate | Correct stop | Realized VOI | Oracle regret |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in policy_results:
        if not isinstance(item, Mapping):
            raise ValueError("policy result must be a mapping")
        lines.append(
            "| {policy} | {accuracy} | {avg_tool_calls} | {avg_visual_cost} | "
            "{zoom_rate} | {correct_stopping_rate} | {mean_realized_voi} | "
            "{mean_oracle_regret} |".format(
                **{key: _format(value) for key, value in item.items()}
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- `entropy_search` observes every candidate's post-action entropy, so its selection "
            "cost counts all candidate tool calls.",
            "- `learned_voi` scores candidates using only fields available before execution, then "
            "either performs one zoom or stops.",
            "- `oracle_voi` reads counterfactual labels and is only a diagnostic upper bound.",
            "- Replace synthetic rows with real paired VLM rollouts before making research claims.",
            "",
        ]
    )
    return "\n".join(lines)
