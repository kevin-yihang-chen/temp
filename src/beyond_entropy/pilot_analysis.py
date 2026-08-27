from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Mapping, Sequence

from .dataset import group_by_decision
from .metrics import (
    bootstrap_entropy_diagnostic,
    bootstrap_policy_evaluation,
    diagnostic_to_dict,
    entropy_diagnostic,
    evaluate_policy,
)
from .policies import (
    AnswerNowPolicy,
    EntropySearchPolicy,
    FixedCenterZoomPolicy,
    OracleVOIPolicy,
    Policy,
    RandomZoomPolicy,
)
from .schema import ActionRecord


def _transition_name(record: ActionRecord) -> str:
    return f"{record.correct_before:g}->{record.correct_after:g}"


def _analyze_slice(
    records: Sequence[ActionRecord],
    *,
    lambda_cost: float,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, object]:
    grouped = group_by_decision(records)
    zooms = [record for record in records if record.action_type == "ZOOM"]
    transitions = Counter(_transition_name(record) for record in zooms)
    helpful_decisions: list[bool] = []
    harmful_decisions: list[bool] = []
    oracle_headroom: list[float] = []
    for siblings in grouped.values():
        answer = next(record for record in siblings if record.action_type == "ANSWER")
        sibling_zooms = [record for record in siblings if record.action_type == "ZOOM"]
        gains = [record.delta_success for record in sibling_zooms]
        helpful_decisions.append(any(gain > 0.0 for gain in gains))
        harmful_decisions.append(any(gain < 0.0 for gain in gains))
        oracle_headroom.append(max(0.0, max(gains)))
        if any(record.correct_before != answer.correct_before for record in sibling_zooms):
            raise ValueError("slice contains inconsistent baseline correctness")
    policies: list[Policy] = [
        AnswerNowPolicy(),
        RandomZoomPolicy(seed=seed),
        FixedCenterZoomPolicy(),
        EntropySearchPolicy(),
        OracleVOIPolicy(lambda_cost),
    ]
    policy_results: list[dict[str, object]] = []
    for policy_index, policy in enumerate(policies):
        result: dict[str, object] = dict(
            evaluate_policy(records, policy, lambda_cost=lambda_cost)
        )
        result["bootstrap"] = bootstrap_policy_evaluation(
            records,
            policy,
            lambda_cost=lambda_cost,
            n_resamples=bootstrap_resamples,
            seed=seed + policy_index,
        )
        policy_results.append(result)
    return {
        "n_records": len(records),
        "n_states": len({record.state_id for record in records}),
        "n_decisions": len(grouped),
        "n_zoom_actions": len(zooms),
        "transition_counts": dict(sorted(transitions.items())),
        "any_helpful_zoom_rate": mean(helpful_decisions),
        "any_harmful_zoom_rate": mean(harmful_decisions),
        "mean_oracle_success_headroom": mean(oracle_headroom),
        "entropy_diagnostic": diagnostic_to_dict(entropy_diagnostic(records)),
        "entropy_bootstrap": bootstrap_entropy_diagnostic(
            records,
            n_resamples=bootstrap_resamples,
            seed=seed,
        ),
        "policy_results": policy_results,
    }


def analyze_counterfactual_pilot(
    records: Sequence[ActionRecord],
    *,
    state_strata: Mapping[str, str],
    lambda_cost: float = 0.05,
    bootstrap_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, object]:
    if lambda_cost < 0.0:
        raise ValueError("lambda_cost must be non-negative")
    record_states = {record.state_id for record in records}
    missing = record_states - set(state_strata)
    if missing:
        raise ValueError(f"manifest strata missing states: {sorted(missing)}")
    strata: dict[str, list[ActionRecord]] = {}
    for record in records:
        strata.setdefault(state_strata[record.state_id], []).append(record)
    return {
        "scientific_status": "frozen diagnostic pilot; not a final benchmark claim",
        "lambda_cost": lambda_cost,
        "bootstrap_resamples": bootstrap_resamples,
        "seed": seed,
        "overall": _analyze_slice(
            records,
            lambda_cost=lambda_cost,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
        ),
        "by_stratum": {
            name: _analyze_slice(
                stratum_records,
                lambda_cost=lambda_cost,
                bootstrap_resamples=bootstrap_resamples,
                seed=seed,
            )
            for name, stratum_records in sorted(strata.items())
        },
    }


def build_pilot_markdown(report: Mapping[str, object]) -> str:
    slices: list[tuple[str, Mapping[str, object]]] = []
    overall = report["overall"]
    if not isinstance(overall, Mapping):
        raise ValueError("overall report must be a mapping")
    slices.append(("overall", overall))
    by_stratum = report["by_stratum"]
    if not isinstance(by_stratum, Mapping):
        raise ValueError("by_stratum report must be a mapping")
    slices.extend(
        (str(name), value)
        for name, value in sorted(by_stratum.items())
        if isinstance(value, Mapping)
    )

    def policy_result(
        slice_report: Mapping[str, object], name: str
    ) -> Mapping[str, object]:
        policies = slice_report["policy_results"]
        if not isinstance(policies, Sequence):
            raise ValueError("policy_results must be a sequence")
        for policy in policies:
            if isinstance(policy, Mapping) and policy.get("policy") == name:
                return policy
        raise ValueError(f"missing policy result: {name}")

    def policy_accuracy_cell(slice_report: Mapping[str, object], name: str) -> str:
        policy = policy_result(slice_report, name)
        point = policy["accuracy"]
        if not isinstance(point, (int, float)):
            raise ValueError("policy accuracy must be numeric")
        result = f"{float(point):.4f}"
        bootstrap = policy.get("bootstrap")
        if isinstance(bootstrap, Mapping):
            metrics = bootstrap.get("metrics")
            if isinstance(metrics, Mapping):
                accuracy = metrics.get("accuracy")
                if isinstance(accuracy, Mapping) and accuracy.get("ci_low") is not None:
                    result += " [{:.4f}, {:.4f}]".format(
                        float(accuracy["ci_low"]), float(accuracy["ci_high"])
                    )
        return result

    lines = [
        "# Frozen counterfactual pilot report",
        "",
        "> Diagnostic pilot only; do not cite these numbers as a final benchmark result.",
        "",
        "| Slice | States | Answer now [95% CI] | Entropy search [95% CI] | Oracle VOI [95% CI] | Strict SCGR | Non-beneficial confidence | Entropy Top-1 mismatch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, slice_report in slices:
        diagnostic = slice_report["entropy_diagnostic"]
        if not isinstance(diagnostic, Mapping):
            raise ValueError("entropy_diagnostic must be a mapping")
        n_states = slice_report["n_states"]
        if not isinstance(n_states, (int, float)):
            raise ValueError("n_states must be numeric")
        lines.append(
            "| {name} | {states} | {answer} | {entropy} | {oracle} | "
            "{scgr:.4f} | {nonbeneficial:.4f} | {mismatch:.4f} |".format(
                name=name,
                states=int(n_states),
                answer=policy_accuracy_cell(slice_report, "answer_now"),
                entropy=policy_accuracy_cell(slice_report, "entropy_search"),
                oracle=policy_accuracy_cell(slice_report, "oracle_voi"),
                scgr=float(diagnostic["spurious_confidence_gain_rate"]),
                nonbeneficial=float(
                    diagnostic["nonbeneficial_confidence_gain_rate"]
                ),
                mismatch=float(diagnostic["entropy_top1_mismatch_rate"]),
            )
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Entropy search pays for all candidate executions.",
            "- Oracle VOI reads counterfactual correctness and is not deployable.",
            "- Confidence intervals resample complete states, never action rows.",
            "- A larger frozen run is required before making a scientific claim.",
            "",
        ]
    )
    return "\n".join(lines)
