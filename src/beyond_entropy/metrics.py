from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Iterable, Sequence

from .dataset import group_by_decision
from .policies import Policy, realized_policy_utility
from .schema import ActionRecord


@dataclass(frozen=True)
class EntropyDiagnostic:
    n_zoom_actions: int
    n_decisions: int
    confidence_gain_rate: float
    task_improvement_rate: float
    spurious_confidence_gain_rate: float
    nonbeneficial_confidence_gain_rate: float
    confidence_gain_precision: float | None
    entropy_success_pearson: float | None
    entropy_top1_mismatch_rate: float
    mean_entropy_selection_regret: float


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2:
        return None
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_norm = sum((x - left_mean) ** 2 for x in left) ** 0.5
    right_norm = sum((y - right_mean) ** 2 for y in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return numerator / (left_norm * right_norm)


def entropy_diagnostic(records: Iterable[ActionRecord]) -> EntropyDiagnostic:
    materialized = list(records)
    zooms = [record for record in materialized if record.action_type == "ZOOM"]
    if not zooms:
        raise ValueError("entropy diagnostic requires ZOOM actions")
    entropy_deltas = [record.delta_entropy for record in zooms]
    success_deltas = [record.delta_success for record in zooms]
    confidence_gain_count = sum(delta > 0.0 for delta in entropy_deltas)
    mismatches: list[bool] = []
    selection_regrets: list[float] = []
    grouped = group_by_decision(materialized)
    for siblings in grouped.values():
        sibling_zooms = [record for record in siblings if record.action_type == "ZOOM"]
        entropy_top = max(
            sibling_zooms,
            key=lambda record: (record.delta_entropy, record.action_id),
        )
        best_success = max(record.delta_success for record in sibling_zooms)
        success_top_ids = {
            record.action_id
            for record in sibling_zooms
            if abs(record.delta_success - best_success) <= 1e-12
        }
        mismatches.append(entropy_top.action_id not in success_top_ids)
        selection_regrets.append(best_success - entropy_top.delta_success)
    return EntropyDiagnostic(
        n_zoom_actions=len(zooms),
        n_decisions=len(grouped),
        confidence_gain_rate=confidence_gain_count / len(zooms),
        task_improvement_rate=mean(delta > 0.0 for delta in success_deltas),
        spurious_confidence_gain_rate=mean(
            entropy > 0.0 and success < 0.0
            for entropy, success in zip(entropy_deltas, success_deltas)
        ),
        nonbeneficial_confidence_gain_rate=mean(
            entropy > 0.0 and success <= 0.0
            for entropy, success in zip(entropy_deltas, success_deltas)
        ),
        confidence_gain_precision=(
            sum(
                entropy > 0.0 and success > 0.0
                for entropy, success in zip(entropy_deltas, success_deltas)
            )
            / confidence_gain_count
            if confidence_gain_count
            else None
        ),
        entropy_success_pearson=_pearson(entropy_deltas, success_deltas),
        entropy_top1_mismatch_rate=mean(mismatches),
        mean_entropy_selection_regret=mean(selection_regrets),
    )


def evaluate_policy(
    records: Sequence[ActionRecord],
    policy: Policy,
    *,
    lambda_cost: float,
) -> dict[str, float | int | str | None]:
    grouped = group_by_decision(records)
    if not grouped:
        raise ValueError("policy evaluation requires at least one decision")
    outcomes: list[float] = []
    baseline_outcomes: list[float] = []
    tool_calls: list[int] = []
    costs: list[float] = []
    gains: list[float] = []
    utilities: list[float] = []
    regrets: list[float] = []
    zoom_decisions = 0
    tool_decisions = 0
    unnecessary_tools = 0
    stopped_decisions = 0
    missed_information = 0
    correct_stops = 0
    for decision_key in sorted(grouped):
        siblings = grouped[decision_key]
        answer = next(record for record in siblings if record.action_type == "ANSWER")
        decision = policy.select(siblings)
        selected = decision.selected
        zooms = [record for record in siblings if record.action_type == "ZOOM"]
        oracle_utility = max(0.0, max(record.voi(lambda_cost) for record in zooms))
        selected_gain = selected.delta_success if selected.action_type == "ZOOM" else 0.0
        policy_utility = realized_policy_utility(decision, lambda_cost)
        outcomes.append(selected.correct_after)
        baseline_outcomes.append(answer.correct_after)
        tool_calls.append(decision.tool_calls)
        costs.append(decision.visual_cost)
        gains.append(selected_gain)
        utilities.append(policy_utility)
        regrets.append(oracle_utility - policy_utility)
        should_stop = oracle_utility <= 0.0
        did_stop = decision.tool_calls == 0
        correct_stops += int(should_stop == did_stop)
        if did_stop:
            stopped_decisions += 1
            missed_information += int(not should_stop)
        if decision.tool_calls > 0:
            tool_decisions += 1
            unnecessary_tools += int(policy_utility <= 0.0)
        zoom_decisions += int(selected.action_type == "ZOOM")
    avg_calls = mean(tool_calls)
    accuracy = mean(outcomes)
    baseline_accuracy = mean(baseline_outcomes)
    mean_utility = mean(utilities)
    return {
        "policy": policy.name,
        "n_decisions": len(grouped),
        "accuracy": accuracy,
        "answer_now_accuracy": baseline_accuracy,
        "accuracy_gain": accuracy - baseline_accuracy,
        "avg_tool_calls": avg_calls,
        "avg_visual_cost": mean(costs),
        "tool_use_rate": tool_decisions / len(grouped),
        "zoom_rate": zoom_decisions / len(grouped),
        "unnecessary_tool_call_rate": (
            unnecessary_tools / tool_decisions if tool_decisions else 0.0
        ),
        "missed_information_rate": (
            missed_information / stopped_decisions if stopped_decisions else 0.0
        ),
        "correct_stopping_rate": correct_stops / len(grouped),
        "mean_success_gain": mean(gains),
        "mean_policy_utility": mean_utility,
        "mean_realized_voi": mean_utility,
        "mean_oracle_regret": mean(regrets),
        "marginal_accuracy_gain_per_tool_call": (
            (accuracy - baseline_accuracy) / avg_calls if avg_calls > 0.0 else None
        ),
    }


def diagnostic_to_dict(diagnostic: EntropyDiagnostic) -> dict[str, object]:
    return asdict(diagnostic)
