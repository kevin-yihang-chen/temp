from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Iterable, Sequence

from .dataset import group_by_state
from .policies import Policy
from .schema import ActionRecord


@dataclass(frozen=True)
class EntropyDiagnostic:
    n_zoom_actions: int
    confidence_gain_rate: float
    task_improvement_rate: float
    spurious_confidence_gain_rate: float
    entropy_success_pearson: float | None


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
    zooms = [record for record in records if record.action_type == "ZOOM"]
    if not zooms:
        raise ValueError("entropy diagnostic requires ZOOM actions")
    entropy_deltas = [record.delta_entropy for record in zooms]
    success_deltas = [record.delta_success for record in zooms]
    return EntropyDiagnostic(
        n_zoom_actions=len(zooms),
        confidence_gain_rate=mean(delta > 0.0 for delta in entropy_deltas),
        task_improvement_rate=mean(delta > 0.0 for delta in success_deltas),
        spurious_confidence_gain_rate=mean(
            entropy > 0.0 and success < 0.0
            for entropy, success in zip(entropy_deltas, success_deltas)
        ),
        entropy_success_pearson=_pearson(entropy_deltas, success_deltas),
    )


def evaluate_policy(
    records: Sequence[ActionRecord],
    policy: Policy,
    *,
    lambda_cost: float,
) -> dict[str, float | int | str | None]:
    grouped = group_by_state(records)
    if not grouped:
        raise ValueError("policy evaluation requires at least one state")
    outcomes: list[float] = []
    tool_calls: list[int] = []
    costs: list[float] = []
    selected_values: list[float] = []
    regrets: list[float] = []
    zoom_decisions = 0
    unnecessary_zooms = 0
    answer_decisions = 0
    missed_information = 0
    correct_stops = 0
    for state_id in sorted(grouped):
        siblings = grouped[state_id]
        decision = policy.select(siblings)
        selected = decision.selected
        zooms = [record for record in siblings if record.action_type == "ZOOM"]
        best_zoom_value = max(record.voi(lambda_cost) for record in zooms)
        oracle_value = max(0.0, best_zoom_value)
        selected_value = selected.voi(lambda_cost) if selected.action_type == "ZOOM" else 0.0
        outcomes.append(selected.correct_after)
        tool_calls.append(decision.tool_calls)
        costs.append(decision.visual_cost)
        selected_values.append(selected_value)
        regrets.append(oracle_value - selected_value)
        should_stop = best_zoom_value <= 0.0
        did_stop = selected.action_type == "ANSWER"
        correct_stops += int(should_stop == did_stop)
        if did_stop:
            answer_decisions += 1
            missed_information += int(not should_stop)
        else:
            zoom_decisions += 1
            unnecessary_zooms += int(selected.delta_success <= 0.0)
    avg_calls = mean(tool_calls)
    return {
        "policy": policy.name,
        "n_states": len(grouped),
        "accuracy": mean(outcomes),
        "avg_tool_calls": avg_calls,
        "avg_visual_cost": mean(costs),
        "zoom_rate": zoom_decisions / len(grouped),
        "unnecessary_tool_call_rate": (
            unnecessary_zooms / zoom_decisions if zoom_decisions else 0.0
        ),
        "missed_information_rate": (
            missed_information / answer_decisions if answer_decisions else 0.0
        ),
        "correct_stopping_rate": correct_stops / len(grouped),
        "mean_realized_voi": mean(selected_values),
        "mean_oracle_regret": mean(regrets),
        "success_per_tool_call": mean(outcomes) / avg_calls if avg_calls > 0.0 else None,
    }


def diagnostic_to_dict(diagnostic: EntropyDiagnostic) -> dict[str, object]:
    return asdict(diagnostic)
