from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Iterable, Literal, Sequence

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


@dataclass(frozen=True)
class _PolicyOutcome:
    """Sufficient statistics for one fixed policy decision.

    Keeping the selected decision fixed before resampling matters for stochastic
    policies: a bootstrap duplicate should repeat the observed policy decision,
    not draw a different random crop merely because its synthetic identifier
    changed.
    """

    decision_key: tuple[str, str]
    state_id: str
    image_id: str
    source_id: str
    outcome: float
    baseline_outcome: float
    tool_calls: int
    visual_cost: float
    success_gain: float
    utility: float
    oracle_regret: float
    tool_used: int
    zoom_selected: int
    unnecessary_tool: int
    stopped: int
    missed_information: int
    correct_stop: int


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


def _policy_outcomes(
    records: Sequence[ActionRecord],
    policy: Policy,
    *,
    lambda_cost: float,
) -> list[_PolicyOutcome]:
    grouped = group_by_decision(records)
    if not grouped:
        raise ValueError("policy evaluation requires at least one decision")
    outcomes: list[_PolicyOutcome] = []
    for decision_key in sorted(grouped):
        siblings = grouped[decision_key]
        answer = next(record for record in siblings if record.action_type == "ANSWER")
        decision = policy.select(siblings)
        selected = decision.selected
        zooms = [record for record in siblings if record.action_type == "ZOOM"]
        oracle_utility = max(0.0, max(record.voi(lambda_cost) for record in zooms))
        selected_gain = selected.delta_success if selected.action_type == "ZOOM" else 0.0
        policy_utility = realized_policy_utility(decision, lambda_cost)
        should_stop = oracle_utility <= 0.0
        did_stop = decision.tool_calls == 0
        outcomes.append(
            _PolicyOutcome(
                decision_key=decision_key,
                state_id=answer.state_id,
                image_id=answer.image_id,
                source_id=answer.source_id,
                outcome=selected.correct_after,
                baseline_outcome=answer.correct_after,
                tool_calls=decision.tool_calls,
                visual_cost=decision.visual_cost,
                success_gain=selected_gain,
                utility=policy_utility,
                oracle_regret=oracle_utility - policy_utility,
                tool_used=int(decision.tool_calls > 0),
                zoom_selected=int(selected.action_type == "ZOOM"),
                unnecessary_tool=int(
                    decision.tool_calls > 0 and policy_utility <= 0.0
                ),
                stopped=int(did_stop),
                missed_information=int(did_stop and not should_stop),
                correct_stop=int(should_stop == did_stop),
            )
        )
    return outcomes


def _summarize_policy_outcomes(
    outcomes: Sequence[_PolicyOutcome],
    *,
    policy_name: str,
) -> dict[str, float | int | str | None]:
    if not outcomes:
        raise ValueError("policy evaluation requires at least one outcome")
    avg_calls = mean(outcome.tool_calls for outcome in outcomes)
    accuracy = mean(outcome.outcome for outcome in outcomes)
    baseline_accuracy = mean(outcome.baseline_outcome for outcome in outcomes)
    mean_utility = mean(outcome.utility for outcome in outcomes)
    tool_decisions = sum(outcome.tool_used for outcome in outcomes)
    stopped_decisions = sum(outcome.stopped for outcome in outcomes)
    return {
        "policy": policy_name,
        "n_decisions": len(outcomes),
        "accuracy": accuracy,
        "answer_now_accuracy": baseline_accuracy,
        "accuracy_gain": accuracy - baseline_accuracy,
        "avg_tool_calls": avg_calls,
        "avg_visual_cost": mean(outcome.visual_cost for outcome in outcomes),
        "tool_use_rate": tool_decisions / len(outcomes),
        "zoom_rate": mean(outcome.zoom_selected for outcome in outcomes),
        "unnecessary_tool_call_rate": (
            sum(outcome.unnecessary_tool for outcome in outcomes) / tool_decisions
            if tool_decisions
            else 0.0
        ),
        "missed_information_rate": (
            sum(outcome.missed_information for outcome in outcomes) / stopped_decisions
            if stopped_decisions
            else 0.0
        ),
        "correct_stopping_rate": mean(outcome.correct_stop for outcome in outcomes),
        "mean_success_gain": mean(outcome.success_gain for outcome in outcomes),
        "mean_policy_utility": mean_utility,
        "mean_realized_voi": mean_utility,
        "mean_oracle_regret": mean(outcome.oracle_regret for outcome in outcomes),
        "marginal_accuracy_gain_per_tool_call": (
            (accuracy - baseline_accuracy) / avg_calls if avg_calls > 0.0 else None
        ),
    }


def evaluate_policy(
    records: Sequence[ActionRecord],
    policy: Policy,
    *,
    lambda_cost: float,
) -> dict[str, float | int | str | None]:
    return _summarize_policy_outcomes(
        _policy_outcomes(records, policy, lambda_cost=lambda_cost),
        policy_name=policy.name,
    )


def _policy_sufficient_vector(outcomes: Sequence[_PolicyOutcome]) -> tuple[float, ...]:
    return (
        float(len(outcomes)),
        sum(outcome.outcome for outcome in outcomes),
        sum(outcome.baseline_outcome for outcome in outcomes),
        float(sum(outcome.tool_calls for outcome in outcomes)),
        sum(outcome.visual_cost for outcome in outcomes),
        sum(outcome.success_gain for outcome in outcomes),
        sum(outcome.utility for outcome in outcomes),
        sum(outcome.oracle_regret for outcome in outcomes),
        float(sum(outcome.tool_used for outcome in outcomes)),
        float(sum(outcome.zoom_selected for outcome in outcomes)),
        float(sum(outcome.unnecessary_tool for outcome in outcomes)),
        float(sum(outcome.stopped for outcome in outcomes)),
        float(sum(outcome.missed_information for outcome in outcomes)),
        float(sum(outcome.correct_stop for outcome in outcomes)),
    )


def _summarize_policy_vector(
    vector: Sequence[float],
    *,
    policy_name: str,
) -> dict[str, float | int | str | None]:
    (
        n_decisions,
        outcome_sum,
        baseline_sum,
        tool_call_sum,
        visual_cost_sum,
        success_gain_sum,
        utility_sum,
        regret_sum,
        tool_decisions,
        zoom_decisions,
        unnecessary_tools,
        stopped_decisions,
        missed_information,
        correct_stops,
    ) = vector
    if n_decisions <= 0.0:
        raise ValueError("policy evaluation requires at least one outcome")
    accuracy = outcome_sum / n_decisions
    baseline_accuracy = baseline_sum / n_decisions
    avg_calls = tool_call_sum / n_decisions
    mean_utility = utility_sum / n_decisions
    return {
        "policy": policy_name,
        "n_decisions": int(n_decisions),
        "accuracy": accuracy,
        "answer_now_accuracy": baseline_accuracy,
        "accuracy_gain": accuracy - baseline_accuracy,
        "avg_tool_calls": avg_calls,
        "avg_visual_cost": visual_cost_sum / n_decisions,
        "tool_use_rate": tool_decisions / n_decisions,
        "zoom_rate": zoom_decisions / n_decisions,
        "unnecessary_tool_call_rate": (
            unnecessary_tools / tool_decisions if tool_decisions else 0.0
        ),
        "missed_information_rate": (
            missed_information / stopped_decisions if stopped_decisions else 0.0
        ),
        "correct_stopping_rate": correct_stops / n_decisions,
        "mean_success_gain": success_gain_sum / n_decisions,
        "mean_policy_utility": mean_utility,
        "mean_realized_voi": mean_utility,
        "mean_oracle_regret": regret_sum / n_decisions,
        "marginal_accuracy_gain_per_tool_call": (
            (accuracy - baseline_accuracy) / avg_calls if avg_calls > 0.0 else None
        ),
    }


def bootstrap_policy_evaluation(
    records: Sequence[ActionRecord],
    policy: Policy,
    *,
    lambda_cost: float,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
    cluster_by: Literal["state_id", "image_id", "source_id"] = "state_id",
) -> dict[str, object]:
    """Bootstrap a fixed policy evaluation using whole deployment groups."""

    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    outcomes = _policy_outcomes(records, policy, lambda_cost=lambda_cost)
    by_cluster: dict[str, list[_PolicyOutcome]] = {}
    for outcome in outcomes:
        cluster_id = getattr(outcome, cluster_by)
        by_cluster.setdefault(cluster_id, []).append(outcome)
    cluster_ids = sorted(by_cluster)
    rng = random.Random(seed)
    cluster_vectors = {
        cluster_id: _policy_sufficient_vector(cluster_outcomes)
        for cluster_id, cluster_outcomes in by_cluster.items()
    }
    vector_width = len(next(iter(cluster_vectors.values())))
    samples: dict[str, list[float]] = {}
    for _ in range(n_resamples):
        totals = [0.0] * vector_width
        for cluster_id in rng.choices(cluster_ids, k=len(cluster_ids)):
            vector = cluster_vectors[cluster_id]
            for index, component_value in enumerate(vector):
                totals[index] += component_value
        summary = _summarize_policy_vector(totals, policy_name=policy.name)
        for name, value in summary.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                samples.setdefault(name, []).append(float(value))
    point = _summarize_policy_outcomes(outcomes, policy_name=policy.name)
    alpha = (1.0 - confidence) / 2.0
    intervals: dict[str, dict[str, float | int | None]] = {}
    for name, estimate in point.items():
        if name in ("policy", "n_decisions"):
            continue
        values = samples.get(name, [])
        intervals[name] = {
            "estimate": estimate if isinstance(estimate, (int, float)) else None,
            "ci_low": _percentile(values, alpha) if values else None,
            "ci_high": _percentile(values, 1.0 - alpha) if values else None,
        }
    return {
        "resampling_unit": cluster_by,
        "confidence": confidence,
        "n_resamples": n_resamples,
        "seed": seed,
        "policy": policy.name,
        "lambda_cost": lambda_cost,
        "n_states": len({outcome.state_id for outcome in outcomes}),
        "n_clusters": len(cluster_ids),
        "n_decisions": len(outcomes),
        "metrics": intervals,
    }


def paired_bootstrap_policy_difference(
    left_records: Sequence[ActionRecord],
    left_policy: Policy,
    right_records: Sequence[ActionRecord],
    right_policy: Policy,
    *,
    lambda_cost: float,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, object]:
    """Estimate right-minus-left policy differences on matched state clusters."""

    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    left_outcomes = _policy_outcomes(
        left_records,
        left_policy,
        lambda_cost=lambda_cost,
    )
    right_outcomes = _policy_outcomes(
        right_records,
        right_policy,
        lambda_cost=lambda_cost,
    )
    left_by_key = {outcome.decision_key: outcome for outcome in left_outcomes}
    right_by_key = {outcome.decision_key: outcome for outcome in right_outcomes}
    if set(left_by_key) != set(right_by_key):
        raise ValueError("paired policy inputs contain different decision keys")
    for decision_key in left_by_key:
        left = left_by_key[decision_key]
        right = right_by_key[decision_key]
        if abs(left.baseline_outcome - right.baseline_outcome) > 1e-12:
            raise ValueError(f"paired decision {decision_key!r} has different baseline outcome")

    left_by_state: dict[str, list[_PolicyOutcome]] = {}
    right_by_state: dict[str, list[_PolicyOutcome]] = {}
    for decision_key in sorted(left_by_key):
        left = left_by_key[decision_key]
        right = right_by_key[decision_key]
        left_by_state.setdefault(left.state_id, []).append(left)
        right_by_state.setdefault(right.state_id, []).append(right)
    state_ids = sorted(left_by_state)
    if set(state_ids) != set(right_by_state):
        raise ValueError("paired policy inputs contain different state clusters")

    left_vectors = {
        state_id: _policy_sufficient_vector(outcomes)
        for state_id, outcomes in left_by_state.items()
    }
    right_vectors = {
        state_id: _policy_sufficient_vector(right_by_state[state_id])
        for state_id in state_ids
    }
    left_point = _summarize_policy_outcomes(
        left_outcomes,
        policy_name=left_policy.name,
    )
    right_point = _summarize_policy_outcomes(
        right_outcomes,
        policy_name=right_policy.name,
    )
    numeric_metrics = sorted(
        name
        for name in set(left_point) & set(right_point)
        if name not in ("n_decisions", "marginal_accuracy_gain_per_tool_call")
        and isinstance(left_point[name], (int, float))
        and not isinstance(left_point[name], bool)
        and isinstance(right_point[name], (int, float))
        and not isinstance(right_point[name], bool)
    )
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {name: [] for name in numeric_metrics}
    vector_width = len(next(iter(left_vectors.values())))
    for _ in range(n_resamples):
        left_total = [0.0] * vector_width
        right_total = [0.0] * vector_width
        for state_id in rng.choices(state_ids, k=len(state_ids)):
            for index in range(vector_width):
                left_total[index] += left_vectors[state_id][index]
                right_total[index] += right_vectors[state_id][index]
        left_summary = _summarize_policy_vector(
            left_total,
            policy_name=left_policy.name,
        )
        right_summary = _summarize_policy_vector(
            right_total,
            policy_name=right_policy.name,
        )
        for name in numeric_metrics:
            samples[name].append(
                float(right_summary[name]) - float(left_summary[name])  # type: ignore[arg-type]
            )

    alpha = (1.0 - confidence) / 2.0
    return {
        "comparison": "right_minus_left",
        "resampling_unit": "state_id",
        "confidence": confidence,
        "n_resamples": n_resamples,
        "seed": seed,
        "lambda_cost": lambda_cost,
        "n_states": len(state_ids),
        "n_decisions": len(left_outcomes),
        "left_policy": left_policy.name,
        "right_policy": right_policy.name,
        "metrics": {
            name: {
                "estimate": float(right_point[name]) - float(left_point[name]),  # type: ignore[arg-type]
                "ci_low": _percentile(samples[name], alpha),
                "ci_high": _percentile(samples[name], 1.0 - alpha),
            }
            for name in numeric_metrics
        },
    }


def diagnostic_to_dict(diagnostic: EntropyDiagnostic) -> dict[str, object]:
    return asdict(diagnostic)


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _entropy_sufficient_vector(records: Sequence[ActionRecord]) -> tuple[float, ...]:
    """Return additive statistics needed by every entropy diagnostic metric."""

    zooms = [record for record in records if record.action_type == "ZOOM"]
    grouped = group_by_decision(records)
    entropy_deltas = [record.delta_entropy for record in zooms]
    success_deltas = [record.delta_success for record in zooms]
    mismatches = 0
    regret_sum = 0.0
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
        mismatches += int(entropy_top.action_id not in success_top_ids)
        regret_sum += best_success - entropy_top.delta_success
    return (
        float(len(zooms)),
        float(len(grouped)),
        float(sum(value > 0.0 for value in entropy_deltas)),
        float(sum(value > 0.0 for value in success_deltas)),
        float(
            sum(
                entropy > 0.0 and success < 0.0
                for entropy, success in zip(entropy_deltas, success_deltas)
            )
        ),
        float(
            sum(
                entropy > 0.0 and success <= 0.0
                for entropy, success in zip(entropy_deltas, success_deltas)
            )
        ),
        float(
            sum(
                entropy > 0.0 and success > 0.0
                for entropy, success in zip(entropy_deltas, success_deltas)
            )
        ),
        sum(entropy_deltas),
        sum(success_deltas),
        sum(value * value for value in entropy_deltas),
        sum(value * value for value in success_deltas),
        sum(left * right for left, right in zip(entropy_deltas, success_deltas)),
        float(mismatches),
        regret_sum,
    )


def _entropy_diagnostic_from_vector(vector: Sequence[float]) -> dict[str, float | int | None]:
    (
        n_zoom,
        n_decisions,
        confidence_gains,
        task_improvements,
        spurious_gains,
        nonbeneficial_gains,
        beneficial_confidence_gains,
        sum_entropy,
        sum_success,
        sum_entropy_squared,
        sum_success_squared,
        sum_cross,
        mismatches,
        regret_sum,
    ) = vector
    if n_zoom <= 0.0 or n_decisions <= 0.0:
        raise ValueError("entropy diagnostic requires ZOOM actions and decisions")
    centered_cross = sum_cross - sum_entropy * sum_success / n_zoom
    entropy_square = max(0.0, sum_entropy_squared - sum_entropy**2 / n_zoom)
    success_square = max(0.0, sum_success_squared - sum_success**2 / n_zoom)
    denominator = (entropy_square * success_square) ** 0.5
    return {
        "n_zoom_actions": int(n_zoom),
        "n_decisions": int(n_decisions),
        "confidence_gain_rate": confidence_gains / n_zoom,
        "task_improvement_rate": task_improvements / n_zoom,
        "spurious_confidence_gain_rate": spurious_gains / n_zoom,
        "nonbeneficial_confidence_gain_rate": nonbeneficial_gains / n_zoom,
        "confidence_gain_precision": (
            beneficial_confidence_gains / confidence_gains
            if confidence_gains > 0.0
            else None
        ),
        "entropy_success_pearson": (
            centered_cross / denominator if denominator > 0.0 else None
        ),
        "entropy_top1_mismatch_rate": mismatches / n_decisions,
        "mean_entropy_selection_regret": regret_sum / n_decisions,
    }


def bootstrap_entropy_diagnostic(
    records: Sequence[ActionRecord],
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, object]:
    """State-cluster bootstrap intervals for entropy/success diagnostics."""

    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    by_state: dict[str, list[ActionRecord]] = {}
    for record in records:
        by_state.setdefault(record.state_id, []).append(record)
    if not by_state:
        raise ValueError("bootstrap requires at least one state")
    state_ids = sorted(by_state)
    rng = random.Random(seed)
    state_vectors = {
        state_id: _entropy_sufficient_vector(state_records)
        for state_id, state_records in by_state.items()
    }
    vector_width = len(next(iter(state_vectors.values())))
    samples: dict[str, list[float]] = {}
    for _ in range(n_resamples):
        totals = [0.0] * vector_width
        for state_id in rng.choices(state_ids, k=len(state_ids)):
            vector = state_vectors[state_id]
            for index, component_value in enumerate(vector):
                totals[index] += component_value
        diagnostic = _entropy_diagnostic_from_vector(totals)
        for name, metric_value in diagnostic.items():
            if isinstance(metric_value, (int, float)) and not isinstance(
                metric_value, bool
            ):
                samples.setdefault(name, []).append(float(metric_value))
    point = diagnostic_to_dict(entropy_diagnostic(records))
    alpha = (1.0 - confidence) / 2.0
    intervals: dict[str, dict[str, float | int | None]] = {}
    for name, estimate in point.items():
        values = samples.get(name, [])
        intervals[name] = {
            "estimate": estimate if isinstance(estimate, (int, float)) else None,
            "ci_low": _percentile(values, alpha) if values else None,
            "ci_high": _percentile(values, 1.0 - alpha) if values else None,
        }
    return {
        "resampling_unit": "state_id",
        "confidence": confidence,
        "n_resamples": n_resamples,
        "seed": seed,
        "metrics": intervals,
    }
