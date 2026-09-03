from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from statistics import mean
from typing import Any, Iterable, Sequence

from .dataset import group_by_decision, validate_sibling_groups
from .predictability_audit import BinaryToolOutcome
from .predictability_evaluation import (
    align_policy_outcomes,
    policy_metrics,
    select_score_threshold,
)
from .schema import ActionRecord


STRONG_BASELINE_NAMES = (
    "answer_now",
    "entropy_gate_fixed_visual_tool",
    "random_gate_fixed_visual_tool",
    "fixed_crop_with_matched_gate",
    "uniform_random_crop_expectation_with_matched_gate",
    "exhaustive_ug_entropy_search_charged_four_calls",
)
FROZEN_UG_ACTION_IDS = (
    "ug-grid-00",
    "ug-grid-01",
    "ug-grid-02",
    "ug-grid-03",
)


@dataclass(frozen=True)
class FrozenPolicyTrace:
    name: str
    outcomes: tuple[BinaryToolOutcome, ...]
    calls: tuple[bool, ...]

    def __post_init__(self) -> None:
        if self.name not in STRONG_BASELINE_NAMES:
            raise ValueError(f"unregistered strong baseline: {self.name!r}")
        if not self.outcomes or len(self.outcomes) != len(self.calls):
            raise ValueError(
                "policy trace requires aligned non-empty outcomes and calls"
            )
        decision_ids = [item.decision_id for item in self.outcomes]
        if len(set(decision_ids)) != len(decision_ids):
            raise ValueError("policy trace decision IDs must be unique")
        if not all(isinstance(call, bool) for call in self.calls):
            raise ValueError("policy trace calls must contain booleans")

    def metrics(self, *, lambda_cost: float) -> dict[str, float | int | None]:
        return policy_metrics(self.outcomes, self.calls, lambda_cost=lambda_cost)


@dataclass(frozen=True)
class FrozenStrongBaselinePolicy:
    """All validation-selected choices needed to evaluate strong baselines."""

    lambda_cost: float
    random_gate_seed: int
    action_ids: tuple[str, ...]
    entropy_gate_threshold: float
    random_gate_threshold: float
    fixed_crop_action_id: str
    strongest_name: str
    validation_traces: tuple[FrozenPolicyTrace, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.lambda_cost) or self.lambda_cost < 0.0:
            raise ValueError("lambda_cost must be finite and non-negative")
        if not isinstance(self.random_gate_seed, int):
            raise ValueError("random_gate_seed must be an integer")
        if len(self.action_ids) != 4 or len(set(self.action_ids)) != 4:
            raise ValueError("strong baseline policy requires four unique crop actions")
        if tuple(sorted(self.action_ids)) != self.action_ids:
            raise ValueError("crop action IDs must use ascending frozen order")
        if self.fixed_crop_action_id not in self.action_ids:
            raise ValueError("fixed crop action is outside the frozen action bank")
        if not all(
            math.isfinite(value)
            for value in (self.entropy_gate_threshold, self.random_gate_threshold)
        ):
            raise ValueError("strong baseline thresholds must be finite")
        traces = _trace_map(self.validation_traces)
        if tuple(traces) != STRONG_BASELINE_NAMES:
            raise ValueError("validation traces must contain the frozen baseline order")
        _validate_trace_alignment(self.validation_traces)
        selected = _select_strongest_name(
            self.validation_traces, lambda_cost=self.lambda_cost
        )
        if self.strongest_name != selected:
            raise ValueError("strongest baseline is not the validation-selected policy")


def _trace_map(
    traces: Sequence[FrozenPolicyTrace],
) -> dict[str, FrozenPolicyTrace]:
    result = {trace.name: trace for trace in traces}
    if len(result) != len(traces):
        raise ValueError("strong baseline trace names must be unique")
    return result


def _validate_trace_alignment(traces: Sequence[FrozenPolicyTrace]) -> None:
    if not traces:
        raise ValueError("strong baseline traces must be non-empty")
    reference = traces[0].outcomes
    for trace in traces[1:]:
        align_policy_outcomes(reference, trace.outcomes, trace.calls)


def _select_strongest_name(
    traces: Sequence[FrozenPolicyTrace], *, lambda_cost: float
) -> str:
    metrics = {trace.name: trace.metrics(lambda_cost=lambda_cost) for trace in traces}

    def key(name: str) -> tuple[float, float, float, str]:
        values = metrics[name]
        utility = values["incremental_utility"]
        cost = values["cost"]
        call_rate = values["call_rate"]
        if not isinstance(utility, (int, float)):
            raise AssertionError("strong baseline utility is missing")
        if not isinstance(cost, (int, float)):
            raise AssertionError("strong baseline cost is missing")
        if not isinstance(call_rate, (int, float)):
            raise AssertionError("strong baseline selection metrics are missing")
        return (-float(utility), float(cost), float(call_rate), name)

    return min((trace.name for trace in traces), key=key)


def _materialize_action_bank(
    records: Iterable[ActionRecord],
    *,
    expected_action_ids: tuple[str, ...] | None = None,
) -> tuple[list[tuple[tuple[str, str], list[ActionRecord]]], tuple[str, ...]]:
    materialized = list(records)
    validate_sibling_groups(materialized)
    grouped = sorted(group_by_decision(materialized).items())
    action_ids: tuple[str, ...] | None = None
    for decision_id, siblings in grouped:
        zooms = sorted(
            (record for record in siblings if record.action_type == "ZOOM"),
            key=lambda record: record.action_id,
        )
        current = tuple(record.action_id for record in zooms)
        if len(current) != 4 or len(set(current)) != 4:
            raise ValueError(
                f"decision {decision_id!r} must contain four unique ZOOM actions"
            )
        if action_ids is None:
            action_ids = current
        elif current != action_ids:
            raise ValueError("the four-crop action bank changes across decisions")
    if action_ids is None:
        raise ValueError("strong baseline action bank is empty")
    if action_ids != FROZEN_UG_ACTION_IDS:
        raise ValueError(
            f"strong baseline requires frozen UG action IDs {FROZEN_UG_ACTION_IDS!r}"
        )
    if expected_action_ids is not None and action_ids != expected_action_ids:
        raise ValueError("evaluation action bank differs from validation")
    return grouped, action_ids


def _answer(siblings: Sequence[ActionRecord]) -> ActionRecord:
    answers = [record for record in siblings if record.action_type == "ANSWER"]
    if len(answers) != 1:
        raise ValueError("strong baseline decision requires exactly one ANSWER")
    return answers[0]


def _make_outcome(
    answer: ActionRecord,
    *,
    selected_action_id: str,
    y_tool: float,
    tool_cost: float,
    tool_calls: int,
) -> BinaryToolOutcome:
    return BinaryToolOutcome(
        state_id=answer.state_id,
        replicate_id=answer.replicate_id,
        image_id=answer.image_id,
        source_id=answer.source_id,
        selected_action_id=selected_action_id,
        y0=answer.correct_before,
        y_tool=y_tool,
        tool_cost=tool_cost,
        tool_calls=tool_calls,
    )


def _action_outcomes(
    grouped: Sequence[tuple[tuple[str, str], list[ActionRecord]]],
    *,
    action_id: str,
) -> tuple[BinaryToolOutcome, ...]:
    outcomes: list[BinaryToolOutcome] = []
    for _, siblings in grouped:
        answer = _answer(siblings)
        matches = [record for record in siblings if record.action_id == action_id]
        if len(matches) != 1 or matches[0].action_type != "ZOOM":
            raise ValueError(f"fixed crop {action_id!r} is missing from a decision")
        selected = matches[0]
        outcomes.append(
            _make_outcome(
                answer,
                selected_action_id=selected.action_id,
                y_tool=selected.correct_after,
                tool_cost=selected.tool_cost,
                tool_calls=1,
            )
        )
    return tuple(outcomes)


def _uniform_random_outcomes(
    grouped: Sequence[tuple[tuple[str, str], list[ActionRecord]]],
) -> tuple[BinaryToolOutcome, ...]:
    outcomes: list[BinaryToolOutcome] = []
    for _, siblings in grouped:
        answer = _answer(siblings)
        zooms = [record for record in siblings if record.action_type == "ZOOM"]
        outcomes.append(
            _make_outcome(
                answer,
                selected_action_id="uniform-random-crop-expectation",
                y_tool=mean(record.correct_after for record in zooms),
                tool_cost=mean(record.tool_cost for record in zooms),
                tool_calls=1,
            )
        )
    return tuple(outcomes)


def _exhaustive_outcomes(
    grouped: Sequence[tuple[tuple[str, str], list[ActionRecord]]],
) -> tuple[BinaryToolOutcome, ...]:
    outcomes: list[BinaryToolOutcome] = []
    for _, siblings in grouped:
        answer = _answer(siblings)
        zooms = [record for record in siblings if record.action_type == "ZOOM"]
        selected = min(
            zooms, key=lambda record: (record.entropy_after, record.action_id)
        )
        outcomes.append(
            _make_outcome(
                answer,
                selected_action_id=selected.action_id,
                y_tool=selected.correct_after,
                tool_cost=sum(record.tool_cost for record in zooms),
                tool_calls=len(zooms),
            )
        )
    return tuple(outcomes)


def _entropy_scores(
    grouped: Sequence[tuple[tuple[str, str], list[ActionRecord]]],
) -> list[float]:
    return [float(_answer(siblings).entropy_before) for _, siblings in grouped]


def random_gate_score(state_id: str, replicate_id: str, *, seed: int) -> float:
    """Return a stable, outcome-free uniform score in [0, 1)."""

    payload = f"predictability-random-gate-v1:{seed}:{state_id}:{replicate_id}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _random_scores(outcomes: Sequence[BinaryToolOutcome], *, seed: int) -> list[float]:
    return [
        random_gate_score(item.state_id, item.replicate_id, seed=seed)
        for item in outcomes
    ]


def _calls(scores: Sequence[float], *, threshold: float) -> tuple[bool, ...]:
    return tuple(float(score) >= threshold for score in scores)


def _build_traces(
    grouped: Sequence[tuple[tuple[str, str], list[ActionRecord]]],
    *,
    entropy_gate_threshold: float,
    random_gate_threshold: float,
    random_gate_seed: int,
    fixed_crop_action_id: str,
) -> tuple[FrozenPolicyTrace, ...]:
    exhaustive = _exhaustive_outcomes(grouped)
    entropy_calls = _calls(_entropy_scores(grouped), threshold=entropy_gate_threshold)
    random_calls = _calls(
        _random_scores(exhaustive, seed=random_gate_seed),
        threshold=random_gate_threshold,
    )
    never = (False,) * len(exhaustive)
    always = (True,) * len(exhaustive)
    return (
        FrozenPolicyTrace("answer_now", exhaustive, never),
        FrozenPolicyTrace("entropy_gate_fixed_visual_tool", exhaustive, entropy_calls),
        FrozenPolicyTrace("random_gate_fixed_visual_tool", exhaustive, random_calls),
        FrozenPolicyTrace(
            "fixed_crop_with_matched_gate",
            _action_outcomes(grouped, action_id=fixed_crop_action_id),
            entropy_calls,
        ),
        FrozenPolicyTrace(
            "uniform_random_crop_expectation_with_matched_gate",
            _uniform_random_outcomes(grouped),
            entropy_calls,
        ),
        FrozenPolicyTrace(
            "exhaustive_ug_entropy_search_charged_four_calls",
            exhaustive,
            always,
        ),
    )


def fit_strong_baselines(
    validation_records: Iterable[ActionRecord],
    *,
    lambda_cost: float,
    random_gate_seed: int,
) -> FrozenStrongBaselinePolicy:
    """Freeze all baseline choices using validation records only."""

    grouped, action_ids = _materialize_action_bank(validation_records)
    exhaustive = _exhaustive_outcomes(grouped)
    entropy_scores = _entropy_scores(grouped)
    entropy_selection = select_score_threshold(
        exhaustive, entropy_scores, lambda_cost=lambda_cost
    )
    random_scores = _random_scores(exhaustive, seed=random_gate_seed)
    random_selection = select_score_threshold(
        exhaustive, random_scores, lambda_cost=lambda_cost
    )
    entropy_threshold = float(entropy_selection["threshold"])
    random_threshold = float(random_selection["threshold"])
    entropy_calls = _calls(entropy_scores, threshold=entropy_threshold)

    fixed_candidates = {
        action_id: FrozenPolicyTrace(
            "fixed_crop_with_matched_gate",
            _action_outcomes(grouped, action_id=action_id),
            entropy_calls,
        )
        for action_id in action_ids
    }

    def fixed_key(action_id: str) -> tuple[float, float, float, str]:
        metrics = fixed_candidates[action_id].metrics(lambda_cost=lambda_cost)
        utility = metrics["incremental_utility"]
        cost = metrics["cost"]
        call_rate = metrics["call_rate"]
        if not isinstance(utility, (int, float)):
            raise AssertionError("fixed crop utility is missing")
        if not isinstance(cost, (int, float)):
            raise AssertionError("fixed crop cost is missing")
        if not isinstance(call_rate, (int, float)):
            raise AssertionError("fixed crop selection metrics are missing")
        return (-float(utility), float(cost), float(call_rate), action_id)

    fixed_action = min(action_ids, key=fixed_key)
    validation_traces = _build_traces(
        grouped,
        entropy_gate_threshold=entropy_threshold,
        random_gate_threshold=random_threshold,
        random_gate_seed=random_gate_seed,
        fixed_crop_action_id=fixed_action,
    )
    strongest = _select_strongest_name(validation_traces, lambda_cost=lambda_cost)
    return FrozenStrongBaselinePolicy(
        lambda_cost=lambda_cost,
        random_gate_seed=random_gate_seed,
        action_ids=action_ids,
        entropy_gate_threshold=entropy_threshold,
        random_gate_threshold=random_threshold,
        fixed_crop_action_id=fixed_action,
        strongest_name=strongest,
        validation_traces=validation_traces,
    )


def apply_strong_baselines(
    frozen: FrozenStrongBaselinePolicy,
    records: Iterable[ActionRecord],
) -> tuple[FrozenPolicyTrace, ...]:
    """Apply choices without using evaluation labels to select a policy."""

    grouped, _ = _materialize_action_bank(
        records, expected_action_ids=frozen.action_ids
    )
    return _build_traces(
        grouped,
        entropy_gate_threshold=frozen.entropy_gate_threshold,
        random_gate_threshold=frozen.random_gate_threshold,
        random_gate_seed=frozen.random_gate_seed,
        fixed_crop_action_id=frozen.fixed_crop_action_id,
    )


def strong_baseline_report(
    frozen: FrozenStrongBaselinePolicy,
    test_traces: Sequence[FrozenPolicyTrace],
) -> dict[str, Any]:
    test = _trace_map(test_traces)
    if tuple(test) != STRONG_BASELINE_NAMES:
        raise ValueError("test traces must contain the frozen baseline order")
    _validate_trace_alignment(test_traces)
    validation = _trace_map(frozen.validation_traces)
    return {
        "schema": "predictability_strong_baselines_v1",
        "selection_role": "validation_only",
        "lambda_cost": frozen.lambda_cost,
        "action_ids": list(frozen.action_ids),
        "random_gate": {
            "algorithm": "sha256_first_64_bits_uniform_v1",
            "seed": frozen.random_gate_seed,
            "threshold": frozen.random_gate_threshold,
        },
        "entropy_gate": {
            "direction": "call_if_entropy_before_greater_than_or_equal",
            "threshold": frozen.entropy_gate_threshold,
        },
        "matched_gate": "entropy_gate_fixed_visual_tool",
        "fixed_crop_action_id": frozen.fixed_crop_action_id,
        "strongest_baseline": frozen.strongest_name,
        "strongest_selection_tie_break": (
            "validation_source_balanced_utility_then_lower_cost_then_lower_call_rate_"
            "then_ascending_name"
        ),
        "validation": {
            name: validation[name].metrics(lambda_cost=frozen.lambda_cost)
            for name in STRONG_BASELINE_NAMES
        },
        "test": {
            name: test[name].metrics(lambda_cost=frozen.lambda_cost)
            for name in STRONG_BASELINE_NAMES
        },
    }


def trace_by_name(traces: Sequence[FrozenPolicyTrace], name: str) -> FrozenPolicyTrace:
    return _trace_map(traces)[name]


def validate_fixed_tool_outcomes(
    expected: Sequence[BinaryToolOutcome],
    actual: Sequence[BinaryToolOutcome],
) -> None:
    """Require feature labels to equal the exhaustive sibling collapse."""

    expected_ids = [item.decision_id for item in expected]
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("fixed-tool feature outcomes contain duplicate decision IDs")
    indexed = {item.decision_id: item for item in actual}
    if len(indexed) != len(actual):
        raise ValueError("fixed-tool outcomes contain duplicate decision IDs")
    if {item.decision_id for item in expected} != set(indexed):
        raise ValueError("fixed-tool feature and sibling coverage differ")
    for item in expected:
        sibling = indexed[item.decision_id]
        if item != sibling:
            raise ValueError(
                f"fixed-tool feature label differs from siblings for {item.decision_id!r}"
            )
