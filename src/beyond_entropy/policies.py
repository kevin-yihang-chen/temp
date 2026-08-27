from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from statistics import mean
from typing import Protocol, Sequence

from .dataset import group_by_decision
from .model import LinearGainModel
from .schema import ActionRecord


@dataclass(frozen=True)
class PolicyDecision:
    selected: ActionRecord
    tool_calls: int
    visual_cost: float


class Policy(Protocol):
    name: str

    def select(self, siblings: Sequence[ActionRecord]) -> PolicyDecision: ...


def _partition(
    siblings: Sequence[ActionRecord],
) -> tuple[ActionRecord, list[ActionRecord]]:
    answers = [record for record in siblings if record.action_type == "ANSWER"]
    zooms = [record for record in siblings if record.action_type == "ZOOM"]
    if len(answers) != 1 or not zooms:
        raise ValueError("policy input requires one ANSWER and at least one ZOOM")
    return answers[0], zooms


def realized_policy_utility(decision: PolicyDecision, lambda_cost: float) -> float:
    gain = decision.selected.delta_success if decision.selected.action_type == "ZOOM" else 0.0
    return gain - lambda_cost * decision.visual_cost


class AnswerNowPolicy:
    name = "answer_now"

    def select(self, siblings: Sequence[ActionRecord]) -> PolicyDecision:
        answer, _ = _partition(siblings)
        return PolicyDecision(answer, tool_calls=0, visual_cost=0.0)


class RandomZoomPolicy:
    name = "random_zoom"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def select(self, siblings: Sequence[ActionRecord]) -> PolicyDecision:
        _, zooms = _partition(siblings)
        decision_id = f"{siblings[0].state_id}:{siblings[0].replicate_id}"
        digest = hashlib.sha256(f"{self.seed}:{decision_id}".encode()).digest()
        selected = random.Random(digest).choice(zooms)
        return PolicyDecision(selected, tool_calls=1, visual_cost=selected.tool_cost)


class FixedCenterZoomPolicy:
    name = "fixed_center_zoom"

    def select(self, siblings: Sequence[ActionRecord]) -> PolicyDecision:
        _, zooms = _partition(siblings)

        def center_distance(record: ActionRecord) -> float:
            bbox = record.candidate_bbox
            if bbox is None:
                raise ValueError("ZOOM record is missing candidate_bbox")
            return abs((bbox.x1 + bbox.x2) / 2.0 - 0.5) + abs(
                (bbox.y1 + bbox.y2) / 2.0 - 0.5
            )

        selected = min(
            zooms,
            key=center_distance,
        )
        return PolicyDecision(selected, tool_calls=1, visual_cost=selected.tool_cost)


class EntropySearchPolicy:
    """Post-action selection after paying to evaluate every candidate crop."""

    name = "entropy_search"

    def select(self, siblings: Sequence[ActionRecord]) -> PolicyDecision:
        _, zooms = _partition(siblings)
        selected = min(zooms, key=lambda record: (record.entropy_after, record.action_id))
        return PolicyDecision(
            selected,
            tool_calls=len(zooms),
            visual_cost=sum(record.tool_cost for record in zooms),
        )


class EntropyThresholdPolicy:
    """Stop before crop search when baseline entropy is below a tuned threshold."""

    name = "entropy_threshold"

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def select(self, siblings: Sequence[ActionRecord]) -> PolicyDecision:
        answer, _ = _partition(siblings)
        if answer.entropy_before < self.threshold:
            return PolicyDecision(answer, tool_calls=0, visual_cost=0.0)
        return EntropySearchPolicy().select(siblings)


class EntropyReductionThresholdPolicy:
    """Post-action stopping diagnostic; candidate evaluation cost is already sunk."""

    name = "entropy_reduction_threshold"

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def select(self, siblings: Sequence[ActionRecord]) -> PolicyDecision:
        answer, zooms = _partition(siblings)
        selected = max(zooms, key=lambda record: (record.delta_entropy, record.action_id))
        calls = len(zooms)
        total_cost = sum(record.tool_cost for record in zooms)
        if selected.delta_entropy < self.threshold:
            return PolicyDecision(answer, tool_calls=calls, visual_cost=total_cost)
        return PolicyDecision(selected, tool_calls=calls, visual_cost=total_cost)


class LearnedVOIPolicy:
    name = "learned_voi"

    def __init__(self, model: LinearGainModel, lambda_cost: float) -> None:
        if lambda_cost < 0.0:
            raise ValueError("lambda_cost must be non-negative")
        self.model = model
        self.lambda_cost = lambda_cost

    def select(self, siblings: Sequence[ActionRecord]) -> PolicyDecision:
        answer, zooms = _partition(siblings)
        scored = [
            (self.model.predict_gain(record) - self.lambda_cost * record.tool_cost, record)
            for record in zooms
        ]
        best_utility, selected = max(scored, key=lambda item: (item[0], item[1].action_id))
        if best_utility <= 0.0:
            return PolicyDecision(answer, tool_calls=0, visual_cost=0.0)
        return PolicyDecision(selected, tool_calls=1, visual_cost=selected.tool_cost)


class OracleVOIPolicy:
    """Diagnostic upper bound; it consumes labels and is not deployable."""

    name = "oracle_voi"

    def __init__(self, lambda_cost: float) -> None:
        self.lambda_cost = lambda_cost

    def select(self, siblings: Sequence[ActionRecord]) -> PolicyDecision:
        answer, zooms = _partition(siblings)
        selected = max(zooms, key=lambda record: (record.voi(self.lambda_cost), record.action_id))
        if selected.voi(self.lambda_cost) <= 0.0:
            return PolicyDecision(answer, tool_calls=0, visual_cost=0.0)
        return PolicyDecision(selected, tool_calls=1, visual_cost=selected.tool_cost)


def _threshold_candidates(values: Sequence[float]) -> list[float]:
    unique = sorted(set(values))
    if not unique:
        raise ValueError("threshold tuning requires observations")
    candidates = [unique[0] - 1e-9]
    candidates.extend((left + right) / 2.0 for left, right in zip(unique, unique[1:]))
    candidates.append(unique[-1] + 1e-9)
    return candidates


def _tune_threshold(
    records: Sequence[ActionRecord],
    *,
    lambda_cost: float,
    reduction: bool,
) -> float:
    grouped = group_by_decision(records)
    if reduction:
        values = [
            max(record.delta_entropy for record in siblings if record.action_type == "ZOOM")
            for siblings in grouped.values()
        ]
    else:
        values = [
            next(record.entropy_before for record in siblings if record.action_type == "ANSWER")
            for siblings in grouped.values()
        ]
    best_threshold = values[0]
    best_utility = float("-inf")
    for threshold in _threshold_candidates(values):
        policy: Policy
        if reduction:
            policy = EntropyReductionThresholdPolicy(threshold)
        else:
            policy = EntropyThresholdPolicy(threshold)
        utility = mean(
            realized_policy_utility(policy.select(siblings), lambda_cost)
            for siblings in grouped.values()
        )
        if utility > best_utility:
            best_utility = utility
            best_threshold = threshold
    return best_threshold


def tune_entropy_thresholds(
    records: Sequence[ActionRecord],
    *,
    lambda_cost: float,
) -> tuple[float, float]:
    """Tune both entropy stopping baselines on training/validation records."""

    return (
        _tune_threshold(records, lambda_cost=lambda_cost, reduction=False),
        _tune_threshold(records, lambda_cost=lambda_cost, reduction=True),
    )
