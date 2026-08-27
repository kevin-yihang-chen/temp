from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Protocol, Sequence

from .model import LinearValueModel
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
        state_id = siblings[0].state_id
        digest = hashlib.sha256(f"{self.seed}:{state_id}".encode()).digest()
        selected = random.Random(digest).choice(zooms)
        return PolicyDecision(selected, tool_calls=1, visual_cost=selected.tool_cost)


class FixedCenterZoomPolicy:
    name = "fixed_center_zoom"

    def select(self, siblings: Sequence[ActionRecord]) -> PolicyDecision:
        _, zooms = _partition(siblings)
        selected = min(
            zooms,
            key=lambda record: (
                abs((record.candidate_bbox.x1 + record.candidate_bbox.x2) / 2.0 - 0.5)
                + abs((record.candidate_bbox.y1 + record.candidate_bbox.y2) / 2.0 - 0.5)
            ),
        )
        return PolicyDecision(selected, tool_calls=1, visual_cost=selected.tool_cost)


class EntropySearchPolicy:
    """UG-style post-action selection after evaluating every candidate crop."""

    name = "entropy_search"

    def select(self, siblings: Sequence[ActionRecord]) -> PolicyDecision:
        _, zooms = _partition(siblings)
        selected = min(zooms, key=lambda record: (record.entropy_after, record.action_id))
        return PolicyDecision(
            selected,
            tool_calls=len(zooms),
            visual_cost=sum(record.tool_cost for record in zooms),
        )


class LearnedVOIPolicy:
    name = "learned_voi"

    def __init__(self, model: LinearValueModel) -> None:
        self.model = model

    def select(self, siblings: Sequence[ActionRecord]) -> PolicyDecision:
        answer, zooms = _partition(siblings)
        selected = max(zooms, key=lambda record: (self.model.predict(record), record.action_id))
        if self.model.predict(selected) <= 0.0:
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
