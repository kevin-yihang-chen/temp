"""Finite, executable spatial actions; no model-generated tool syntax."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .rollout import AgentState, InferenceRequest, VisualObservation
from .schema import ActionRecord, BBox


@dataclass(frozen=True)
class SpatialAction:
    index: int
    action_id: str
    bbox: BBox | None
    visual_cost: float

    def __post_init__(self) -> None:
        if self.index < 0 or not self.action_id:
            raise ValueError("invalid action identity")
        if not math.isfinite(self.visual_cost) or self.visual_cost < 0:
            raise ValueError("action cost must be finite and nonnegative")
        if self.index == 0:
            if self.bbox is not None or self.visual_cost != 0:
                raise ValueError("action 0 must be zero-cost ANSWER")
        elif self.bbox is None:
            raise ValueError("ZOOM requires a valid BBox")

    @property
    def name(self) -> str:
        return "ANSWER" if self.index == 0 else f"ZOOM_{self.index}"


@dataclass(frozen=True)
class SpatialActionSpace:
    actions: tuple[SpatialAction, ...]

    def __post_init__(self) -> None:
        if not 2 <= len(self.actions) <= 5:
            raise ValueError("MVP requires ANSWER and 1..4 ZOOM actions")
        if tuple(a.index for a in self.actions) != tuple(range(len(self.actions))):
            raise ValueError("action indices must be contiguous, ANSWER first")
        if len({a.action_id for a in self.actions}) != len(self.actions):
            raise ValueError("duplicate action ID")
        zooms = self.actions[1:]
        if tuple(a.action_id for a in zooms) != tuple(sorted(a.action_id for a in zooms)):
            raise ValueError("ZOOM mapping must use stable sorted action IDs")

    @classmethod
    def from_siblings(cls, siblings: Iterable[ActionRecord]) -> SpatialActionSpace:
        rows = list(siblings)
        answers = [r for r in rows if r.action_type == "ANSWER"]
        if len(answers) != 1:
            raise ValueError("exactly one ANSWER required")
        ordered = answers + sorted(
            (r for r in rows if r.action_type == "ZOOM"), key=lambda r: r.action_id
        )
        return cls(tuple(
            SpatialAction(i, r.action_id, r.candidate_bbox, r.tool_cost)
            for i, r in enumerate(ordered)
        ))

    def select(self, gains: Iterable[float], *, lambda_cost: float) -> int:
        values = tuple(gains)
        if len(values) != len(self.actions) or not all(math.isfinite(x) for x in values):
            raise ValueError("one finite predicted gain per action required")
        if not math.isfinite(lambda_cost) or lambda_cost < 0:
            raise ValueError("lambda must be finite and nonnegative")
        # Smallest index wins ties, so ANSWER wins zero-net-gain ties.
        return max(range(len(values)), key=lambda i: (
            values[i] - lambda_cost * self.actions[i].visual_cost, -i
        ))

    def request(
        self, index: int, state: AgentState, *, generation_seed: int | None
    ) -> InferenceRequest:
        if type(index) is not int or not 0 <= index < len(self.actions):
            raise ValueError("action index outside the legal support")
        action = self.actions[index]
        original = VisualObservation("ORIGINAL", state.image_path, "original", None)
        observations = (original,) if index == 0 else (
            original, VisualObservation("ZOOM", state.image_path, action.action_id, action.bbox)
        )
        return InferenceRequest(state, observations, generation_seed)
