"""Schema for paired STOP/CONTINUE rollouts from a shared visual prefix."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from .schema import BBox


def _probability(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


@dataclass(frozen=True)
class AcquiredObservationSpec:
    """One observation already present at the stopping decision."""

    action_id: str
    bbox: BBox
    visual_cost: float = 1.0

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("acquired action_id must be non-empty")
        if not math.isfinite(self.visual_cost) or self.visual_cost < 0.0:
            raise ValueError("acquired visual_cost must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "bbox": self.bbox.to_list(),
            "visual_cost": self.visual_cost,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AcquiredObservationSpec":
        bbox = BBox.from_value(value.get("bbox"))
        if bbox is None:
            raise ValueError("acquired observation requires a bbox")
        return cls(str(value["action_id"]), bbox, float(value["visual_cost"]))


@dataclass(frozen=True)
class SequentialRolloutRecord:
    """Paired outcomes for STOP and one fixed additional acquisition.

    The STOP and CONTINUE branches are stored together so they cannot silently
    drift in seed, prefix, scorer, or generation configuration.  Outcome fields
    are labels/diagnostics and must never be copied into critic inputs.
    """

    state_id: str
    image_id: str
    source_id: str
    question: str
    original_image: str
    step_index: int
    acquired_observations: tuple[AcquiredObservationSpec, ...]
    proposed_action_id: str
    proposed_bbox: BBox
    proposed_visual_cost: float
    replicate_id: str
    generation_seed: int | None
    stop_answer: str
    stop_correct: float
    stop_entropy: float
    stop_max_probability: float
    stop_top1_top2_margin: float
    continue_answer: str
    continue_correct: float
    continue_entropy: float
    continue_max_probability: float
    continue_top1_top2_margin: float
    stop_backend: Mapping[str, Any] = field(default_factory=dict)
    continue_backend: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        identities = (
            self.state_id,
            self.image_id,
            self.source_id,
            self.question,
            self.original_image,
            self.proposed_action_id,
            self.replicate_id,
        )
        if any(not item for item in identities):
            raise ValueError("sequential rollout identities must be non-empty")
        if self.step_index < 1:
            raise ValueError("a stopping decision requires at least one acquired step")
        if len(self.acquired_observations) != self.step_index:
            raise ValueError("step_index must equal acquired observation count")
        action_ids = [item.action_id for item in self.acquired_observations]
        if self.proposed_action_id in action_ids or len(set(action_ids)) != len(action_ids):
            raise ValueError("acquired and proposed action IDs must be distinct")
        if not math.isfinite(self.proposed_visual_cost) or self.proposed_visual_cost < 0:
            raise ValueError("proposed visual cost must be finite and non-negative")
        for name in ("stop_correct", "continue_correct"):
            _probability(getattr(self, name), name=name)
        for name in ("stop_entropy", "continue_entropy"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in (
            "stop_max_probability",
            "stop_top1_top2_margin",
            "continue_max_probability",
            "continue_top1_top2_margin",
        ):
            _probability(getattr(self, name), name=name)

    @property
    def acquired_visual_cost(self) -> float:
        return sum(item.visual_cost for item in self.acquired_observations)

    @property
    def stop_total_visual_cost(self) -> float:
        return self.acquired_visual_cost

    @property
    def continue_total_visual_cost(self) -> float:
        return self.acquired_visual_cost + self.proposed_visual_cost

    @property
    def delta_success(self) -> float:
        return self.continue_correct - self.stop_correct

    @property
    def delta_entropy(self) -> float:
        return self.stop_entropy - self.continue_entropy

    def incremental_utility(self, lambda_cost: float) -> float:
        if not math.isfinite(lambda_cost) or lambda_cost < 0.0:
            raise ValueError("lambda_cost must be finite and non-negative")
        return self.delta_success - lambda_cost * self.proposed_visual_cost

    @property
    def decision_id(self) -> tuple[str, str]:
        return self.state_id, self.replicate_id

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["acquired_observations"] = [
            item.to_dict() for item in self.acquired_observations
        ]
        result["proposed_bbox"] = self.proposed_bbox.to_list()
        result["stop_backend"] = dict(self.stop_backend)
        result["continue_backend"] = dict(self.continue_backend)
        result["metadata"] = dict(self.metadata)
        result["derived"] = {
            "delta_success": self.delta_success,
            "delta_entropy": self.delta_entropy,
            "cost_independent_gain": self.delta_success,
            "stop_total_visual_cost": self.stop_total_visual_cost,
            "continue_total_visual_cost": self.continue_total_visual_cost,
        }
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SequentialRolloutRecord":
        bbox = BBox.from_value(value.get("proposed_bbox"))
        if bbox is None:
            raise ValueError("CONTINUE proposal requires a bbox")
        return cls(
            state_id=str(value["state_id"]),
            image_id=str(value["image_id"]),
            source_id=str(value["source_id"]),
            question=str(value["question"]),
            original_image=str(value["original_image"]),
            step_index=int(value["step_index"]),
            acquired_observations=tuple(
                AcquiredObservationSpec.from_dict(item)
                for item in value["acquired_observations"]
            ),
            proposed_action_id=str(value["proposed_action_id"]),
            proposed_bbox=bbox,
            proposed_visual_cost=float(value["proposed_visual_cost"]),
            replicate_id=str(value["replicate_id"]),
            generation_seed=(
                None
                if value.get("generation_seed") is None
                else int(value["generation_seed"])
            ),
            stop_answer=str(value["stop_answer"]),
            stop_correct=float(value["stop_correct"]),
            stop_entropy=float(value["stop_entropy"]),
            stop_max_probability=float(value["stop_max_probability"]),
            stop_top1_top2_margin=float(value["stop_top1_top2_margin"]),
            continue_answer=str(value["continue_answer"]),
            continue_correct=float(value["continue_correct"]),
            continue_entropy=float(value["continue_entropy"]),
            continue_max_probability=float(value["continue_max_probability"]),
            continue_top1_top2_margin=float(value["continue_top1_top2_margin"]),
            stop_backend=dict(value.get("stop_backend", {})),
            continue_backend=dict(value.get("continue_backend", {})),
            metadata=dict(value.get("metadata", {})),
        )
