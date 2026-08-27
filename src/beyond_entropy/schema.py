from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping


ActionType = Literal["ANSWER", "ZOOM"]


@dataclass(frozen=True)
class BBox:
    """Normalized xyxy bounding box."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        values = (self.x1, self.y1, self.x2, self.y2)
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"bbox coordinates must be in [0, 1], got {values}")
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ValueError(f"bbox must have positive area, got {values}")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @classmethod
    def from_value(cls, value: object) -> "BBox | None":
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError("candidate_bbox must be null or a four-element xyxy list")
        return cls(*(float(item) for item in value))

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(frozen=True)
class ActionRecord:
    """One sibling action rollout originating from a shared visual-agent state.

    Only ``pre_action_features`` may be consumed by the learned value model.
    Fields ending in ``_after`` are counterfactual outcomes and are labels or
    diagnostics, never pre-action inputs.
    """

    state_id: str
    image_id: str
    source_id: str
    question: str
    original_image: str
    replicate_id: str
    generation_seed: int | None
    action_id: str
    action_type: ActionType
    candidate_bbox: BBox | None
    entropy_before: float
    entropy_after: float
    answer_before: str
    answer_after: str
    correct_before: float
    correct_after: float
    tool_cost: float
    pre_action_features: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not all((self.state_id, self.image_id, self.source_id, self.replicate_id, self.action_id)):
            raise ValueError(
                "state_id, image_id, source_id, replicate_id, and action_id must be non-empty"
            )
        if self.action_type not in ("ANSWER", "ZOOM"):
            raise ValueError(f"unsupported action_type: {self.action_type}")
        if self.action_type == "ANSWER" and self.candidate_bbox is not None:
            raise ValueError("ANSWER must not have a candidate_bbox")
        if self.action_type == "ZOOM" and self.candidate_bbox is None:
            raise ValueError("ZOOM requires a candidate_bbox")
        if self.entropy_before < 0.0 or self.entropy_after < 0.0:
            raise ValueError("entropy values must be non-negative")
        if not 0.0 <= self.correct_before <= 1.0:
            raise ValueError("correct_before must be in [0, 1]")
        if not 0.0 <= self.correct_after <= 1.0:
            raise ValueError("correct_after must be in [0, 1]")
        if self.tool_cost < 0.0:
            raise ValueError("tool_cost must be non-negative")
        if self.action_type == "ANSWER" and self.tool_cost != 0.0:
            raise ValueError("ANSWER must have zero tool_cost")
        for name, value in self.pre_action_features.items():
            if not isinstance(name, str) or not name:
                raise ValueError("pre_action_features keys must be non-empty strings")
            if not isinstance(value, (int, float)):
                raise ValueError(f"pre-action feature {name!r} must be numeric")

    @property
    def delta_entropy(self) -> float:
        return self.entropy_before - self.entropy_after

    @property
    def delta_success(self) -> float:
        return self.correct_after - self.correct_before

    def voi(self, lambda_cost: float) -> float:
        return self.delta_success - lambda_cost * self.tool_cost

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ActionRecord":
        return cls(
            state_id=str(value["state_id"]),
            image_id=str(value.get("image_id", value["original_image"])),
            source_id=str(value.get("source_id", value.get("image_id", value["original_image"]))),
            question=str(value["question"]),
            original_image=str(value["original_image"]),
            replicate_id=str(value.get("replicate_id", "replicate-000")),
            generation_seed=(
                None if value.get("generation_seed") is None else int(value["generation_seed"])
            ),
            action_id=str(value["action_id"]),
            action_type=str(value["action_type"]),  # type: ignore[arg-type]
            candidate_bbox=BBox.from_value(value.get("candidate_bbox")),
            entropy_before=float(value["entropy_before"]),
            entropy_after=float(value["entropy_after"]),
            answer_before=str(value["answer_before"]),
            answer_after=str(value["answer_after"]),
            correct_before=float(value["correct_before"]),
            correct_after=float(value["correct_after"]),
            tool_cost=float(value["tool_cost"]),
            pre_action_features={
                str(key): float(item)
                for key, item in dict(value.get("pre_action_features", {})).items()
            },
            metadata=dict(value.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["candidate_bbox"] = (
            None if self.candidate_bbox is None else self.candidate_bbox.to_list()
        )
        result["pre_action_features"] = dict(self.pre_action_features)
        result["metadata"] = dict(self.metadata)
        return result
