from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from .schema import ActionRecord


_FORBIDDEN_FEATURE_PARTS = (
    "answer_after",
    "correct_after",
    "entropy_after",
    "delta_",
    "reward",
    "target",
    "label",
)


@dataclass(frozen=True)
class FeatureEncoder:
    """Encodes information available before a candidate action is executed."""

    custom_names: tuple[str, ...]

    BASE_NAMES = (
        "entropy_before",
        "tool_cost",
        "bbox_x_center",
        "bbox_y_center",
        "bbox_width",
        "bbox_height",
        "bbox_area",
        "bbox_center_distance",
    )

    @classmethod
    def fit(cls, records: Iterable[ActionRecord]) -> "FeatureEncoder":
        names = sorted(
            {
                name
                for record in records
                if record.action_type == "ZOOM"
                for name in record.pre_action_features
            }
        )
        for name in names:
            lowered = name.lower()
            if any(part in lowered for part in _FORBIDDEN_FEATURE_PARTS):
                raise ValueError(
                    f"pre_action_features contains likely post-action leakage: {name!r}"
                )
        return cls(tuple(names))

    @property
    def names(self) -> tuple[str, ...]:
        return self.BASE_NAMES + tuple(f"pre:{name}" for name in self.custom_names)

    def transform_one(self, record: ActionRecord) -> list[float]:
        if record.action_type != "ZOOM" or record.candidate_bbox is None:
            raise ValueError("value features are defined only for ZOOM actions")
        bbox = record.candidate_bbox
        x_center = (bbox.x1 + bbox.x2) / 2.0
        y_center = (bbox.y1 + bbox.y2) / 2.0
        center_distance = math.hypot(x_center - 0.5, y_center - 0.5)
        base = [
            record.entropy_before,
            record.tool_cost,
            x_center,
            y_center,
            bbox.width,
            bbox.height,
            bbox.area,
            center_distance,
        ]
        return base + [float(record.pre_action_features.get(name, 0.0)) for name in self.custom_names]

    def transform(self, records: Iterable[ActionRecord]) -> list[list[float]]:
        return [self.transform_one(record) for record in records]

    def to_dict(self) -> dict[str, object]:
        return {"custom_names": list(self.custom_names)}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "FeatureEncoder":
        names = value.get("custom_names")
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise ValueError("invalid serialized FeatureEncoder")
        return cls(tuple(names))


def select_zooms(records: Sequence[ActionRecord]) -> list[ActionRecord]:
    return [record for record in records if record.action_type == "ZOOM"]
