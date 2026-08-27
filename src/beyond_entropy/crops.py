from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .rollout import ActionSpec, AgentState
from .schema import BBox


def _axis_positions(length: int, box_size: int, stride: int) -> list[int]:
    positions = list(range(0, length - box_size + 1, stride))
    if not positions:
        return [0]
    final_position = length - box_size
    if positions[-1] != final_position:
        positions.append(final_position)
    return positions


def ug_grid_boxes(
    width: int,
    height: int,
    *,
    visual_crop_ratio: float = 2.0,
) -> list[BBox]:
    """Reproduce the 50%-overlap square grid used by UG visual search.

    UG sets the crop side to ``min(width, height) / visual_crop_ratio`` and the
    stride to half that side.  Pixel boxes are returned as normalized xyxy
    coordinates so they can be stored independently of the image loader.
    """

    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if visual_crop_ratio <= 0.0:
        raise ValueError("visual_crop_ratio must be positive")
    box_size = int(min(width, height) / visual_crop_ratio)
    if box_size < 2:
        raise ValueError("visual_crop_ratio produces a crop smaller than two pixels")
    stride = max(1, box_size // 2)
    x_positions = _axis_positions(width, box_size, stride)
    y_positions = _axis_positions(height, box_size, stride)
    return [
        BBox(
            x / width,
            y / height,
            (x + box_size) / width,
            (y + box_size) / height,
        )
        for y in y_positions
        for x in x_positions
    ]


def _anchor_grid(count: int) -> list[tuple[float, float]]:
    columns = math.ceil(math.sqrt(count))
    rows = math.ceil(count / columns)
    anchors: list[tuple[float, float]] = []
    for row in range(rows):
        for column in range(columns):
            if len(anchors) == count:
                return anchors
            anchors.append(((column + 0.5) / columns, (row + 0.5) / rows))
    return anchors


def spatially_balanced_subset(boxes: Sequence[BBox], count: int) -> list[BBox]:
    """Select a deterministic, spatially spread subset of an UG grid."""

    if count <= 0:
        raise ValueError("count must be positive")
    if count >= len(boxes):
        return list(boxes)
    remaining = list(enumerate(boxes))
    selected: list[BBox] = []
    for anchor_x, anchor_y in _anchor_grid(count):
        position, (_, box) = min(
            enumerate(remaining),
            key=lambda item: (
                ((item[1][1].x1 + item[1][1].x2) / 2.0 - anchor_x) ** 2
                + ((item[1][1].y1 + item[1][1].y2) / 2.0 - anchor_y) ** 2,
                item[1][0],
            ),
        )
        selected.append(box)
        remaining.pop(position)
    return selected


@dataclass(frozen=True)
class UGGridProposer:
    """Ground-truth-free crop proposer based on the official UG geometry."""

    candidate_count: int = 4
    visual_crop_ratio: float = 2.0
    visual_cost: float = 1.0

    def __post_init__(self) -> None:
        if self.candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        if self.visual_cost < 0.0:
            raise ValueError("visual_cost must be non-negative")

    def __call__(self, state: AgentState) -> list[ActionSpec]:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("UGGridProposer requires Pillow") from exc

        image_path = Path(state.image_path)
        with Image.open(image_path) as image:
            width, height = image.size
        full_grid = ug_grid_boxes(
            width,
            height,
            visual_crop_ratio=self.visual_crop_ratio,
        )
        boxes = spatially_balanced_subset(full_grid, self.candidate_count)
        return [
            ActionSpec(
                action_id=f"ug-grid-{index:02d}",
                bbox=box,
                visual_cost=self.visual_cost,
                pre_action_features={
                    "bbox_area": box.area,
                    "bbox_center_x": (box.x1 + box.x2) / 2.0,
                    "bbox_center_y": (box.y1 + box.y2) / 2.0,
                    "ug_grid_size": float(len(full_grid)),
                },
            )
            for index, box in enumerate(boxes)
        ]
