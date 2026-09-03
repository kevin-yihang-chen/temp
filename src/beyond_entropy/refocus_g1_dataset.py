from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from beyond_entropy.refocus_chart_audit import (
    STRUCTURAL_METADATA_FIELDS,
    canonical_sha256,
    normalize_official_bbox_columns,
    structural_chart_signature,
)


CONVERTER_SCHEMA = "refocus_official_g1_converter_v1"
DATA_SOURCE = "ReFocus/ReFocus_Data"
AGENT_NAME = "counterfactual_credit_vtool_agent"
TOOL_NAME = "refocus"
GROUP_SPLIT_SEED = "refocus-official-g1-group-split-20260902-v1"
CURVE_EVAL_FRACTION = 0.20

# This prompt is part of the frozen G1 environment, not a claim of equivalence
# with any VTool prompt. It exposes the executable action space without using
# row-level thoughts, edited images, focus boxes, answers, or outcomes.
ACTION_SYSTEM_PROMPT_V1 = """You are a chart question-answering assistant. Inspect the original chart and answer the user's question.

Your first response must choose exactly one of these paths:
1. If the original chart is sufficient, answer directly and end with `FINAL ANSWER: <answer> TERMINATE`.
2. If focusing on a subset of axis labels would help, emit exactly one executable Python code block. Call exactly one of the functions below and display its returned image. Do not answer in the same response as the tool call.

Available image and bounding-box variables:
- `image_1`
- `x_values_bbox`, `y_values_bbox`, `columns_bbox`, `rows_bbox`

Available visual focus functions:
- `focus_on_x_values_with_draw`, `focus_on_y_values_with_draw`
- `focus_on_x_values_with_highlight`, `focus_on_y_values_with_highlight`
- `focus_on_x_values_with_mask`, `focus_on_y_values_with_mask`

Use only axis-label strings listed by the user. Use as few tools as possible. Never use Python to calculate or print the answer; Python is only for producing one focused image. After an observation is returned, answer the original question and end with `FINAL ANSWER: <answer> TERMINATE`."""

# V2 is an independent baseline contract introduced after the frozen G1 result.
# It must never be used to reinterpret or overwrite V1/Job 206205 evidence.
ACTION_SYSTEM_PROMPT_V2 = """You are a chart question-answering assistant. Inspect the original chart and answer the user's question.

Your first response must choose exactly one of these paths:
1. If the original chart is sufficient, answer directly and end with `FINAL ANSWER: <answer> TERMINATE`.
2. If focusing on axis labels would help, emit exactly one complete `python` code block and nothing else. The block must contain exactly one `display(...)` expression with one focus call.

The only valid x-axis form is:
`display(focus_on_x_values_with_MODE(image_1, ["LABEL_FROM_X_LIST"], columns_bbox))`

The only valid y-axis form is:
`display(focus_on_y_values_with_MODE(image_1, ["LABEL_FROM_Y_LIST"], rows_bbox))`

Replace `MODE` with exactly one of `draw`, `highlight`, or `mask`. Replace each label placeholder with one or more exact strings from the corresponding user-provided axis-label list. Do not change `image_1` or the corresponding bbox variable. Do not use keyword arguments, coordinates, assignments, imports, calculations, print, or a final answer in the tool-call response.

After an observation is returned, answer the original question and end with `FINAL ANSWER: <answer> TERMINATE`."""


@dataclass(frozen=True)
class GroupSelection:
    group_to_split: dict[str, str]
    all_train_groups: int
    all_curve_eval_groups: int

    @property
    def selected_train_groups(self) -> int:
        return sum(split == "g1_train" for split in self.group_to_split.values())

    @property
    def selected_curve_eval_groups(self) -> int:
        return sum(split == "g1_curve_eval" for split in self.group_to_split.values())


def _require_text(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"official row {field} must be a non-empty string")
    return value.strip()


def _require_label_sequence(row: Mapping[str, Any], field: str) -> list[str]:
    value = row.get(field)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"official row {field} must be a sequence")
    labels = [str(label) for label in value]
    if len(labels) != len(set(labels)):
        raise ValueError(f"official row {field} labels must be unique")
    return labels


def build_official_tool_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    """Build only the metadata required to execute the visual focus tool.

    Question-conditioned ``focus_areas_bbox`` is deliberately absent: exposing it
    would leak the dataset's teacher action into the policy input.
    """

    source = _require_text(row, "source")
    x_values = _require_label_sequence(row, "x_values")
    y_values = _require_label_sequence(row, "y_values")
    figure_bbox = row.get("figure_bbox")
    if not isinstance(figure_bbox, Mapping):
        raise ValueError("official row figure_bbox must be a mapping")
    metadata = {
        "source": source,
        "x_values": x_values,
        "y_values": y_values,
        "x_values_bbox": normalize_official_bbox_columns(
            x_values,
            row.get("x_values_bbox"),
            field_name="x_values_bbox",
        ),
        "y_values_bbox": normalize_official_bbox_columns(
            y_values,
            row.get("y_values_bbox"),
            field_name="y_values_bbox",
        ),
        "figure_bbox": dict(figure_bbox),
    }
    if set(metadata) != set(STRUCTURAL_METADATA_FIELDS):
        raise AssertionError("tool metadata must contain exactly structural fields")
    return metadata


def build_action_prompt(
    *, question: str, x_values: Sequence[str], y_values: Sequence[str]
) -> list[dict[str, str]]:
    if not question.strip():
        raise ValueError("question must be non-empty")
    user_content = (
        "<image>\n"
        f"Question: {question.strip()}\n"
        f"Available x-axis labels: {json.dumps(list(x_values), ensure_ascii=False)}\n"
        f"Available y-axis labels: {json.dumps(list(y_values), ensure_ascii=False)}"
    )
    return [
        {"role": "system", "content": ACTION_SYSTEM_PROMPT_V1},
        {"role": "user", "content": user_content},
    ]


def build_typed_action_prompt(
    *, question: str, x_values: Sequence[str], y_values: Sequence[str]
) -> list[dict[str, str]]:
    """Build the post-G1 V2 baseline prompt without mutating the frozen V1 path."""

    prompt = build_action_prompt(
        question=question,
        x_values=x_values,
        y_values=y_values,
    )
    prompt[0] = {"role": "system", "content": ACTION_SYSTEM_PROMPT_V2}
    return prompt


def _validated_original_image(row: Mapping[str, Any], *, row_id: str) -> dict[str, Any]:
    from PIL import Image

    image = row.get("image")
    if not isinstance(image, Mapping):
        raise ValueError(f"official row {row_id!r} image must be a mapping")
    image_bytes = image.get("bytes")
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise ValueError(f"official row {row_id!r} image.bytes must be non-empty")
    try:
        with Image.open(BytesIO(image_bytes)) as decoded:
            decoded.verify()
    except Exception as exc:
        raise ValueError(
            f"official row {row_id!r} image bytes are not decodable"
        ) from exc
    suffix = Path(str(image.get("path") or "image.png")).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        suffix = ".png"
    return {"bytes": image_bytes, "path": f"{row_id}{suffix}"}


def convert_official_train_row(
    row: Mapping[str, Any],
    *,
    index: int,
    development_split: str,
    agent_name: str = AGENT_NAME,
) -> dict[str, Any]:
    if type(index) is not int or index < 0:
        raise ValueError("index must be a non-negative integer")
    if development_split not in {"g1_train", "g1_curve_eval", "g1_smoke"}:
        raise ValueError("unsupported development split")
    if agent_name not in {AGENT_NAME, "vtool_agent"}:
        raise ValueError("unsupported agent_name")
    if row.get("split") != "train":
        raise ValueError("converter accepts official train rows only")

    row_id = _require_text(row, "id")
    question = _require_text(row, "question")
    answer = _require_text(row, "answer")
    metadata = build_official_tool_metadata(row)
    prompt = build_action_prompt(
        question=question,
        x_values=metadata["x_values"],
        y_values=metadata["y_values"],
    )
    metadata_json = json.dumps(
        metadata,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    group_sha256 = structural_chart_signature(metadata)
    return {
        "id": row_id,
        "source": metadata["source"],
        "split": development_split,
        "data_source": DATA_SOURCE,
        "ability": "chart_qa",
        "agent_name": agent_name,
        "prompt": prompt,
        "images": [_validated_original_image(row, row_id=row_id)],
        "reward_model": {"ground_truth": answer, "style": "rule"},
        "extra_info": {
            "answer": answer,
            "index": index,
            "need_tools_kwargs": True,
            "question": question,
            "row_id": row_id,
            "source": metadata["source"],
            "split": development_split,
            "structural_chart_sha256": group_sha256,
            "prompt_sha256": canonical_sha256(prompt),
            "tools_kwargs": {"name": TOOL_NAME, "metadata": metadata_json},
        },
    }


def group_development_split(
    structural_sha256: str,
    *,
    curve_eval_fraction: float = CURVE_EVAL_FRACTION,
    seed: str = GROUP_SPLIT_SEED,
) -> str:
    if len(structural_sha256) != 64:
        raise ValueError("structural_sha256 must be a SHA-256 digest")
    if not 0.0 < curve_eval_fraction < 1.0:
        raise ValueError("curve_eval_fraction must be strictly between zero and one")
    digest = canonical_sha256([seed, structural_sha256, "assignment"])
    unit_interval = int(digest, 16) / (1 << 256)
    return "g1_curve_eval" if unit_interval < curve_eval_fraction else "g1_train"


def select_structural_groups(
    rows: Iterable[Mapping[str, Any]],
    *,
    max_train_groups: int,
    max_curve_eval_groups: int,
    curve_eval_fraction: float = CURVE_EVAL_FRACTION,
    seed: str = GROUP_SPLIT_SEED,
) -> GroupSelection:
    if max_train_groups < 0 or max_curve_eval_groups < 0:
        raise ValueError("group limits must be non-negative")
    row_ids: set[str] = set()
    groups: set[str] = set()
    for row in rows:
        if row.get("split") != "train":
            raise ValueError("group selection accepts official train rows only")
        row_id = _require_text(row, "id")
        if row_id in row_ids:
            raise ValueError("official train row IDs must be unique")
        row_ids.add(row_id)
        groups.add(structural_chart_signature(build_official_tool_metadata(row)))
    if not groups:
        raise ValueError("official train rows must be non-empty")

    candidates: dict[str, list[str]] = {"g1_train": [], "g1_curve_eval": []}
    for group_sha256 in groups:
        split = group_development_split(
            group_sha256,
            curve_eval_fraction=curve_eval_fraction,
            seed=seed,
        )
        candidates[split].append(group_sha256)
    for split, values in candidates.items():
        values.sort(key=lambda value: canonical_sha256([seed, split, value, "select"]))

    selected = {
        group_sha256: split
        for split, limit in (
            ("g1_train", max_train_groups),
            ("g1_curve_eval", max_curve_eval_groups),
        )
        for group_sha256 in candidates[split][:limit]
    }
    return GroupSelection(
        group_to_split=selected,
        all_train_groups=len(candidates["g1_train"]),
        all_curve_eval_groups=len(candidates["g1_curve_eval"]),
    )
