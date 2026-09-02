from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

from PIL import Image
import pytest

from beyond_entropy.refocus_chart_audit import structural_chart_signature
from beyond_entropy.refocus_g1_dataset import (
    ACTION_SYSTEM_PROMPT_V1,
    AGENT_NAME,
    DATA_SOURCE,
    build_official_tool_metadata,
    convert_official_train_row,
    group_development_split,
    select_structural_groups,
)


def _png_bytes(color: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    output = BytesIO()
    Image.new("RGB", (12, 8), color).save(output, format="PNG")
    return output.getvalue()


def _official_row(
    index: int = 0,
    *,
    question: str = "How many wins did Alpha have?",
    answer: str = "7",
    figure_x2: int = 100,
) -> dict:
    return {
        "id": f"chart-{index}",
        "question": question,
        "answer": answer,
        "image": {"bytes": _png_bytes(), "path": f"chart-{index}.png"},
        "thoughts": ["PRIVATE_TEACHER_THOUGHT_CANARY"],
        "edited_image": {
            "bytes": _png_bytes((200, 10, 10)),
            "path": "EDITED_IMAGE_CANARY.png",
        },
        "focus_areas_bbox": {
            "x1": [987654321],
            "y1": [2],
            "x2": [3],
            "y2": [4],
        },
        "source": "chartqa_h_bar",
        "split": "train",
        "x_values": [],
        "y_values": ["Alpha", "Beta"],
        "x_values_bbox": {"x1": [], "y1": [], "x2": [], "y2": []},
        "y_values_bbox": {
            "x1": [1, 5],
            "y1": [2, 6],
            "x2": [3, 7],
            "y2": [4, 8],
        },
        "figure_bbox": {"x1": 0, "y1": 0, "x2": figure_x2, "y2": 80},
    }


def test_converter_exposes_only_original_image_and_structural_tool_metadata() -> None:
    official = _official_row(answer="ANSWER_TARGET_CANARY")
    converted = convert_official_train_row(
        official, index=11, development_split="g1_train"
    )

    assert converted["data_source"] == DATA_SOURCE
    assert converted["agent_name"] == AGENT_NAME
    assert converted["images"] == [
        {"bytes": official["image"]["bytes"], "path": "chart-0.png"}
    ]
    prompt_text = json.dumps(converted["prompt"], ensure_ascii=False)
    assert prompt_text.count("<image>") == 1
    assert "How many wins did Alpha have?" in prompt_text
    assert "Alpha" in prompt_text
    assert "ANSWER_TARGET_CANARY" not in prompt_text
    assert "PRIVATE_TEACHER_THOUGHT_CANARY" not in prompt_text
    assert "EDITED_IMAGE_CANARY" not in prompt_text
    assert "987654321" not in prompt_text

    metadata = json.loads(converted["extra_info"]["tools_kwargs"]["metadata"])
    assert set(metadata) == {
        "source",
        "x_values",
        "y_values",
        "x_values_bbox",
        "y_values_bbox",
        "figure_bbox",
    }
    assert "focus_areas_bbox" not in metadata
    assert converted["reward_model"]["ground_truth"] == "ANSWER_TARGET_CANARY"
    assert converted["extra_info"]["index"] == 11
    assert converted["extra_info"]["structural_chart_sha256"] == (
        structural_chart_signature(metadata)
    )


def test_frozen_action_prompt_matches_executable_parser_surface() -> None:
    for symbol in (
        "image_1",
        "x_values_bbox",
        "y_values_bbox",
        "focus_on_x_values_with_draw",
        "focus_on_y_values_with_highlight",
        "FINAL ANSWER",
    ):
        assert symbol in ACTION_SYSTEM_PROMPT_V1
    assert "teacher" not in ACTION_SYSTEM_PROMPT_V1.lower()
    assert "ground truth" not in ACTION_SYSTEM_PROMPT_V1.lower()


def test_converter_fails_closed_on_protected_split_or_bad_original_image() -> None:
    protected = _official_row()
    protected["split"] = "test"
    with pytest.raises(ValueError, match="train rows only"):
        convert_official_train_row(protected, index=0, development_split="g1_train")

    bad_image = _official_row()
    bad_image["image"] = {"bytes": b"not-an-image", "path": "bad.png"}
    with pytest.raises(ValueError, match="not decodable"):
        convert_official_train_row(bad_image, index=0, development_split="g1_train")

    with pytest.raises(ValueError, match="agent_name"):
        convert_official_train_row(
            _official_row(),
            index=0,
            development_split="g1_train",
            agent_name="unregistered_agent",
        )


def test_outcome_only_variant_changes_only_agent_routing() -> None:
    official = _official_row()
    paired = convert_official_train_row(official, index=0, development_split="g1_train")
    outcome_only = convert_official_train_row(
        official,
        index=0,
        development_split="g1_train",
        agent_name="vtool_agent",
    )
    assert paired["agent_name"] == AGENT_NAME
    assert outcome_only["agent_name"] == "vtool_agent"
    assert {**paired, "agent_name": "vtool_agent"} == outcome_only


def test_structural_group_selection_is_deterministic_and_disjoint() -> None:
    rows = [_official_row(index, figure_x2=100 + index) for index in range(100)]
    # Add a sibling question over the exact same chart structure as row zero.
    sibling = _official_row(
        1000,
        question="What about Beta?",
        answer="4",
        figure_x2=100,
    )
    rows.append(sibling)

    first = select_structural_groups(
        rows, max_train_groups=100, max_curve_eval_groups=100
    )
    second = select_structural_groups(
        reversed(rows), max_train_groups=100, max_curve_eval_groups=100
    )
    assert first == second
    assert first.all_train_groups + first.all_curve_eval_groups == 100
    assert first.all_train_groups > 0
    assert first.all_curve_eval_groups > 0
    assert set(first.group_to_split.values()) == {"g1_train", "g1_curve_eval"}

    zero_group = structural_chart_signature(build_official_tool_metadata(rows[0]))
    sibling_group = structural_chart_signature(build_official_tool_metadata(sibling))
    assert zero_group == sibling_group
    assert first.group_to_split[zero_group] == first.group_to_split[sibling_group]


def test_group_assignment_validates_digest_and_fraction() -> None:
    digest = "a" * 64
    assert group_development_split(digest) in {"g1_train", "g1_curve_eval"}
    with pytest.raises(ValueError, match="SHA-256"):
        group_development_split("short")
    with pytest.raises(ValueError, match="strictly"):
        group_development_split(digest, curve_eval_fraction=0.0)


def test_converter_rows_roundtrip_through_arrow_schema() -> None:
    pa = pytest.importorskip("pyarrow")
    rows = [
        convert_official_train_row(
            _official_row(index), index=index, development_split="g1_train"
        )
        for index in range(2)
    ]
    restored = pa.Table.from_pylist(rows).to_pylist()
    assert [row["id"] for row in restored] == ["chart-0", "chart-1"]
    assert restored[0]["images"][0]["bytes"] == rows[0]["images"][0]["bytes"]
    assert restored[0]["extra_info"]["tools_kwargs"]["name"] == "refocus"


def test_converter_runner_has_train_only_column_contract() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "convert_refocus_official_g1.py"
    ).read_text(encoding="utf-8")
    metadata_assignment = script.split("METADATA_COLUMNS = (", 1)[1].split(")", 1)[0]
    assert '"split"' in metadata_assignment
    assert '"image"' not in metadata_assignment
    assert '"thoughts"' not in metadata_assignment
    assert '"edited_image"' not in metadata_assignment
    assert '"focus_areas_bbox"' not in metadata_assignment
    assert 'IMAGE_COLUMN = "image"' in script
    assert "converter accepts only pinned data/train shards" in script
