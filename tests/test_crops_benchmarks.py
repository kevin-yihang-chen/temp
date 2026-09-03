import json

import pytest
from PIL import Image

from beyond_entropy.benchmarks import (
    chartqa_relaxed_match,
    extract_answer_letter,
    load_manifest,
    vstar_match,
)
from beyond_entropy.crops import (
    ChartLayoutProposer,
    UGGridProposer,
    chart_layout_boxes,
    spatially_balanced_subset,
    ug_grid_boxes,
)
from beyond_entropy.image_ops import normalized_crop_resized_to_source
from beyond_entropy.qwen_backend import Qwen25VLBackend
from beyond_entropy.rollout import AgentState, GroundTruth
from beyond_entropy.rollout import VisualObservation
from beyond_entropy.schema import BBox


def test_ug_grid_matches_reference_geometry():
    boxes = ug_grid_boxes(400, 400, visual_crop_ratio=2)
    assert len(boxes) == 9
    assert boxes[0].to_list() == [0.0, 0.0, 0.5, 0.5]
    assert boxes[4].to_list() == [0.25, 0.25, 0.75, 0.75]
    assert boxes[-1].to_list() == [0.5, 0.5, 1.0, 1.0]
    subset = spatially_balanced_subset(boxes, 4)
    assert [box.to_list() for box in subset] == [
        [0.0, 0.0, 0.5, 0.5],
        [0.5, 0.0, 1.0, 0.5],
        [0.0, 0.5, 0.5, 1.0],
        [0.5, 0.5, 1.0, 1.0],
    ]


def test_backend_and_shared_normalized_crop_pixels_are_identical():
    image = Image.new("RGB", (7, 5))
    for y in range(5):
        for x in range(7):
            image.putpixel((x, y), (x * 20, y * 30, x + y))
    bbox = BBox(0.1, 0.2, 0.8, 0.9)
    observation = VisualObservation("ZOOM", "unused.png", "crop", bbox)
    shared = normalized_crop_resized_to_source(image, bbox)
    backend = Qwen25VLBackend._crop_pixels(image, observation)
    assert shared.size == image.size
    assert shared.tobytes() == backend.tobytes()


def test_proposer_reads_dimensions_without_ground_truth(tmp_path):
    image_path = tmp_path / "wide.png"
    Image.new("RGB", (600, 400), "white").save(image_path)
    state = AgentState("s1", "i1", "src1", str(image_path), "Question?")
    proposals = UGGridProposer(candidate_count=4)(state)
    assert len(proposals) == 4
    assert all(proposal.bbox.area == pytest.approx(1 / 6) for proposal in proposals)
    assert all(
        proposal.pre_action_features["ug_grid_size"] == 15.0 for proposal in proposals
    )


def test_chart_layout_proposer_covers_axes_center_and_right_without_labels(tmp_path):
    boxes = chart_layout_boxes(600, 400, visual_crop_ratio=2)
    assert [box.to_list() for box in boxes] == [
        [0.0, 0.0, 1 / 3, 0.5],
        [0.0, 0.25, 1 / 3, 0.75],
        [1 / 3, 0.25, 2 / 3, 0.75],
        [2 / 3, 0.25, 1.0, 0.75],
    ]
    image_path = tmp_path / "chart.png"
    Image.new("RGB", (600, 400), "white").save(image_path)
    state = AgentState("s1", "i1", "src1", str(image_path), "Question?")
    proposals = ChartLayoutProposer()(state)
    assert [proposal.action_id for proposal in proposals] == [
        "chart-layout-00",
        "chart-layout-01",
        "chart-layout-02",
        "chart-layout-03",
    ]
    assert all(
        proposal.pre_action_features["chart_layout"] == 1.0 for proposal in proposals
    )


def test_manifest_and_reference_scorers(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (16, 16), "white").save(image_path)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "state_id": "s1",
                "image_path": "image.png",
                "question": "Choose one: (A) x (B) y",
                "target": "B",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    examples = load_manifest(manifest)
    assert examples[0].state.image_path == str(image_path.resolve())
    assert extract_answer_letter("The answer is B.") == "B"
    assert vstar_match("B", GroundTruth("B")) == 1.0
    assert chartqa_relaxed_match("104", GroundTruth("100")) == 1.0
    assert chartqa_relaxed_match("106", GroundTruth("100")) == 0.0
    assert chartqa_relaxed_match("0", GroundTruth("0")) == 1.0


def test_manifest_separates_gate_question_from_backend_prompt(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (16, 16), "white").save(image_path)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "state_id": "s1",
                "image_path": "image.png",
                "question": "Gate-visible core question",
                "model_prompt": "Backend-only formatted prompt",
                "target": "answer",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    state = load_manifest(manifest)[0].state
    assert state.question == "Gate-visible core question"
    assert state.model_prompt == "Backend-only formatted prompt"
    assert state.backend_prompt == "Backend-only formatted prompt"


def test_manifest_rejects_missing_images(tmp_path):
    manifest = tmp_path / "bad.jsonl"
    manifest.write_text(
        '{"state_id":"s","image_path":"missing.png","question":"q","target":"a"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="image does not exist"):
        load_manifest(manifest)


def test_qwen_messages_keep_original_and_resize_zoom(tmp_path):
    image_path = tmp_path / "quadrants.png"
    image = Image.new("RGB", (20, 20), "white")
    for x in range(10, 20):
        for y in range(10, 20):
            image.putpixel((x, y), (0, 0, 255))
    image.save(image_path)
    state = AgentState("s", "i", "src", str(image_path), "What color?")
    backend = Qwen25VLBackend.__new__(Qwen25VLBackend)
    backend.min_pixels = 1
    backend.max_pixels = 10_000
    backend.system_prompt = "System"
    messages = backend._messages(
        state,
        (
            VisualObservation("ORIGINAL", str(image_path), "original", None),
            VisualObservation(
                "ZOOM",
                str(image_path),
                "zoom",
                BBox(0.5, 0.5, 1.0, 1.0),
            ),
        ),
    )
    content = messages[1]["content"]
    assert len(content) == 3
    assert content[0]["image"].size == (20, 20)
    assert content[1]["image"].size == (20, 20)
    assert content[1]["image"].getpixel((5, 5)) == (0, 0, 255)
    assert content[2] == {"type": "text", "text": "What color?"}


def test_qwen_messages_use_backend_only_model_prompt(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (10, 10), "white").save(image_path)
    state = AgentState(
        "s",
        "i",
        "src",
        str(image_path),
        "Gate-visible core question",
        model_prompt="Backend-only formatted prompt",
    )
    backend = Qwen25VLBackend.__new__(Qwen25VLBackend)
    backend.min_pixels = 1
    backend.max_pixels = 10_000
    backend.system_prompt = "System"
    messages = backend._messages(
        state,
        (VisualObservation("ORIGINAL", str(image_path), "original", None),),
    )
    assert messages[1]["content"][-1] == {
        "type": "text",
        "text": "Backend-only formatted prompt",
    }
