import json

import pytest
from PIL import Image

from beyond_entropy.cli import build_parser
from beyond_entropy.manifest_export import (
    export_benchmark_manifest,
    stratified_sample_indices,
)


def test_stratified_indices_are_balanced_and_deterministic():
    labels = ["a"] * 8 + ["b"] * 8 + ["c"] * 2
    first = stratified_sample_indices(labels, count=9, seed=3)
    second = stratified_sample_indices(labels, count=9, seed=3)
    assert first == second
    counts = {label: sum(labels[index] == label for index in first) for label in set(labels)}
    assert counts == {"a": 4, "b": 3, "c": 2}
    assert len(first) == 9


def test_stratified_indices_validate_count():
    with pytest.raises(ValueError, match="exceed"):
        stratified_sample_indices(["a"], count=2, seed=0)


def test_export_vstar_manifest_deduplicates_images(tmp_path):
    image = Image.new("RGB", (20, 10), "blue")
    rows = [
        {
            "image": image,
            "text": "Question? (A) one (B) two",
            "category": "direct_attributes",
            "question_id": f"q{index}",
            "label": "B",
        }
        for index in range(2)
    ]
    result = export_benchmark_manifest(
        rows,
        source_indices=[4, 9],
        task="vstar",
        dataset_id="lmms-lab/vstar-bench",
        dataset_revision="revision",
        output_dir=tmp_path,
        seed=17,
    )
    manifest_path = tmp_path / "manifest.jsonl"
    payloads = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    assert len(list((tmp_path / "images").glob("*.png"))) == 1
    assert payloads[0]["image_id"] == payloads[1]["image_id"]
    assert payloads[0]["image_path"].startswith("images/")
    assert payloads[0]["question"].endswith("choices directly.")
    assert result["manifest_sha256"]
    assert result["stratum_counts"] == {"direct_attributes": 2}


def test_export_does_not_duplicate_existing_vstar_instruction(tmp_path):
    instruction = "Answer with the option's letter from the given choices directly."
    row = {
        "image": Image.new("RGB", (10, 10), "white"),
        "text": f"Question? (A) one (B) two\n{instruction}",
        "category": "relative_position",
        "question_id": "q",
        "label": "A",
    }
    export_benchmark_manifest(
        [row],
        source_indices=[0],
        task="vstar",
        dataset_id="lmms-lab/vstar-bench",
        dataset_revision="revision",
        output_dir=tmp_path,
        seed=17,
    )
    payload = json.loads((tmp_path / "manifest.jsonl").read_text())
    assert payload["question"].count(instruction) == 1


def test_collect_qwen_rejects_changed_manifest_before_model_load(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    args = build_parser().parse_args(
        [
            "collect-qwen",
            "--manifest",
            str(manifest),
            "--expected-manifest-sha256",
            "0" * 64,
            "--output",
            str(tmp_path / "output.jsonl"),
            "--scorer",
            "vstar",
        ]
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        args.func(args)
