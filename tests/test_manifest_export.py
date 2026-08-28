import base64
import io
import json

import pytest
from PIL import Image

from beyond_entropy.cli import build_parser
from beyond_entropy.manifest_export import (
    BENCHMARK_SPECS,
    benchmark_stratum,
    export_benchmark_manifest,
    hash_ranked_source_group_indices,
    stratified_sample_indices,
    stratified_unique_group_sample_indices,
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


def test_stratified_unique_group_indices_are_balanced_and_group_disjoint():
    labels = ["a", "b", "a", "b", "a", "b", "a", "b"]
    groups = ["shared", "shared", "a1", "b1", "a2", "b2", "a3", "b3"]
    first = stratified_unique_group_sample_indices(
        labels,
        groups,
        count=6,
        seed=11,
    )
    second = stratified_unique_group_sample_indices(
        labels,
        groups,
        count=6,
        seed=11,
    )
    assert first == second
    assert len({groups[index] for index in first}) == 6
    assert {label: sum(labels[index] == label for index in first) for label in set(labels)} == {
        "a": 3,
        "b": 3,
    }


def test_stratified_unique_group_indices_reject_impossible_count():
    with pytest.raises(ValueError, match="unique groups"):
        stratified_unique_group_sample_indices(
            ["a", "b"],
            ["shared", "shared"],
            count=2,
            seed=0,
        )


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


def test_export_chartqa_supports_split_specific_state_namespace(tmp_path):
    row = {
        "image": Image.new("RGB", (10, 10), "white"),
        "type": "human_val",
        "question": "Which bar is highest?",
        "answer": "blue",
    }
    result = export_benchmark_manifest(
        [row],
        source_indices=[0],
        task="chartqa",
        dataset_id="HuggingFaceM4/ChartQA",
        dataset_revision="revision",
        output_dir=tmp_path,
        seed=0,
        state_namespace="chartqa-val",
    )
    payload = json.loads((tmp_path / "manifest.jsonl").read_text())
    assert payload["state_id"] == "chartqa-val:00000"
    assert payload["source_id"] == "chartqa-val:00000"
    assert payload["stratum"] == "human_val"
    assert result["state_namespace"] == "chartqa-val"


def test_cross_benchmark_specs_are_revision_pinned():
    assert BENCHMARK_SPECS["docvqa"].dataset_name == "DocVQA"
    assert len(BENCHMARK_SPECS["docvqa"].default_revision) == 40
    assert BENCHMARK_SPECS["textvqa"].scorer == "textvqa"
    assert BENCHMARK_SPECS["hrbench4k"].split == "hrbench_4k"
    assert BENCHMARK_SPECS["hrbench8k"].split == "hrbench_8k"


def test_cross_benchmark_strata_use_only_pre_outcome_fields():
    assert benchmark_stratum(
        {"question_types": ["table", "figure"]}, task="docvqa"
    ) == "figure+table"
    assert benchmark_stratum({"ocr_tokens": []}, task="textvqa") == "ocr-000"
    assert benchmark_stratum(
        {"ocr_tokens": [str(index) for index in range(6)]}, task="textvqa"
    ) == "ocr-006-015"
    assert benchmark_stratum(
        {"category": "cross", "cycle_category": "text"}, task="hrbench4k"
    ) == "cross:text"


def test_hash_ranked_source_selection_is_order_independent_and_keeps_groups():
    groups = ["shared", "a", "shared", "b", "c", "d", "e"]
    first = hash_ranked_source_group_indices(
        groups,
        count=3,
        seed=20260828,
        namespace="cross-benchmark-v1",
    )
    reversed_groups = list(reversed(groups))
    second = hash_ranked_source_group_indices(
        reversed_groups,
        count=3,
        seed=20260828,
        namespace="cross-benchmark-v1",
    )
    first_groups = {groups[index] for index in first}
    second_groups = {reversed_groups[index] for index in second}
    assert first_groups == second_groups
    assert len(first_groups) == 3
    assert all(
        (index in first) == (group in first_groups)
        for index, group in enumerate(groups)
    )


def test_hash_ranked_source_offsets_are_disjoint():
    groups = [f"source-{index}" for index in range(10)]
    development = hash_ranked_source_group_indices(
        groups,
        count=3,
        offset=0,
        seed=17,
        namespace="split-v1",
    )
    formal = hash_ranked_source_group_indices(
        groups,
        count=4,
        offset=3,
        seed=17,
        namespace="split-v1",
    )
    assert {groups[index] for index in development}.isdisjoint(
        {groups[index] for index in formal}
    )
    with pytest.raises(ValueError, match="eligible groups"):
        hash_ranked_source_group_indices(
            groups,
            count=8,
            offset=3,
            seed=17,
            namespace="split-v1",
        )


def test_hash_ranked_source_exclusion_backfills_without_changing_count():
    groups = [f"source-{index}" for index in range(12)]
    original = hash_ranked_source_group_indices(
        groups,
        count=5,
        offset=3,
        seed=17,
        namespace="split-v1",
    )
    excluded = groups[original[0]]
    replacement = hash_ranked_source_group_indices(
        groups,
        count=5,
        offset=3,
        seed=17,
        namespace="split-v1",
        excluded_groups=[excluded],
    )
    replacement_groups = {groups[index] for index in replacement}
    assert len(replacement_groups) == 5
    assert excluded not in replacement_groups
    assert len(replacement_groups - {groups[index] for index in original}) == 1


def test_export_docvqa_groups_questions_from_the_same_document(tmp_path):
    image = Image.new("RGB", (12, 8), "white")
    rows = [
        {
            "image": image,
            "questionId": f"q{index}",
            "question": f"Question {index}?",
            "question_types": ["table"],
            "docId": "shared-document",
            "answers": ["answer", "Answer"],
        }
        for index in range(2)
    ]
    export_benchmark_manifest(
        rows,
        source_indices=[2, 7],
        task="docvqa",
        dataset_id="lmms-lab/DocVQA",
        dataset_revision="revision",
        output_dir=tmp_path,
        seed=19,
    )
    payloads = [
        json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()
    ]
    assert payloads[0]["source_id"] == payloads[1]["source_id"]
    assert payloads[0]["image_id"] == payloads[1]["image_id"]
    assert payloads[0]["question"] == "Question 0?"
    assert payloads[0]["model_prompt"].endswith("single word or phrase.")
    assert payloads[0]["target"] == {"answers": ["answer", "Answer"]}


def test_export_textvqa_separates_gate_question_from_ocr_prompt(tmp_path):
    row = {
        "image": Image.new("RGB", (10, 10), "blue"),
        "image_id": "open-images-id",
        "question_id": 42,
        "question": "WHAT word is shown?",
        "ocr_tokens": ["HELLO", "WORLD"],
        "answers": ["hello"] * 10,
    }
    result = export_benchmark_manifest(
        [row],
        source_indices=[42],
        task="textvqa",
        dataset_id="lmms-lab/textvqa",
        dataset_revision="revision",
        output_dir=tmp_path,
        seed=23,
    )
    payload = json.loads((tmp_path / "manifest.jsonl").read_text())
    assert payload["question"] == "WHAT word is shown?"
    assert "Reference OCR token: HELLO, WORLD" in payload["model_prompt"]
    assert payload["source_id"] == "textvqa:open-images-id"
    assert payload["target"] == {"answers": ["hello"] * 10}
    assert result["scorer"] == "textvqa"


def test_export_hrbench_decodes_base64_and_pairs_resolution_source(tmp_path):
    buffer = io.BytesIO()
    Image.new("RGB", (14, 9), "green").save(buffer, format="PNG")
    row = {
        "index": 7,
        "question": "Which option is visible?",
        "answer": "C",
        "category": "single",
        "cycle_category": "text",
        "A": "alpha",
        "B": "beta",
        "C": "gamma",
        "D": "delta",
        "image": base64.b64encode(buffer.getvalue()).decode(),
    }
    export_benchmark_manifest(
        [row],
        source_indices=[7],
        task="hrbench4k",
        dataset_id="DreamMr/HR-Bench",
        dataset_revision="revision",
        output_dir=tmp_path,
        seed=29,
    )
    payload = json.loads((tmp_path / "manifest.jsonl").read_text())
    assert payload["source_id"] == "hrbench:7"
    assert payload["question"].endswith("D. delta")
    assert payload["model_prompt"].endswith("option letter directly.")
    assert payload["target"] == {
        "answer": "C",
        "category": "single",
        "cycle_category": "text",
    }
    exported = Image.open(tmp_path / payload["image_path"])
    assert exported.size == (14, 9)


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


@pytest.mark.parametrize("scorer", ["docvqa", "textvqa", "hrbench"])
def test_collect_qwen_accepts_cross_benchmark_scorers(tmp_path, scorer):
    args = build_parser().parse_args(
        [
            "collect-qwen",
            "--manifest",
            str(tmp_path / "manifest.jsonl"),
            "--output",
            str(tmp_path / "output.jsonl"),
            "--scorer",
            scorer,
        ]
    )
    assert args.scorer == scorer
