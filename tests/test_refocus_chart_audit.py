from __future__ import annotations

import json
from pathlib import Path

import pytest

from beyond_entropy.refocus_chart_audit import (
    audit_chartqa_train_lineage,
    audit_split_rows,
    build_row_manifest,
    structural_chart_signature,
)


def _metadata(*, focus: int = 1, y_values: list[str] | None = None) -> dict:
    return {
        "source": "chartqa_h_bar",
        "x_values": [],
        "y_values": ["alpha", "beta"] if y_values is None else y_values,
        "x_values_bbox": {},
        "y_values_bbox": {"alpha": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}},
        "figure_bbox": {"x1": 1, "y1": 2, "x2": 30, "y2": 40},
        "focus_areas_bbox": {"x1": [focus], "y1": [2], "x2": [3], "y2": [4]},
    }


def _row(
    index: int,
    *,
    split: str,
    question: str,
    answer: str,
    metadata: dict | None = None,
) -> dict:
    return {
        "id": f"row-{split}-{index}",
        "source": "chartqa_h_bar",
        "split": split,
        "data_source": "ReFocus/ReFocus_Data",
        "ability": "math",
        "agent_name": "vtool_agent",
        "prompt": [{"role": "user", "content": question}],
        "reward_model": {"ground_truth": answer, "style": "rule"},
        "extra_info": {
            "answer": answer,
            "index": index,
            "need_tools_kwargs": True,
            "question": question,
            "split": split,
            "tools_kwargs": {
                "name": "refocus",
                "metadata": json.dumps(_metadata() if metadata is None else metadata),
            },
        },
    }


def test_structural_signature_excludes_question_specific_focus_area() -> None:
    first = _metadata(focus=1)
    second = _metadata(focus=99)
    assert structural_chart_signature(first) == structural_chart_signature(second)
    assert structural_chart_signature(first) != structural_chart_signature(
        _metadata(y_values=["different"])
    )


def test_row_manifest_binds_target_prompt_and_structural_proxy() -> None:
    row = _row(0, split="train", question="How many?", answer="7")
    manifest = build_row_manifest(row, expected_split="train")
    assert manifest.row_id == "row-train-0"
    assert manifest.row_index == 0
    assert manifest.question_sha256 != manifest.answer_sha256
    assert manifest.tools_name == "refocus"


def test_row_manifest_fails_closed_on_split_target_and_source_mismatch() -> None:
    row = _row(0, split="train", question="How many?", answer="7")
    with pytest.raises(ValueError, match="expected"):
        build_row_manifest(row, expected_split="test")

    wrong_target = _row(0, split="train", question="How many?", answer="7")
    wrong_target["reward_model"]["ground_truth"] = "8"
    with pytest.raises(ValueError, match="target disagrees"):
        build_row_manifest(wrong_target, expected_split="train")

    wrong_source = _row(0, split="train", question="How many?", answer="7")
    wrong_source["source"] = "chartqa_v_bar"
    with pytest.raises(ValueError, match="source mismatch"):
        build_row_manifest(wrong_source, expected_split="train")

    missing_id = _row(0, split="train", question="How many?", answer="7")
    missing_id["id"] = None
    with pytest.raises(ValueError, match="row id"):
        build_row_manifest(missing_id, expected_split="train")


def test_split_audit_reports_conservative_structural_groups() -> None:
    train = audit_split_rows(
        [
            _row(0, split="train", question="Question A", answer="1"),
            _row(
                1,
                split="train",
                question="Question B",
                answer="2",
                metadata=_metadata(focus=3),
            ),
        ],
        split="train",
        dataset_revision="a" * 40,
        parquet_sha256="b" * 64,
    )
    assert train["rows"] == 2
    assert train["unique_structural_chart_signatures"] == 1
    assert train["max_rows_per_structural_chart"] == 2


def test_split_audit_rejects_noncontiguous_indices() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        audit_split_rows(
            [_row(1, split="train", question="Question", answer="1")],
            split="train",
            dataset_revision="a" * 40,
            parquet_sha256="b" * 64,
        )


def test_split_audit_rejects_unpinned_digests() -> None:
    row = _row(0, split="train", question="Question", answer="1")
    with pytest.raises(ValueError, match="dataset_revision"):
        audit_split_rows(
            [row],
            split="train",
            dataset_revision="main",
            parquet_sha256="b" * 64,
        )
    with pytest.raises(ValueError, match="parquet_sha256"):
        audit_split_rows(
            [row],
            split="train",
            dataset_revision="a" * 40,
            parquet_sha256="unknown",
        )


def test_remote_audit_runner_is_hard_coded_train_only() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "audit_refocus_chart_metadata.py"
    ).read_text(encoding="utf-8")
    assert '_read_split(args.dataset, args.revision, "train")' in script
    assert '_read_split(args.dataset, args.revision, "test")' not in script
    assert "--test-sha256" not in script
    assert '"test_accessed": False' in script


def test_chartqa_lineage_matches_only_train_png_stems() -> None:
    lineage = audit_chartqa_train_lineage(
        ["chart-a", "chart-b"],
        train_png_entries=[
            {"path": "chart-a.png", "type": "blob"},
            {"path": "nested/chart-b.PNG", "type": "blob"},
            {"path": "notes.txt", "type": "blob"},
        ],
        repository="vis-nlp/ChartQA",
        root_tree_sha="a" * 40,
        train_png_tree_sha="b" * 40,
    )
    assert lineage["matched_unique_row_ids"] == 2
    assert lineage["missing_unique_row_ids"] == 0
    assert lineage["pixel_identity_checked"] is False
    assert lineage["protected_split_contents_accessed"] is False
    assert (
        lineage["decision"]
        == "all_refocus_train_row_ids_match_pinned_chartqa_train_png_stems"
    )


def test_chartqa_lineage_fails_closed_on_missing_or_duplicate_png_stems() -> None:
    missing = audit_chartqa_train_lineage(
        ["chart-a", "chart-missing"],
        train_png_entries=[{"path": "chart-a.png", "type": "blob"}],
        repository="vis-nlp/ChartQA",
        root_tree_sha="a" * 40,
        train_png_tree_sha="b" * 40,
    )
    assert missing["decision"] == "refocus_train_lineage_not_fully_resolved"
    assert missing["missing_unique_row_ids"] == 1
    assert len(missing["missing_row_id_sha256"]) == 1

    with pytest.raises(ValueError, match="stems must be unique"):
        audit_chartqa_train_lineage(
            ["chart-a"],
            train_png_entries=[
                {"path": "chart-a.png", "type": "blob"},
                {"path": "nested/chart-a.png", "type": "blob"},
            ],
            repository="vis-nlp/ChartQA",
            root_tree_sha="a" * 40,
            train_png_tree_sha="b" * 40,
        )


def test_lineage_runner_traverses_only_the_train_tree() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "audit_refocus_chart_lineage.py"
    ).read_text(encoding="utf-8")
    assert 'TRAIN_TREE_COMPONENTS = ("ChartQA Dataset", "train", "png")' in script
    assert "test.parquet" not in script
    assert "validation/test subtree contents" in script
