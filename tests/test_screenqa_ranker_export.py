from __future__ import annotations

import json

from scripts.export_screenqa_ranker_manifest import extract_selected_rows


def test_screenqa_ranker_extractor_deserializes_only_selected_objects(tmp_path):
    annotation = tmp_path / "train.json"
    annotation.write_text(
        '[{"image_id": 1, "question": invalid-unopened-target}, '
        '{"image_id": 2, "question": "selected?", "ground_truth": ["yes"]}]',
        encoding="utf-8",
    )
    rows, indices, source_count = extract_selected_rows(annotation, {"2"})
    assert rows == [{"image_id": 2, "question": "selected?", "ground_truth": ["yes"]}]
    assert indices == [1]
    assert source_count == 2


def test_screenqa_ranker_extractor_preserves_source_indices_and_all_questions(tmp_path):
    annotation = tmp_path / "train.json"
    payload = [
        {"image_id": 7, "question": "first?", "ground_truth": ["a"]},
        {"image_id": 8, "question": "middle?", "ground_truth": ["b"]},
        {"image_id": 7, "question": "third?", "ground_truth": ["c"]},
    ]
    annotation.write_text(json.dumps(payload), encoding="utf-8")
    rows, indices, source_count = extract_selected_rows(annotation, {"7"})
    assert [row["question"] for row in rows] == ["first?", "third?"]
    assert indices == [0, 2]
    assert source_count == 3
