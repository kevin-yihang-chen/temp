from __future__ import annotations

from beyond_entropy.infographicvqa_decar_allocation import (
    PILOT_SOURCE_FIELDS,
    SOURCE_FIELDS,
    SourceSummary,
    assign_balanced_source_folds,
    build_decar_allocation,
)


def _row(source: str, question: str, image: str) -> dict[str, object]:
    row: dict[str, object] = {
        "decoded_rgb_sha256": image,
        "encoded_sha256": "encoded-" + image,
        "height": 200,
        "image_path": image + ".png",
        "normalized_hostname": source + ".example",
        "question_id": question,
        "source_id": source,
        "transport_file": "train.parquet",
        "transport_row": int(question.split("-")[-1]),
        "width": 100,
    }
    assert set(row) == SOURCE_FIELDS
    return row


def test_balanced_folds_are_deterministic_and_keep_sources_indivisible() -> None:
    sources = [
        SourceSummary(f"source-{index}", 10 - index, 1)
        for index in range(8)
    ]
    first = assign_balanced_source_folds(
        sources, n_folds=3, namespace="test", seed=17
    )
    second = assign_balanced_source_folds(
        list(reversed(sources)), n_folds=3, namespace="test", seed=17
    )
    assert first == second
    totals = [0, 0, 0]
    for source in sources:
        totals[first[source.source_id]] += source.question_count
    assert max(totals) - min(totals) <= max(source.question_count for source in sources)


def test_decar_allocation_builds_outer_inner_and_one_pilot_question() -> None:
    source_rows = []
    pilot_rows = []
    for source_index in range(6):
        source = f"source-{source_index}"
        for question_index in range(source_index + 1):
            source_rows.append(
                _row(source, f"q-{source_index}-{question_index}", f"im-{source_index}")
            )
        pilot_row = {
            "image_count": 1,
            "question_count": source_index + 1,
            "selection_rank": source_index,
            "source_id": source,
        }
        assert set(pilot_row) == PILOT_SOURCE_FIELDS
        pilot_rows.append(pilot_row)

    report, outer, inner, pilot = build_decar_allocation(
        source_rows,
        pilot_rows,
        n_outer_folds=3,
        n_inner_folds=2,
        seed=17,
    )
    assert len(outer) == 6
    assert len(inner) == 6 * 2
    assert len(pilot) == 6
    assert len({row["source_id"] for row in pilot}) == 6
    outer_by_source = {row["source_id"]: row["outer_fold"] for row in outer}
    assert all(
        outer_by_source[row["source_id"]] != row["outer_test_fold"]
        for row in inner
    )
    assert report["audits"] == {
        "source_fields_exact": True,
        "question_ids_unique": True,
        "outer_source_disjoint": True,
        "inner_excludes_outer_test_sources": True,
        "inner_context_coverage_exact": True,
        "pilot_one_question_per_source": True,
        "question_text_read": False,
        "answers_read": False,
        "task_outcomes_read": False,
        "validation_or_test_rows_read": False,
    }
    forbidden = {"question", "answers", "answer_type", "ocr", "target"}
    assert not any(forbidden & set(row) for row in outer + inner + pilot)
