from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


OUTER_NAMESPACE = "infovqa-decar-outer-v1"
INNER_NAMESPACE = "infovqa-decar-inner-v1"
PILOT_QUESTION_NAMESPACE = "infovqa-decar-pilot-question-v1"
DECAR_SEED = 20_260_917
SOURCE_FIELDS = frozenset(
    {
        "decoded_rgb_sha256",
        "encoded_sha256",
        "height",
        "image_path",
        "normalized_hostname",
        "question_id",
        "source_id",
        "transport_file",
        "transport_row",
        "width",
    }
)
PILOT_SOURCE_FIELDS = frozenset(
    {"image_count", "question_count", "selection_rank", "source_id"}
)


@dataclass(frozen=True)
class SourceSummary:
    source_id: str
    question_count: int
    image_count: int


def _digest(namespace: str, seed: int, *parts: object) -> str:
    payload = "\0".join((namespace, str(seed), *(str(part) for part in parts)))
    return hashlib.sha256(payload.encode()).hexdigest()


def summarize_sources(rows: Sequence[Mapping[str, Any]]) -> list[SourceSummary]:
    if not rows:
        raise ValueError("InfographicVQA source allocation requires rows")
    questions: dict[str, set[str]] = defaultdict(set)
    images: dict[str, set[str]] = defaultdict(set)
    seen_questions: set[str] = set()
    for row in rows:
        if set(row) != SOURCE_FIELDS:
            raise ValueError("InfographicVQA source-manifest field inventory changed")
        source_id = str(row["source_id"])
        question_id = str(row["question_id"])
        image_id = str(row["decoded_rgb_sha256"])
        if not source_id or not question_id or not image_id:
            raise ValueError("InfographicVQA allocation identities must be non-empty")
        if question_id in seen_questions:
            raise ValueError("InfographicVQA allocation question IDs must be unique")
        seen_questions.add(question_id)
        questions[source_id].add(question_id)
        images[source_id].add(image_id)
    if set(questions) != set(images):
        raise RuntimeError("InfographicVQA allocation source coverage differs")
    return [
        SourceSummary(
            source_id=source_id,
            question_count=len(questions[source_id]),
            image_count=len(images[source_id]),
        )
        for source_id in sorted(questions)
    ]


def assign_balanced_source_folds(
    sources: Sequence[SourceSummary],
    *,
    n_folds: int,
    namespace: str,
    seed: int,
    prefix_parts: Sequence[object] = (),
) -> dict[str, int]:
    if n_folds < 2 or len(sources) < n_folds:
        raise ValueError("source-balanced allocation requires at least two folds")
    if len({source.source_id for source in sources}) != len(sources):
        raise ValueError("source-balanced allocation source IDs must be unique")
    ordered = sorted(
        sources,
        key=lambda source: (
            -source.question_count,
            _digest(namespace, seed, *prefix_parts, source.source_id),
            source.source_id,
        ),
    )
    question_totals = [0] * n_folds
    source_totals = [0] * n_folds
    assignments: dict[str, int] = {}
    for source in ordered:
        fold = min(
            range(n_folds),
            key=lambda index: (
                question_totals[index], source_totals[index], index
            ),
        )
        assignments[source.source_id] = fold
        question_totals[fold] += source.question_count
        source_totals[fold] += 1
    return assignments


def _fold_summary(
    sources: Sequence[SourceSummary], assignments: Mapping[str, int], n_folds: int
) -> list[dict[str, int]]:
    questions = [0] * n_folds
    source_counts = [0] * n_folds
    for source in sources:
        fold = int(assignments[source.source_id])
        if fold < 0 or fold >= n_folds:
            raise ValueError("InfographicVQA fold index is outside its contract")
        questions[fold] += source.question_count
        source_counts[fold] += 1
    return [
        {
            "fold": fold,
            "question_count": questions[fold],
            "source_count": source_counts[fold],
        }
        for fold in range(n_folds)
    ]


def build_decar_allocation(
    source_rows: Sequence[Mapping[str, Any]],
    pilot_source_rows: Sequence[Mapping[str, Any]],
    *,
    n_outer_folds: int = 5,
    n_inner_folds: int = 4,
    seed: int = DECAR_SEED,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    sources = summarize_sources(source_rows)
    source_by_id = {source.source_id: source for source in sources}
    outer = assign_balanced_source_folds(
        sources,
        n_folds=n_outer_folds,
        namespace=OUTER_NAMESPACE,
        seed=seed,
    )
    outer_rows = [
        {
            "source_id": source.source_id,
            "question_count": source.question_count,
            "image_count": source.image_count,
            "outer_fold": outer[source.source_id],
            "tie_sha256": _digest(OUTER_NAMESPACE, seed, source.source_id),
        }
        for source in sources
    ]

    inner_rows: list[dict[str, Any]] = []
    inner_report: list[dict[str, Any]] = []
    for outer_test_fold in range(n_outer_folds):
        training_sources = [
            source
            for source in sources
            if outer[source.source_id] != outer_test_fold
        ]
        assignments = assign_balanced_source_folds(
            training_sources,
            n_folds=n_inner_folds,
            namespace=INNER_NAMESPACE,
            seed=seed,
            prefix_parts=(outer_test_fold,),
        )
        inner_rows.extend(
            {
                "outer_test_fold": outer_test_fold,
                "source_id": source.source_id,
                "inner_fold": assignments[source.source_id],
                "question_count": source.question_count,
                "tie_sha256": _digest(
                    INNER_NAMESPACE, seed, outer_test_fold, source.source_id
                ),
            }
            for source in sorted(training_sources, key=lambda value: value.source_id)
        )
        inner_report.append(
            {
                "outer_test_fold": outer_test_fold,
                "folds": _fold_summary(
                    training_sources, assignments, n_inner_folds
                ),
            }
        )

    pilot_rank_by_source: dict[str, int] = {}
    for row in pilot_source_rows:
        if set(row) != PILOT_SOURCE_FIELDS:
            raise ValueError("InfographicVQA pilot-source field inventory changed")
        source_id = str(row["source_id"])
        rank = int(row["selection_rank"])
        if source_id not in source_by_id or source_id in pilot_rank_by_source:
            raise ValueError("InfographicVQA pilot source coverage is invalid")
        source = source_by_id[source_id]
        if (
            int(row["question_count"]) != source.question_count
            or int(row["image_count"]) != source.image_count
        ):
            raise ValueError("InfographicVQA pilot source counts changed")
        pilot_rank_by_source[source_id] = rank
    if sorted(pilot_rank_by_source.values()) != list(range(len(pilot_rank_by_source))):
        raise ValueError("InfographicVQA pilot ranks must be contiguous from zero")

    rows_by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in source_rows:
        source_id = str(row["source_id"])
        if source_id in pilot_rank_by_source:
            rows_by_source[source_id].append(row)
    pilot_questions: list[dict[str, Any]] = []
    for source_id, selection_rank in sorted(
        pilot_rank_by_source.items(), key=lambda item: item[1]
    ):
        choices = rows_by_source[source_id]
        if not choices:
            raise RuntimeError("InfographicVQA pilot source has no questions")
        selected = min(
            choices,
            key=lambda row: (
                _digest(
                    PILOT_QUESTION_NAMESPACE,
                    seed,
                    source_id,
                    row["question_id"],
                ),
                str(row["question_id"]),
            ),
        )
        pilot_questions.append(
            {
                "selection_rank": selection_rank,
                "source_id": source_id,
                "question_id": str(selected["question_id"]),
                "image_id": str(selected["decoded_rgb_sha256"]),
                "image_path": str(selected["image_path"]),
                "encoded_sha256": str(selected["encoded_sha256"]),
                "decoded_rgb_sha256": str(selected["decoded_rgb_sha256"]),
                "width": int(selected["width"]),
                "height": int(selected["height"]),
                "transport_file": str(selected["transport_file"]),
                "transport_row": int(selected["transport_row"]),
                "selection_sha256": _digest(
                    PILOT_QUESTION_NAMESPACE,
                    seed,
                    source_id,
                    selected["question_id"],
                ),
            }
        )

    report = {
        "schema": "infographicvqa_decar_identity_allocation_v1",
        "scientific_status": (
            "outcome-blind outer/inner source folds and engineering-pilot "
            "question identities"
        ),
        "population": {
            "questions": len(source_rows),
            "sources": len(sources),
            "images": sum(source.image_count for source in sources),
            "pilot_questions": len(pilot_questions),
            "pilot_sources": len(pilot_rank_by_source),
        },
        "algorithm": {
            "seed": seed,
            "outer_namespace": OUTER_NAMESPACE,
            "inner_namespace": INNER_NAMESPACE,
            "pilot_question_namespace": PILOT_QUESTION_NAMESPACE,
            "balance_order": (
                "descending question count, hashed tie, source ID; assign minimum "
                "question total, source total, fold index"
            ),
        },
        "outer_folds": _fold_summary(sources, outer, n_outer_folds),
        "inner_folds": inner_report,
        "audits": {
            "source_fields_exact": True,
            "question_ids_unique": True,
            "outer_source_disjoint": len(outer) == len(sources),
            "inner_excludes_outer_test_sources": all(
                outer[row["source_id"]] != row["outer_test_fold"]
                for row in inner_rows
            ),
            "inner_context_coverage_exact": len(inner_rows)
            == len(sources) * (n_outer_folds - 1),
            "pilot_one_question_per_source": len(pilot_questions)
            == len(pilot_rank_by_source),
            "question_text_read": False,
            "answers_read": False,
            "task_outcomes_read": False,
            "validation_or_test_rows_read": False,
        },
    }
    return report, outer_rows, inner_rows, pilot_questions
