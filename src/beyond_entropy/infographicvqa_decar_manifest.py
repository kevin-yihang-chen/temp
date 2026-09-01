from __future__ import annotations

import hashlib
import io
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .cross_benchmark import build_docvqa_prompt, docvqa_target


PILOT_IDENTITY_FIELDS = frozenset(
    {
        "decoded_rgb_sha256",
        "encoded_sha256",
        "height",
        "image_id",
        "image_path",
        "question_id",
        "selection_rank",
        "selection_sha256",
        "source_id",
        "transport_file",
        "transport_row",
        "width",
    }
)
FULL_IDENTITY_FIELDS = frozenset(
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
OUTER_FOLD_FIELDS = frozenset(
    {"image_count", "outer_fold", "question_count", "source_id", "tie_sha256"}
)
INNER_FOLD_FIELDS = frozenset(
    {"inner_fold", "outer_test_fold", "question_count", "source_id", "tie_sha256"}
)
PAYLOAD_FIELDS = frozenset(
    {"questionId", "question", "answers", "image", "data_split"}
)


def decoded_rgb_sha256(raw: bytes) -> tuple[str, int, int]:
    from PIL import Image, ImageFile  # type: ignore[import-not-found]

    if ImageFile.LOAD_TRUNCATED_IMAGES:
        raise RuntimeError("DECAR materialization forbids truncated-image recovery")
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        rgb = image.convert("RGB")
        width, height = rgb.size
        if width <= 0 or height <= 0:
            raise ValueError("DECAR materialization image dimensions are invalid")
        digest = hashlib.sha256()
        digest.update(width.to_bytes(8, "big"))
        digest.update(height.to_bytes(8, "big"))
        digest.update(rgb.tobytes())
        return digest.hexdigest(), width, height


def materialize_full_task_row(
    identity: Mapping[str, Any], payload: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    """Validate and materialize one registered full-train task row.

    The returned task deliberately excludes hostname, transport, and fold
    fields. Those identities remain available in their separately frozen
    manifests but can never become model inputs through this task manifest.
    """

    if set(identity) != FULL_IDENTITY_FIELDS:
        raise ValueError("DECAR full identity field inventory changed")
    if set(payload) != PAYLOAD_FIELDS:
        raise ValueError("DECAR full payload field inventory changed")
    question_id = str(identity["question_id"])
    source_id = str(identity["source_id"])
    if (
        not question_id
        or not source_id
        or str(payload["questionId"]) != question_id
        or str(payload["data_split"]) != "train"
    ):
        raise ValueError("DECAR full question identity, source, or split changed")
    question = str(payload["question"]).strip()
    raw_answers = payload["answers"]
    image = payload["image"]
    if (
        not question
        or isinstance(raw_answers, (str, bytes))
        or not isinstance(raw_answers, Sequence)
        or not raw_answers
        or not isinstance(image, Mapping)
    ):
        raise ValueError("DECAR full question, answers, or image are invalid")
    answers = [str(answer) for answer in raw_answers]
    if any(not answer.strip() for answer in answers):
        raise ValueError("DECAR full answers must be non-empty")
    raw_image = image.get("bytes")
    if not isinstance(raw_image, bytes) or not raw_image:
        raise ValueError("DECAR full encoded image bytes are missing")
    encoded = hashlib.sha256(raw_image).hexdigest()
    rgb, width, height = decoded_rgb_sha256(raw_image)
    if (
        encoded != str(identity["encoded_sha256"])
        or rgb != str(identity["decoded_rgb_sha256"])
        or width != int(identity["width"])
        or height != int(identity["height"])
    ):
        raise ValueError("DECAR full image identity changed")
    task = {
        "state_id": f"infovqa-train:{question_id}",
        "question_id": question_id,
        "image_id": rgb,
        "source_id": source_id,
        "image_path": f"images/{encoded}.img",
        "question": question,
        "model_prompt": build_docvqa_prompt(question),
        "target": docvqa_target(answers),
    }
    image_row = {
        "encoded_sha256": encoded,
        "decoded_rgb_sha256": rgb,
        "bytes": len(raw_image),
        "width": width,
        "height": height,
        "path": f"images/{encoded}.img",
    }
    return task, image_row, raw_image


def validate_decar_fold_manifests(
    identity_rows: Sequence[Mapping[str, Any]],
    outer_rows: Sequence[Mapping[str, Any]],
    inner_rows: Sequence[Mapping[str, Any]],
    *,
    n_outer_folds: int = 5,
    n_inner_folds: int = 4,
) -> dict[str, Any]:
    """Require exact source coverage and exclusion in the frozen folds."""

    if not identity_rows:
        raise ValueError("DECAR full identity manifest must be non-empty")
    questions_by_source: dict[str, set[str]] = {}
    images_by_source: dict[str, set[str]] = {}
    seen_questions: set[str] = set()
    for row in identity_rows:
        if set(row) != FULL_IDENTITY_FIELDS:
            raise ValueError("DECAR full identity field inventory changed")
        source_id = str(row["source_id"])
        question_id = str(row["question_id"])
        image_id = str(row["decoded_rgb_sha256"])
        if (
            not source_id
            or not question_id
            or not image_id
            or question_id in seen_questions
        ):
            raise ValueError("DECAR full identity values are invalid or duplicated")
        seen_questions.add(question_id)
        questions_by_source.setdefault(source_id, set()).add(question_id)
        images_by_source.setdefault(source_id, set()).add(image_id)
    sources = set(questions_by_source)
    if sources != set(images_by_source):
        raise RuntimeError("DECAR full source question/image coverage differs")

    outer_by_source: dict[str, int] = {}
    outer_question_counts = [0] * n_outer_folds
    outer_source_counts = [0] * n_outer_folds
    for row in outer_rows:
        if set(row) != OUTER_FOLD_FIELDS:
            raise ValueError("DECAR outer-fold field inventory changed")
        source_id = str(row["source_id"])
        fold = int(row["outer_fold"])
        if source_id in outer_by_source or source_id not in sources:
            raise ValueError("DECAR outer-fold source coverage is invalid")
        if fold < 0 or fold >= n_outer_folds:
            raise ValueError("DECAR outer fold is outside the registered range")
        if (
            int(row["question_count"]) != len(questions_by_source[source_id])
            or int(row["image_count"]) != len(images_by_source[source_id])
        ):
            raise ValueError("DECAR outer-fold source counts changed")
        outer_by_source[source_id] = fold
        outer_question_counts[fold] += len(questions_by_source[source_id])
        outer_source_counts[fold] += 1
    if set(outer_by_source) != sources:
        raise ValueError("DECAR outer-fold coverage is incomplete")

    observed_pairs: set[tuple[int, str]] = set()
    contexts_by_source: Counter[str] = Counter()
    inner_question_counts = [[0] * n_inner_folds for _ in range(n_outer_folds)]
    inner_source_counts = [[0] * n_inner_folds for _ in range(n_outer_folds)]
    for row in inner_rows:
        if set(row) != INNER_FOLD_FIELDS:
            raise ValueError("DECAR inner-fold field inventory changed")
        source_id = str(row["source_id"])
        outer_test = int(row["outer_test_fold"])
        inner_fold = int(row["inner_fold"])
        pair = (outer_test, source_id)
        if (
            source_id not in sources
            or pair in observed_pairs
            or outer_test < 0
            or outer_test >= n_outer_folds
            or inner_fold < 0
            or inner_fold >= n_inner_folds
            or outer_by_source[source_id] == outer_test
        ):
            raise ValueError("DECAR inner-fold source exclusion is invalid")
        if int(row["question_count"]) != len(questions_by_source[source_id]):
            raise ValueError("DECAR inner-fold source question count changed")
        observed_pairs.add(pair)
        contexts_by_source[source_id] += 1
        inner_question_counts[outer_test][inner_fold] += len(
            questions_by_source[source_id]
        )
        inner_source_counts[outer_test][inner_fold] += 1
    expected_pairs = {
        (outer_test, source_id)
        for source_id, held_out in outer_by_source.items()
        for outer_test in range(n_outer_folds)
        if outer_test != held_out
    }
    if observed_pairs != expected_pairs or set(contexts_by_source.values()) != {
        n_outer_folds - 1
    }:
        raise ValueError("DECAR inner-fold coverage is incomplete")
    return {
        "questions": len(seen_questions),
        "images": len(
            {str(row["decoded_rgb_sha256"]) for row in identity_rows}
        ),
        "sources": len(sources),
        "outer_rows": len(outer_rows),
        "inner_rows": len(inner_rows),
        "outer_question_counts": outer_question_counts,
        "outer_source_counts": outer_source_counts,
        "inner_question_counts": inner_question_counts,
        "inner_source_counts": inner_source_counts,
        "source_disjoint": True,
        "outer_test_sources_absent_from_inner_context": True,
    }


def build_pilot_task_manifest(
    identity_rows: Sequence[Mapping[str, Any]],
    payload_by_locator: Mapping[tuple[str, int], Mapping[str, Any]],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, bytes],
]:
    if not identity_rows:
        raise ValueError("DECAR pilot identity manifest must be non-empty")
    ranks: set[int] = set()
    questions: set[str] = set()
    sources: set[str] = set()
    task_rows: list[dict[str, Any]] = []
    image_bytes: dict[str, bytes] = {}
    for identity in sorted(identity_rows, key=lambda row: int(row["selection_rank"])):
        if set(identity) != PILOT_IDENTITY_FIELDS:
            raise ValueError("DECAR pilot identity field inventory changed")
        rank = int(identity["selection_rank"])
        question_id = str(identity["question_id"])
        source_id = str(identity["source_id"])
        locator = (str(identity["transport_file"]), int(identity["transport_row"]))
        if (
            rank in ranks
            or question_id in questions
            or source_id in sources
            or not question_id
            or not source_id
        ):
            raise ValueError("DECAR pilot identities must be one-to-one and non-empty")
        payload = payload_by_locator.get(locator)
        if payload is None or set(payload) != PAYLOAD_FIELDS:
            raise ValueError("DECAR pilot payload coverage or fields changed")
        if str(payload["questionId"]) != question_id or str(payload["data_split"]) != "train":
            raise ValueError("DECAR pilot question identity or split changed")
        question = str(payload["question"]).strip()
        raw_answers = payload["answers"]
        image = payload["image"]
        if (
            not question
            or isinstance(raw_answers, (str, bytes))
            or not isinstance(raw_answers, Sequence)
            or not raw_answers
            or not isinstance(image, Mapping)
        ):
            raise ValueError("DECAR pilot question, answers, or image are invalid")
        answers = [str(answer) for answer in raw_answers]
        if any(not answer.strip() for answer in answers):
            raise ValueError("DECAR pilot answers must be non-empty")
        raw_image = image.get("bytes")
        if not isinstance(raw_image, bytes) or not raw_image:
            raise ValueError("DECAR pilot encoded image bytes are missing")
        encoded = hashlib.sha256(raw_image).hexdigest()
        rgb, width, height = decoded_rgb_sha256(raw_image)
        if (
            encoded != str(identity["encoded_sha256"])
            or rgb != str(identity["decoded_rgb_sha256"])
            or rgb != str(identity["image_id"])
            or width != int(identity["width"])
            or height != int(identity["height"])
        ):
            raise ValueError("DECAR pilot image identity changed")
        prior = image_bytes.setdefault(encoded, raw_image)
        if prior != raw_image:
            raise RuntimeError("DECAR encoded-image hash collision")
        ranks.add(rank)
        questions.add(question_id)
        sources.add(source_id)
        task_rows.append(
            {
                "selection_rank": rank,
                "state_id": f"infovqa-train:{question_id}",
                "question_id": question_id,
                "image_id": rgb,
                "source_id": source_id,
                "image_path": f"images/{encoded}.img",
                "question": question,
                "model_prompt": build_docvqa_prompt(question),
                "target": docvqa_target(answers),
            }
        )
    if ranks != set(range(len(identity_rows))):
        raise ValueError("DECAR pilot selection ranks must be contiguous from zero")
    if set(payload_by_locator) != {
        (str(row["transport_file"]), int(row["transport_row"]))
        for row in identity_rows
    }:
        raise ValueError("DECAR pilot materializer received extra payload rows")
    image_rows = [
        {
            "encoded_sha256": encoded,
            "bytes": len(raw),
            "path": f"images/{encoded}.img",
        }
        for encoded, raw in sorted(image_bytes.items())
    ]
    report = {
        "schema": "infographicvqa_decar_pilot_materialization_v1",
        "scientific_status": (
            "registered 512-source train engineering pilot; endpoints cannot "
            "select scientific or hardware settings"
        ),
        "population": {
            "questions": len(task_rows),
            "sources": len(sources),
            "images": len(image_rows),
            "answer_references": sum(
                len(row["target"]["answers"]) for row in task_rows
            ),
        },
        "columns_read": ["questionId", "question", "answers", "image", "data_split"],
        "columns_not_read": [
            "answer_type",
            "image_url",
            "operation/reasoning",
            "ocr",
        ],
        "audits": {
            "identity_coverage_exact": True,
            "one_question_per_source": len(task_rows) == len(sources),
            "selection_ranks_exact": True,
            "encoded_and_decoded_image_hashes_exact": True,
            "all_split_markers_train": True,
            "question_text_read": True,
            "answers_read": True,
            "task_outcomes_computed": False,
            "teacher_likelihood_computed": False,
            "validation_or_test_rows_read": False,
        },
    }
    return report, task_rows, image_rows, image_bytes
