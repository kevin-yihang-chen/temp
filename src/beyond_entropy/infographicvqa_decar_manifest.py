from __future__ import annotations

import hashlib
import io
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
