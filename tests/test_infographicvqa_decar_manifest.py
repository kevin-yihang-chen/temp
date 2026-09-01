from __future__ import annotations

import hashlib
import io

import pytest

from beyond_entropy.infographicvqa_decar_manifest import (
    PILOT_IDENTITY_FIELDS,
    PAYLOAD_FIELDS,
    build_pilot_task_manifest,
    decoded_rgb_sha256,
)


def _image_bytes(color: tuple[int, int, int]) -> bytes:
    from PIL import Image

    image = Image.new("RGB", (12, 8), color)
    payload = io.BytesIO()
    image.save(payload, format="PNG")
    return payload.getvalue()


def _identity(rank: int, source: str, question: str, raw: bytes) -> dict[str, object]:
    rgb, width, height = decoded_rgb_sha256(raw)
    encoded = hashlib.sha256(raw).hexdigest()
    row: dict[str, object] = {
        "decoded_rgb_sha256": rgb,
        "encoded_sha256": encoded,
        "height": height,
        "image_id": rgb,
        "image_path": f"original-{source}.png",
        "question_id": question,
        "selection_rank": rank,
        "selection_sha256": f"selection-{rank}",
        "source_id": source,
        "transport_file": "train.parquet",
        "transport_row": rank,
        "width": width,
    }
    assert set(row) == PILOT_IDENTITY_FIELDS
    return row


def _payload(question: str, raw: bytes) -> dict[str, object]:
    row: dict[str, object] = {
        "questionId": question,
        "question": "What is shown?",
        "answers": ["A label"],
        "image": {"bytes": raw, "path": "original.png"},
        "data_split": "train",
    }
    assert set(row) == PAYLOAD_FIELDS
    return row


def test_build_pilot_task_manifest_preserves_identity_and_prompt_contract() -> None:
    first_image = _image_bytes((10, 20, 30))
    second_image = _image_bytes((40, 50, 60))
    identities = [
        _identity(0, "source-a", "q-0", first_image),
        _identity(1, "source-b", "q-1", second_image),
    ]
    payloads = {
        ("train.parquet", 0): _payload("q-0", first_image),
        ("train.parquet", 1): _payload("q-1", second_image),
    }
    report, tasks, image_rows, images = build_pilot_task_manifest(
        identities, payloads
    )
    assert report["population"] == {
        "questions": 2,
        "sources": 2,
        "images": 2,
        "answer_references": 2,
    }
    assert tasks[0]["state_id"] == "infovqa-train:q-0"
    assert tasks[0]["target"] == {"answers": ["A label"]}
    assert tasks[0]["model_prompt"].endswith(
        "\nAnswer the question using a single word or phrase."
    )
    assert len(image_rows) == len(images) == 2
    assert report["audits"]["task_outcomes_computed"] is False


def test_build_pilot_task_manifest_fails_closed_on_image_mismatch() -> None:
    expected = _image_bytes((10, 20, 30))
    changed = _image_bytes((11, 20, 30))
    identity = _identity(0, "source-a", "q-0", expected)
    with pytest.raises(ValueError, match="image identity changed"):
        build_pilot_task_manifest(
            [identity], {("train.parquet", 0): _payload("q-0", changed)}
        )
