from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

from .rollout import AgentState, GroundTruth, TaskExample


def load_manifest(path: str | Path, *, limit: int | None = None) -> list[TaskExample]:
    """Load a small frozen benchmark slice from a portable JSONL manifest.

    Required fields are ``state_id``, ``image_path``, ``question``, and
    ``target``. Relative image paths are resolved against the manifest folder.
    ``image_id`` and ``source_id`` default to the image path and image id.
    """

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    manifest_path = Path(path).resolve()
    examples: list[TaskExample] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                state_id = str(value["state_id"])
                raw_image_path = Path(str(value["image_path"]))
                image_path = (
                    raw_image_path
                    if raw_image_path.is_absolute()
                    else manifest_path.parent / raw_image_path
                ).resolve()
                if not image_path.is_file():
                    raise ValueError(f"image does not exist: {image_path}")
                image_id = str(value.get("image_id", image_path))
                source_id = str(value.get("source_id", image_id))
                question = str(value["question"])
                if not state_id or not image_id or not source_id or not question:
                    raise ValueError("state and prompt identifiers must be non-empty")
                examples.append(
                    TaskExample(
                        state=AgentState(
                            state_id=state_id,
                            image_id=image_id,
                            source_id=source_id,
                            image_path=str(image_path),
                            question=question,
                        ),
                        ground_truth=GroundTruth(value["target"]),
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid benchmark example at {manifest_path}:{line_number}: {exc}"
                ) from exc
            if limit is not None and len(examples) >= limit:
                break
    if not examples:
        raise ValueError("benchmark manifest is empty")
    return examples


def extract_answer_letter(response: str) -> str | None:
    """Match the V*Bench answer extraction rules without importing lmms-eval."""

    cleaned = response.strip().upper()
    patterns = (
        r"^([A-D])\s*[\.)\]]*",
        r"(?:THE\s+)?(?:ANSWER|CHOICE|OPTION)(?:\s+IS)?[\s:]+([A-D])",
        r"\(([A-D])\)",
        r"([A-D])\s*(?:\.|\)|])",
        r"(?:^|\s)([A-D])(?:\s|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    letters = re.findall(r"[A-D]", cleaned)
    if len(letters) == 1:
        return letters[0]
    return cleaned[0] if cleaned and cleaned[0] in "ABCD" else None


def vstar_match(answer: str, ground_truth: GroundTruth) -> float:
    return float(
        extract_answer_letter(answer)
        == str(ground_truth.target).strip().upper()
    )


def _to_float(text: str) -> float | None:
    try:
        stripped = text.strip()
        if stripped.endswith("%"):
            return float(stripped[:-1]) / 100.0
        return float(stripped)
    except ValueError:
        return None


def chartqa_relaxed_match(answer: str, ground_truth: GroundTruth) -> float:
    prediction = answer.strip()
    target = str(ground_truth.target).strip()
    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float is not None:
        if target_float == 0.0:
            return float(abs(prediction_float) <= 1e-12)
        return float(abs(prediction_float - target_float) / abs(target_float) <= 0.05)
    return float(prediction.casefold() == target.casefold())


def scorer_by_name(name: str) -> Callable[[str, GroundTruth], float]:
    if name == "vstar":
        return vstar_match
    if name == "chartqa":
        return chartqa_relaxed_match
    raise ValueError(f"unsupported benchmark scorer: {name}")
