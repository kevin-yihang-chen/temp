#!/usr/bin/env python3
"""Audit that rollout, NLL, and label-free features form a DECAR dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from beyond_entropy.dataset import read_jsonl
from beyond_entropy.infographicvqa_decar import (
    DECAR_ACTION_IDS,
    DECAR_SCALAR_NAMES,
    assemble_decar_dataset,
)
from beyond_entropy.qwen_semantic import load_semantic_feature_dataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked(path: Path, expected: str, name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or _sha256(resolved) != expected:
        raise ValueError(f"InfographicVQA DECAR input-audit {name} SHA-256 mismatch")
    return resolved


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rollouts", "answer-nll", "features"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--expected-{name}-sha256", required=True)
    parser.add_argument("--expected-decisions", type=int, required=True)
    parser.add_argument("--expected-sources", type=int, required=True)
    args = parser.parse_args()
    paths = {
        name: _checked(
            getattr(args, name), getattr(args, f"expected_{name}_sha256"), name
        )
        for name in ("rollouts", "answer_nll", "features")
    }

    records = read_jsonl(paths["rollouts"])
    features = load_semantic_feature_dataset(paths["features"])
    geometry: dict[str, tuple[int, int]] = {}
    from PIL import Image  # type: ignore[import-not-found]

    for record in records:
        if record.image_id in geometry:
            continue
        with Image.open(record.original_image) as image:
            geometry[record.image_id] = (int(image.size[0]), int(image.size[1]))
    dataset = assemble_decar_dataset(
        records,
        _jsonl(paths["answer_nll"]),
        features,
        geometry,
    )
    if (
        dataset.decisions != args.expected_decisions
        or len(set(dataset.source_ids)) != args.expected_sources
        or dataset.candidates != len(DECAR_ACTION_IDS)
        or dataset.scalars.shape[-1] != len(DECAR_SCALAR_NAMES)
    ):
        raise ValueError("InfographicVQA DECAR input-audit population changed")
    result = {
        "schema": "infographicvqa_decar_input_audit_v1",
        "passed": True,
        "decisions": dataset.decisions,
        "sources": len(set(dataset.source_ids)),
        "images": len(set(dataset.image_ids)),
        "actions_per_decision": dataset.candidates + 1,
        "question_embedding_dim": int(dataset.question.shape[-1]),
        "global_embedding_dim": int(dataset.global_visual.shape[-1]),
        "region_shape": [
            int(dataset.region.shape[1]),
            int(dataset.region.shape[2]),
        ],
        "scalar_dim": int(dataset.scalars.shape[-1]),
        "scalar_names": list(DECAR_SCALAR_NAMES),
        "generated_token_statistics_complete": True,
        "label_free_feature_storage": True,
        "inference_feature_outcomes_included": False,
        "scientific_endpoints_reported": False,
        "input_sha256": {name: _sha256(path) for name, path in paths.items()},
    }
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
