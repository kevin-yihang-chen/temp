from __future__ import annotations

import hashlib
import importlib.metadata
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class BenchmarkSpec:
    dataset_id: str
    default_revision: str
    split: str
    group_field: str


BENCHMARK_SPECS: dict[str, BenchmarkSpec] = {
    "vstar": BenchmarkSpec(
        dataset_id="lmms-lab/vstar-bench",
        default_revision="b44023b4dca749ed8a76b85eb576627d05a1c174",
        split="test",
        group_field="category",
    ),
    "chartqa": BenchmarkSpec(
        dataset_id="lmms-lab/ChartQA",
        default_revision="9e63b7df1592a1c2158e735cc1725454aef0d6d9",
        split="test",
        group_field="type",
    ),
}


def stratified_sample_indices(
    labels: Sequence[str],
    *,
    count: int,
    seed: int,
) -> list[int]:
    """Choose a deterministic round-robin sample balanced across labels."""

    if count <= 0:
        raise ValueError("count must be positive")
    if count > len(labels):
        raise ValueError("count cannot exceed the dataset size")
    grouped: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        normalized = str(label).strip()
        if not normalized:
            raise ValueError(f"empty stratum label at index {index}")
        grouped.setdefault(normalized, []).append(index)
    if not grouped:
        raise ValueError("labels must not be empty")
    rng = random.Random(seed)
    for label in sorted(grouped):
        rng.shuffle(grouped[label])
    selected: list[int] = []
    offsets = {label: 0 for label in grouped}
    while len(selected) < count:
        made_progress = False
        for label in sorted(grouped):
            offset = offsets[label]
            if offset >= len(grouped[label]):
                continue
            selected.append(grouped[label][offset])
            offsets[label] += 1
            made_progress = True
            if len(selected) == count:
                break
        if not made_progress:  # pragma: no cover - guarded by count validation
            raise RuntimeError("could not construct the requested stratified sample")
    return sorted(selected)


def _image_digest(image: Any) -> str:
    width, height = image.size
    digest = hashlib.sha256()
    digest.update(f"RGB:{width}x{height}:".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def _manifest_payload(
    row: Mapping[str, Any],
    *,
    task: str,
    source_index: int,
    image_id: str,
    image_path: str,
    dataset_revision: str,
) -> dict[str, Any]:
    if task == "vstar":
        question_id = str(row["question_id"])
        category = str(row["category"])
        question = str(row["text"]).strip()
        answer_instruction = (
            "Answer with the option's letter from the given choices directly."
        )
        if answer_instruction.casefold() not in question.casefold():
            question = question + "\n" + answer_instruction
        return {
            "state_id": f"vstar:{question_id}",
            "image_id": image_id,
            "source_id": f"vstar:{question_id}",
            "image_path": image_path,
            "question": question,
            "target": str(row["label"]).strip().upper(),
            "benchmark": "vstar",
            "stratum": category,
            "source_index": source_index,
            "dataset_revision": dataset_revision,
        }
    if task == "chartqa":
        chart_type = str(row["type"])
        return {
            "state_id": f"chartqa:{source_index:05d}",
            "image_id": image_id,
            "source_id": f"chartqa:{source_index:05d}",
            "image_path": image_path,
            "question": str(row["question"]).strip() + " Answer:",
            "target": str(row["answer"]).strip(),
            "benchmark": "chartqa",
            "stratum": chart_type,
            "source_index": source_index,
            "dataset_revision": dataset_revision,
        }
    raise ValueError(f"unsupported benchmark task: {task}")


def export_benchmark_manifest(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_indices: Sequence[int],
    task: str,
    dataset_id: str,
    dataset_revision: str,
    output_dir: str | Path,
    seed: int,
) -> dict[str, Any]:
    """Save decoded images and a frozen manifest with a provenance sidecar."""

    if task not in BENCHMARK_SPECS:
        raise ValueError(f"unsupported benchmark task: {task}")
    if len(rows) != len(source_indices):
        raise ValueError("rows and source_indices must have the same length")
    if not rows:
        raise ValueError("rows must not be empty")
    destination = Path(output_dir).resolve()
    image_dir = destination / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    payloads: list[dict[str, Any]] = []
    for row, source_index in zip(rows, source_indices):
        image = row["image"].convert("RGB")
        image_id = _image_digest(image)
        image_name = f"{image_id}.png"
        image_destination = image_dir / image_name
        if not image_destination.exists():
            image.save(image_destination, format="PNG")
        payloads.append(
            _manifest_payload(
                row,
                task=task,
                source_index=source_index,
                image_id=image_id,
                image_path=f"images/{image_name}",
                dataset_revision=dataset_revision,
            )
        )
    state_ids = [str(payload["state_id"]) for payload in payloads]
    if len(state_ids) != len(set(state_ids)):
        raise ValueError("export produced duplicate state_id values")
    manifest_path = destination / "manifest.jsonl"
    serialized_lines = [
        json.dumps(payload, ensure_ascii=False, sort_keys=True) for payload in payloads
    ]
    manifest_bytes = ("\n".join(serialized_lines) + "\n").encode()
    manifest_path.write_bytes(manifest_bytes)
    stratum_counts: dict[str, int] = {}
    for payload in payloads:
        stratum = str(payload["stratum"])
        stratum_counts[stratum] = stratum_counts.get(stratum, 0) + 1
    provenance = {
        "task": task,
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "split": BENCHMARK_SPECS[task].split,
        "selection": "seeded round-robin stratified sample",
        "seed": seed,
        "count": len(payloads),
        "source_indices": list(source_indices),
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "unique_images": len({payload["image_id"] for payload in payloads}),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "python": sys.version.split()[0],
        "packages": {},
    }
    packages = provenance["packages"]
    assert isinstance(packages, dict)
    for package in ("datasets", "Pillow"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    provenance_path = destination / "manifest.provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "manifest": str(manifest_path),
        "provenance": str(provenance_path),
        **provenance,
    }
