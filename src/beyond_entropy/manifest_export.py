from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import io
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .cross_benchmark import (
    build_docvqa_prompt,
    build_hrbench_context,
    build_hrbench_prompt,
    build_screenqa_prompt,
    build_textvqa_prompt,
    docvqa_target,
    hrbench_target,
    screenqa_target,
    textvqa_target,
)


@dataclass(frozen=True)
class BenchmarkSpec:
    dataset_id: str
    default_revision: str
    split: str
    scorer: str
    selection_fields: tuple[str, ...]
    source_fields: tuple[str, ...] = ()
    dataset_name: str | None = None


BENCHMARK_SPECS: dict[str, BenchmarkSpec] = {
    "vstar": BenchmarkSpec(
        dataset_id="lmms-lab/vstar-bench",
        default_revision="b44023b4dca749ed8a76b85eb576627d05a1c174",
        split="test",
        scorer="vstar",
        selection_fields=("category",),
    ),
    "chartqa": BenchmarkSpec(
        dataset_id="lmms-lab/ChartQA",
        default_revision="9e63b7df1592a1c2158e735cc1725454aef0d6d9",
        split="test",
        scorer="chartqa",
        selection_fields=("type",),
    ),
    "docvqa": BenchmarkSpec(
        dataset_id="lmms-lab/DocVQA",
        dataset_name="DocVQA",
        default_revision="539088ef8a8ada01ac8e2e6d4e372586748a265e",
        split="validation",
        scorer="docvqa",
        selection_fields=("question_types",),
        source_fields=("docId",),
    ),
    "textvqa": BenchmarkSpec(
        dataset_id="lmms-lab/textvqa",
        default_revision="9c0699cd19768ac5ab97568f6b3cbac4c0062884",
        split="validation",
        scorer="textvqa",
        selection_fields=("ocr_tokens",),
        source_fields=("image_id",),
    ),
    "screenqa": BenchmarkSpec(
        dataset_id="google-research-datasets/screen_qa",
        default_revision="1dfdbccaf56948821b5fa8ffe5d186fe4751e46d",
        split="train",
        scorer="screenqa",
        selection_fields=(),
        source_fields=("source_group_id",),
    ),
    "hrbench4k": BenchmarkSpec(
        dataset_id="DreamMr/HR-Bench",
        dataset_name="hrbench_version_split",
        default_revision="83b9013d6293b85dc507e87199ca52517536939c",
        split="hrbench_4k",
        scorer="hrbench",
        selection_fields=("category", "cycle_category"),
        source_fields=("index",),
    ),
    "hrbench8k": BenchmarkSpec(
        dataset_id="DreamMr/HR-Bench",
        dataset_name="hrbench_version_split",
        default_revision="83b9013d6293b85dc507e87199ca52517536939c",
        split="hrbench_8k",
        scorer="hrbench",
        selection_fields=("category", "cycle_category"),
        source_fields=("index",),
    ),
}


def benchmark_stratum(row: Mapping[str, Any], *, task: str) -> str:
    """Return the pre-outcome stratum used for deterministic slice selection."""

    if task == "vstar":
        value = str(row["category"]).strip()
    elif task == "chartqa":
        value = str(row["type"]).strip()
    elif task == "docvqa":
        raw_types = row["question_types"]
        if isinstance(raw_types, (str, bytes)) or not isinstance(raw_types, Sequence):
            raise ValueError("DocVQA question_types must be a sequence")
        types = sorted({str(item).strip() for item in raw_types if str(item).strip()})
        value = "+".join(types)
    elif task == "textvqa":
        raw_tokens = row["ocr_tokens"]
        if isinstance(raw_tokens, (str, bytes)) or not isinstance(raw_tokens, Sequence):
            raise ValueError("TextVQA ocr_tokens must be a sequence")
        count = len(raw_tokens)
        if count == 0:
            value = "ocr-000"
        elif count <= 5:
            value = "ocr-001-005"
        elif count <= 15:
            value = "ocr-006-015"
        else:
            value = "ocr-016-plus"
    elif task == "screenqa":
        value = "allocated-app-duplicate-component"
    elif task in {"hrbench4k", "hrbench8k"}:
        category = str(row["category"]).strip()
        cycle_category = str(row["cycle_category"]).strip()
        value = f"{category}:{cycle_category}"
    else:
        raise ValueError(f"unsupported benchmark task: {task}")
    if not value:
        raise ValueError(f"empty benchmark stratum for task {task}")
    return value


def benchmark_source_group(row: Mapping[str, Any], *, task: str) -> str:
    """Return the public, pre-outcome source identifier used for split isolation."""

    if task == "docvqa":
        value = str(row["docId"]).strip()
    elif task == "textvqa":
        value = str(row["image_id"]).strip()
    elif task == "screenqa":
        value = str(row["source_group_id"]).strip()
    elif task in {"hrbench4k", "hrbench8k"}:
        value = str(row["index"]).strip()
    else:
        raise ValueError(f"task {task} has no registered source-group field")
    if not value:
        raise ValueError(f"empty source group for task {task}")
    return value


def hash_ranked_source_group_indices(
    group_ids: Sequence[str],
    *,
    count: int,
    offset: int = 0,
    seed: int,
    namespace: str,
    excluded_groups: Sequence[str] = (),
) -> list[int]:
    """Select whole source groups by an order-independent SHA-256 ranking."""

    if count <= 0:
        raise ValueError("source-group count must be positive")
    if offset < 0:
        raise ValueError("source-group offset must be non-negative")
    normalized_namespace = str(namespace).strip()
    if not normalized_namespace:
        raise ValueError("source-group namespace must be non-empty")
    normalized_groups = [str(group_id).strip() for group_id in group_ids]
    if not normalized_groups or any(not group_id for group_id in normalized_groups):
        raise ValueError("source group IDs must be non-empty")
    excluded = {str(group_id).strip() for group_id in excluded_groups}
    if "" in excluded:
        raise ValueError("excluded source group IDs must be non-empty")
    unique_groups = set(normalized_groups)

    def rank(group_id: str) -> tuple[str, str]:
        payload = f"{normalized_namespace}\0{seed}\0{group_id}".encode()
        return hashlib.sha256(payload).hexdigest(), group_id

    ordered_groups = sorted(unique_groups, key=rank)
    eligible_groups = [
        group_id for group_id in ordered_groups[offset:] if group_id not in excluded
    ]
    if count > len(eligible_groups):
        raise ValueError(
            f"source-group slice requires {count} eligible groups after offset "
            f"{offset}, but only {len(eligible_groups)} are available"
        )
    selected = set(eligible_groups[:count])
    return [
        index for index, group_id in enumerate(normalized_groups) if group_id in selected
    ]


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


def stratified_unique_group_sample_indices(
    labels: Sequence[str],
    group_ids: Sequence[str],
    *,
    count: int,
    seed: int,
) -> list[int]:
    """Choose a balanced deterministic sample with at most one row per group."""

    if len(labels) != len(group_ids):
        raise ValueError("labels and group_ids must have the same length")
    if count <= 0:
        raise ValueError("count must be positive")
    grouped: dict[str, list[int]] = {}
    for index, (label, group_id) in enumerate(zip(labels, group_ids)):
        normalized_label = str(label).strip()
        normalized_group = str(group_id).strip()
        if not normalized_label or not normalized_group:
            raise ValueError(f"empty label or group ID at index {index}")
        grouped.setdefault(normalized_label, []).append(index)
    rng = random.Random(seed)
    for label in sorted(grouped):
        rng.shuffle(grouped[label])
    offsets = {label: 0 for label in grouped}
    selected: list[int] = []
    selected_groups: set[str] = set()
    while len(selected) < count:
        made_progress = False
        for label in sorted(grouped):
            candidates = grouped[label]
            while offsets[label] < len(candidates):
                index = candidates[offsets[label]]
                offsets[label] += 1
                group_id = str(group_ids[index]).strip()
                if group_id in selected_groups:
                    continue
                selected.append(index)
                selected_groups.add(group_id)
                made_progress = True
                break
            if len(selected) == count:
                break
        if not made_progress:
            raise ValueError(
                f"count {count} exceeds the available unique groups under stratification"
            )
    return sorted(selected)


def image_digest(image: Any) -> str:
    width, height = image.size
    digest = hashlib.sha256()
    digest.update(f"RGB:{width}x{height}:".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def _decode_row_image(row: Mapping[str, Any], *, task: str) -> Any:
    raw_image = row["image"]
    if task == "screenqa" and isinstance(raw_image, (str, Path)):
        try:
            from PIL import Image  # type: ignore[import-untyped]

            with Image.open(raw_image) as image:
                return image.convert("RGB")
        except Exception as exc:
            raise ValueError("could not decode ScreenQA image path") from exc
    if task in {"hrbench4k", "hrbench8k"}:
        if not isinstance(raw_image, str) or not raw_image.strip():
            raise ValueError("HRBench image must be a non-empty base64 string")
        try:
            image_bytes = base64.b64decode(raw_image)
            from PIL import Image  # type: ignore[import-untyped]

            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise ValueError("could not decode HRBench base64 image") from exc
    convert = getattr(raw_image, "convert", None)
    if not callable(convert):
        raise ValueError(f"{task} image must provide convert('RGB')")
    return convert("RGB")


def _hrbench_options(row: Mapping[str, Any]) -> dict[str, Any]:
    options = {key: row[key] for key in ("A", "B", "C", "D") if key in row}
    if len(options) < 2:
        raise ValueError("HRBench row must contain at least two options")
    return options


def _manifest_payload(
    row: Mapping[str, Any],
    *,
    task: str,
    source_index: int,
    image_id: str,
    image_path: str,
    dataset_revision: str,
    state_namespace: str | None = None,
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
        namespace = state_namespace or "chartqa"
        return {
            "state_id": f"{namespace}:{source_index:05d}",
            "image_id": image_id,
            "source_id": f"{namespace}:{source_index:05d}",
            "image_path": image_path,
            "question": str(row["question"]).strip() + " Answer:",
            "target": str(row["answer"]).strip(),
            "benchmark": "chartqa",
            "stratum": chart_type,
            "source_index": source_index,
            "dataset_revision": dataset_revision,
        }
    if task == "docvqa":
        question_id = str(row["questionId"]).strip()
        question = str(row["question"]).strip()
        if not question_id or not question:
            raise ValueError("DocVQA questionId and question must be non-empty")
        raw_document_id = str(row.get("docId", image_id)).strip()
        document_id = raw_document_id or image_id
        namespace = state_namespace or "docvqa"
        return {
            "state_id": f"{namespace}:{question_id}",
            "image_id": image_id,
            "source_id": f"docvqa:{document_id}",
            "image_path": image_path,
            "question": question,
            "model_prompt": build_docvqa_prompt(question),
            "target": docvqa_target(row["answers"]),
            "benchmark": "docvqa",
            "stratum": benchmark_stratum(row, task=task),
            "source_index": source_index,
            "dataset_revision": dataset_revision,
        }
    if task == "textvqa":
        question_id = str(row["question_id"]).strip()
        raw_image_id = str(row["image_id"]).strip()
        question = str(row["question"]).strip()
        if not question_id or not raw_image_id or not question:
            raise ValueError(
                "TextVQA question_id, image_id, and question must be non-empty"
            )
        namespace = state_namespace or "textvqa"
        return {
            "state_id": f"{namespace}:{question_id}",
            "image_id": image_id,
            "source_id": f"textvqa:{raw_image_id}",
            "image_path": image_path,
            "question": question,
            "model_prompt": build_textvqa_prompt(question, row["ocr_tokens"]),
            "target": textvqa_target(row["answers"]),
            "benchmark": "textvqa",
            "stratum": benchmark_stratum(row, task=task),
            "source_index": source_index,
            "dataset_revision": dataset_revision,
        }
    if task == "screenqa":
        raw_image_id = str(row["image_id"]).strip()
        source_group_id = str(row["source_group_id"]).strip()
        question = str(row["question"]).strip()
        if not raw_image_id or not source_group_id or not question:
            raise ValueError(
                "ScreenQA image_id, source_group_id, and question must be non-empty"
            )
        namespace = state_namespace or "screenqa"
        return {
            "state_id": f"{namespace}:{source_index}",
            "image_id": image_id,
            "source_id": f"screenqa:{source_group_id}",
            "image_path": image_path,
            "question": question,
            "model_prompt": build_screenqa_prompt(question),
            "target": screenqa_target(row["ground_truth"]),
            "benchmark": "screenqa",
            "stratum": benchmark_stratum(row, task=task),
            "source_index": source_index,
            "rico_ui_id": raw_image_id,
            "dataset_revision": dataset_revision,
        }
    if task in {"hrbench4k", "hrbench8k"}:
        index = str(row["index"]).strip()
        question = str(row["question"]).strip()
        if not index or not question:
            raise ValueError("HRBench index and question must be non-empty")
        options = _hrbench_options(row)
        namespace = state_namespace or task
        return {
            "state_id": f"{namespace}:{index}",
            "image_id": image_id,
            "source_id": f"hrbench:{index}",
            "image_path": image_path,
            "question": build_hrbench_context(question, options),
            "model_prompt": build_hrbench_prompt(question, options),
            "target": hrbench_target(
                row["answer"],
                category=row["category"],
                cycle_category=row["cycle_category"],
            ),
            "benchmark": task,
            "stratum": benchmark_stratum(row, task=task),
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
    state_namespace: str | None = None,
    dataset_split: str | None = None,
    selection: str = "seeded round-robin stratified sample",
    selection_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Save decoded images and a frozen manifest with a provenance sidecar."""

    if task not in BENCHMARK_SPECS:
        raise ValueError(f"unsupported benchmark task: {task}")
    if len(rows) != len(source_indices):
        raise ValueError("rows and source_indices must have the same length")
    if not rows:
        raise ValueError("rows must not be empty")
    resolved_split = (
        BENCHMARK_SPECS[task].split
        if dataset_split is None
        else str(dataset_split).strip()
    )
    if not resolved_split:
        raise ValueError("dataset_split must be non-empty")
    destination = Path(output_dir).resolve()
    image_dir = destination / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    payloads: list[dict[str, Any]] = []
    screenqa_image_cache: dict[str, tuple[str, str]] = {}
    for row, source_index in zip(rows, source_indices):
        screenqa_raw_id = str(row.get("image_id", "")).strip() if task == "screenqa" else ""
        cached_image = screenqa_image_cache.get(screenqa_raw_id)
        if cached_image is None:
            image = _decode_row_image(row, task=task)
            image_id = image_digest(image)
            image_name = f"{image_id}.png"
            image_destination = image_dir / image_name
            if not image_destination.exists():
                image.save(image_destination, format="PNG")
            if task == "screenqa":
                screenqa_image_cache[screenqa_raw_id] = (image_id, image_name)
        else:
            image_id, image_name = cached_image
        payloads.append(
            _manifest_payload(
                row,
                task=task,
                source_index=source_index,
                image_id=image_id,
                image_path=f"images/{image_name}",
                dataset_revision=dataset_revision,
                state_namespace=state_namespace,
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
    provenance: dict[str, Any] = {
        "task": task,
        "dataset_id": dataset_id,
        "dataset_name": BENCHMARK_SPECS[task].dataset_name,
        "dataset_revision": dataset_revision,
        "split": resolved_split,
        "scorer": BENCHMARK_SPECS[task].scorer,
        "selection_fields": list(BENCHMARK_SPECS[task].selection_fields),
        "selection": selection,
        "seed": seed,
        "count": len(payloads),
        "source_indices": list(source_indices),
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "unique_images": len({payload["image_id"] for payload in payloads}),
        "unique_sources": len({payload["source_id"] for payload in payloads}),
        "state_namespace": state_namespace or task,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "python": sys.version.split()[0],
        "packages": {},
    }
    if selection_metadata is not None:
        provenance["selection_metadata"] = dict(selection_metadata)
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
