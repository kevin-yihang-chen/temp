from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_sha256(image_paths: Sequence[Path], *, root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(image_paths, key=lambda item: str(item.relative_to(root))):
        relative = str(path.relative_to(root))
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(_sha256(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_target(payload: Mapping[str, Any], *, scorer: str) -> None:
    target = payload.get("target")
    if scorer in {"docvqa", "textvqa"}:
        if not isinstance(target, Mapping):
            raise ValueError(f"{scorer} target must be a mapping")
        answers = target.get("answers")
        if isinstance(answers, (str, bytes)) or not isinstance(answers, Sequence):
            raise ValueError(f"{scorer} target answers must be a sequence")
        if not answers:
            raise ValueError(f"{scorer} target answers must be non-empty")
        if scorer == "textvqa" and len(answers) != 10:
            raise ValueError("TextVQA target must contain ten answers")
    elif scorer == "hrbench":
        if not isinstance(target, Mapping):
            raise ValueError("HRBench target must be a mapping")
        if str(target.get("answer", "")).strip().upper() not in {"A", "B", "C", "D"}:
            raise ValueError("HRBench target answer must be A, B, C, or D")


def audit_manifest(directory: str | Path) -> dict[str, Any]:
    root = Path(directory).resolve()
    manifest_path = root / "manifest.jsonl"
    provenance_path = root / "manifest.provenance.json"
    if not manifest_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError(f"manifest pair is incomplete under {root}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(provenance, Mapping):
        raise ValueError("manifest provenance must be a mapping")
    manifest_sha256 = _sha256(manifest_path)
    if manifest_sha256 != provenance.get("manifest_sha256"):
        raise ValueError(f"manifest SHA-256 mismatch under {root}")

    states: set[str] = set()
    sources: set[str] = set()
    images: set[str] = set()
    image_paths: dict[str, Path] = {}
    line_count = 0
    scorer = str(provenance.get("scorer", ""))
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {manifest_path}:{line_number}") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"manifest row {line_number} must be a mapping")
            state_id = str(payload.get("state_id", "")).strip()
            source_id = str(payload.get("source_id", "")).strip()
            image_id = str(payload.get("image_id", "")).strip()
            relative_image = str(payload.get("image_path", "")).strip()
            if not state_id or not source_id or not image_id or not relative_image:
                raise ValueError(f"manifest row {line_number} has empty identifiers")
            if state_id in states:
                raise ValueError(f"duplicate state_id {state_id!r}")
            resolved_image = (root / relative_image).resolve()
            if root not in resolved_image.parents or not resolved_image.is_file():
                raise ValueError(f"invalid image path at manifest row {line_number}")
            previous_path = image_paths.setdefault(image_id, resolved_image)
            if previous_path != resolved_image:
                raise ValueError(f"image ID {image_id!r} maps to multiple files")
            _validate_target(payload, scorer=scorer)
            states.add(state_id)
            sources.add(source_id)
            images.add(image_id)
            line_count += 1
    if line_count == 0:
        raise ValueError("manifest is empty")
    expected_count = int(provenance.get("count", -1))
    if line_count != expected_count:
        raise ValueError(f"manifest count mismatch: {line_count} != {expected_count}")
    if len(sources) != int(provenance.get("unique_sources", -1)):
        raise ValueError("unique source count does not match provenance")
    if len(images) != int(provenance.get("unique_images", -1)):
        raise ValueError("unique image count does not match provenance")
    actual_image_files = sorted((root / "images").glob("*.png"))
    if set(actual_image_files) != set(image_paths.values()):
        raise ValueError("image bundle contains missing or unreferenced PNG files")
    return {
        "root": str(root),
        "task": str(provenance.get("task", "")),
        "dataset_id": str(provenance.get("dataset_id", "")),
        "dataset_name": provenance.get("dataset_name"),
        "dataset_revision": str(provenance.get("dataset_revision", "")),
        "split": str(provenance.get("split", "")),
        "scorer": scorer,
        "selection": str(provenance.get("selection", "")),
        "selection_metadata": dict(provenance.get("selection_metadata", {})),
        "count": line_count,
        "unique_states": len(states),
        "unique_sources": len(sources),
        "unique_images": len(images),
        "manifest_sha256": manifest_sha256,
        "image_bundle_sha256": _bundle_sha256(actual_image_files, root=root),
        "_states": states,
        "_sources": sources,
        "_images": images,
    }


def audit_manifest_pair(
    development_dir: str | Path,
    formal_dir: str | Path,
    *,
    task: str,
    expected_revision: str,
) -> dict[str, Any]:
    development = audit_manifest(development_dir)
    formal = audit_manifest(formal_dir)
    for name, report in (("development", development), ("formal", formal)):
        if report["task"] != task:
            raise ValueError(f"{name} task mismatch: {report['task']} != {task}")
        if report["dataset_revision"] != expected_revision:
            raise ValueError(f"{name} dataset revision mismatch")
    state_overlap = development["_states"] & formal["_states"]
    source_overlap = development["_sources"] & formal["_sources"]
    image_overlap = development["_images"] & formal["_images"]
    if state_overlap or source_overlap or image_overlap:
        raise ValueError(
            "development/formal leakage: "
            f"states={len(state_overlap)}, sources={len(source_overlap)}, "
            f"images={len(image_overlap)}"
        )
    for report in (development, formal):
        report.pop("_states")
        report.pop("_sources")
        report.pop("_images")
    return {
        "passed": True,
        "task": task,
        "expected_revision": expected_revision,
        "development": development,
        "formal": formal,
        "overlap": {"states": 0, "sources": 0, "images": 0},
    }
