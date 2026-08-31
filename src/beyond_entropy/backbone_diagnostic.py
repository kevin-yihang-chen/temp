from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


SELECTION_SCHEMA = "source_disjoint_backbone_diagnostic_selection_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rank(namespace: str, seed: int, *parts: str) -> str:
    value = "\0".join((namespace, str(seed), *parts))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"staging file exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def select_source_disjoint_manifest(
    *,
    manifest: str | Path,
    output: str | Path,
    report: str | Path,
    expected_manifest_sha256: str,
    source_count: int,
    namespace: str,
    seed: int,
    code_revision: str,
) -> dict[str, Any]:
    """Select one hash-ranked state from each of N hash-ranked source groups."""

    manifest_path = Path(manifest).resolve()
    output_path = Path(output).resolve()
    report_path = Path(report).resolve()
    if source_count <= 0:
        raise ValueError("source count must be positive")
    if not namespace.strip() or not code_revision.strip():
        raise ValueError("namespace and code revision must be non-empty")
    if output_path.parent != manifest_path.parent:
        raise ValueError(
            "selected manifest must share the source manifest directory so relative "
            "image paths retain their meaning"
        )
    if output_path == report_path or output_path == manifest_path or report_path == manifest_path:
        raise ValueError("input, selected manifest, and report paths must be distinct")
    if output_path.exists() or report_path.exists():
        raise FileExistsError("selected manifest output already exists")
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != expected_manifest_sha256:
        raise ValueError("source manifest SHA-256 mismatch")

    rows: list[dict[str, Any]] = []
    source_rows: dict[str, list[tuple[int, str]]] = defaultdict(list)
    state_ids: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid manifest JSON at line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"manifest row {line_number} is not an object")
            source_id = str(row.get("source_id", "")).strip()
            state_id = str(row.get("state_id", "")).strip()
            if not source_id or not state_id:
                raise ValueError(f"manifest row {line_number} has an empty source/state ID")
            if state_id in state_ids:
                raise ValueError(f"duplicate state ID in source manifest: {state_id}")
            state_ids.add(state_id)
            index = len(rows)
            rows.append(row)
            source_rows[source_id].append((index, state_id))
    if len(source_rows) < source_count:
        raise ValueError(
            f"requested {source_count} sources but only {len(source_rows)} are available"
        )

    ranked_sources = sorted(
        source_rows,
        key=lambda source_id: (_rank(namespace, seed, "source", source_id), source_id),
    )
    selected_sources = ranked_sources[:source_count]
    selected_indices: set[int] = set()
    selection_rows: list[dict[str, object]] = []
    for source_id in selected_sources:
        candidates = source_rows[source_id]
        selected_index, selected_state_id = min(
            candidates,
            key=lambda item: (
                _rank(namespace, seed, "state", source_id, item[1]),
                item[1],
            ),
        )
        selected_indices.add(selected_index)
        selection_rows.append(
            {
                "source_id": source_id,
                "source_rank_sha256": _rank(namespace, seed, "source", source_id),
                "state_id": selected_state_id,
                "state_rank_sha256": _rank(
                    namespace, seed, "state", source_id, selected_state_id
                ),
                "source_candidate_states": len(candidates),
            }
        )
    selected = [rows[index] for index in sorted(selected_indices)]
    if len(selected) != source_count:
        raise RuntimeError("source-disjoint selection count mismatch")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected
    )
    _atomic_text(output_path, output_text)
    output_sha256 = sha256_file(output_path)
    selected_source_digest = hashlib.sha256(
        "\n".join(sorted(selected_sources)).encode("utf-8")
    ).hexdigest()
    selected_state_digest = hashlib.sha256(
        "\n".join(sorted(row["state_id"] for row in selection_rows)).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema": SELECTION_SCHEMA,
        "scientific_status": (
            "opened ranker-development backbone diagnostic selection; no outcome- or "
            "label-dependent ranking"
        ),
        "input": {
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "rows": len(rows),
            "sources": len(source_rows),
        },
        "selection": {
            "namespace": namespace,
            "seed": seed,
            "source_count": source_count,
            "states_per_source": 1,
            "selection_fields": ["source_id", "state_id"],
            "labels_used_for_ranking": False,
            "outcomes_used_for_ranking": False,
            "selected_source_ids_sha256": selected_source_digest,
            "selected_state_ids_sha256": selected_state_digest,
            "rows": selection_rows,
        },
        "output": {
            "manifest": str(output_path),
            "manifest_sha256": output_sha256,
            "rows": len(selected),
            "sources": len({str(row["source_id"]) for row in selected}),
        },
        "implementation": {
            "code_revision": code_revision,
            "module_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    try:
        _atomic_json(report_path, payload)
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    return payload
