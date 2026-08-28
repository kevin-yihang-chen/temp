from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .dataset import group_by_decision, read_jsonl


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_sibling_rollout_bank(
    manifest_path: str | Path,
    rollout_path: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_states: int,
    expected_candidate_count: int = 4,
    expected_model_revision: str | None = None,
    expected_scientific_status: str | None = None,
) -> dict[str, Any]:
    """Validate exact manifest coverage and complete sibling decisions."""

    manifest = Path(manifest_path).resolve()
    rollouts = Path(rollout_path).resolve()
    provenance_path = rollouts.with_suffix(".provenance.json")
    diagnostic_path = rollouts.with_suffix(".diagnostic.json")
    for path in (manifest, rollouts, provenance_path, diagnostic_path):
        if not path.is_file():
            raise FileNotFoundError(f"required rollout-bank file does not exist: {path}")
    manifest_sha256 = _sha256(manifest)
    if manifest_sha256 != expected_manifest_sha256:
        raise ValueError("rollout audit manifest SHA-256 mismatch")

    manifest_rows: dict[str, Mapping[str, Any]] = {}
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"manifest row {line_number} must be a mapping")
            state_id = str(payload.get("state_id", "")).strip()
            if not state_id or state_id in manifest_rows:
                raise ValueError("manifest state IDs must be non-empty and unique")
            manifest_rows[state_id] = payload
    if len(manifest_rows) != expected_states:
        raise ValueError("rollout audit manifest state count mismatch")

    records = read_jsonl(rollouts)
    grouped = group_by_decision(records)
    expected_records_per_state = expected_candidate_count + 1
    if len(records) != expected_states * expected_records_per_state:
        raise ValueError("rollout audit record count mismatch")
    state_keys: dict[str, tuple[str, str]] = {}
    action_type_counts: Counter[str] = Counter()
    sources: set[str] = set()
    images: set[str] = set()
    for key, siblings in grouped.items():
        state_id = key[0]
        if state_id in state_keys:
            raise ValueError("rollout audit found multiple replicates for one state")
        state_keys[state_id] = key
        if len(siblings) != expected_records_per_state:
            raise ValueError(f"decision {key!r} has an unexpected sibling count")
        manifest_row = manifest_rows.get(state_id)
        if manifest_row is None:
            raise ValueError(f"rollout decision {state_id!r} is absent from manifest")
        expected_source = str(manifest_row.get("source_id", ""))
        expected_image = str(manifest_row.get("image_id", ""))
        expected_question = str(manifest_row.get("question", ""))
        for record in siblings:
            if (
                record.source_id != expected_source
                or record.image_id != expected_image
                or record.question != expected_question
            ):
                raise ValueError(f"rollout identity mismatch for state {state_id!r}")
            action_type_counts[record.action_type] += 1
            sources.add(record.source_id)
            images.add(record.image_id)
    if set(state_keys) != set(manifest_rows):
        raise ValueError("rollout decisions do not exactly cover manifest states")
    if action_type_counts != Counter(
        {
            "ANSWER": expected_states,
            "ZOOM": expected_states * expected_candidate_count,
        }
    ):
        raise ValueError("rollout action-type counts are inconsistent")

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(provenance, Mapping):
        raise ValueError("rollout provenance must be a mapping")
    rollout_sha256 = _sha256(rollouts)
    required_provenance = {
        "manifest_sha256": manifest_sha256,
        "output_sha256": rollout_sha256,
        "examples": expected_states,
        "completed_examples": expected_states,
        "candidate_count": expected_candidate_count,
    }
    for name, expected in required_provenance.items():
        if provenance.get(name) != expected:
            raise ValueError(f"rollout provenance mismatch for {name}")
    if (
        expected_model_revision is not None
        and provenance.get("model_revision") != expected_model_revision
    ):
        raise ValueError("rollout provenance model revision mismatch")
    if (
        expected_scientific_status is not None
        and provenance.get("scientific_status") != expected_scientific_status
    ):
        raise ValueError("rollout provenance scientific status mismatch")
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if not isinstance(diagnostic, Mapping):
        raise ValueError("rollout diagnostic must be a mapping")

    return {
        "passed": True,
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha256,
        "rollouts": str(rollouts),
        "rollouts_sha256": rollout_sha256,
        "rollout_provenance": str(provenance_path),
        "rollout_provenance_sha256": _sha256(provenance_path),
        "rollout_diagnostic": str(diagnostic_path),
        "rollout_diagnostic_sha256": _sha256(diagnostic_path),
        "states": expected_states,
        "records": len(records),
        "unique_sources": len(sources),
        "unique_images": len(images),
        "answer_records": action_type_counts["ANSWER"],
        "zoom_records": action_type_counts["ZOOM"],
        "candidate_count": expected_candidate_count,
        "model_revision": provenance.get("model_revision"),
        "code_revision": provenance.get("code_revision"),
        "scientific_status": provenance.get("scientific_status"),
        "checkpoint_interval": provenance.get("checkpoint_interval"),
    }
