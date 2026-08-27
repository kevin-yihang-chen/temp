from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

from .schema import ActionRecord


def generated_token_count(record: ActionRecord) -> int:
    backend_name = "baseline_backend" if record.action_type == "ANSWER" else "action_backend"
    backend = record.metadata.get(backend_name)
    if not isinstance(backend, dict):
        raise ValueError(f"record {record.action_id!r} is missing {backend_name} metadata")
    value = backend.get("generated_tokens")
    if not isinstance(value, (int, float)):
        raise ValueError(f"record {record.action_id!r} has no generated token count")
    count = int(value)
    if count <= 0 or count != value:
        raise ValueError(f"invalid generated token count: {value!r}")
    return count


def capped_state_ids(
    records: Sequence[ActionRecord],
    *,
    token_cap: int,
) -> set[str]:
    if token_cap <= 0:
        raise ValueError("token_cap must be positive")
    return {
        record.state_id
        for record in records
        if generated_token_count(record) >= token_cap
    }


def export_capped_manifest(
    *,
    records: Sequence[ActionRecord],
    source_manifest: str | Path,
    source_rollouts: str | Path,
    output_manifest: str | Path,
    token_cap: int,
) -> dict[str, object]:
    selected_ids = capped_state_ids(records, token_cap=token_cap)
    if not selected_ids:
        raise ValueError("no states reached the decoding token cap")
    source_path = Path(source_manifest).resolve()
    output_path = Path(output_manifest)
    selected_lines: list[str] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(source_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            state_id = str(value["state_id"])
            raw_image_path = Path(str(value["image_path"]))
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid manifest row {source_path}:{line_number}") from exc
        if state_id in selected_ids:
            source_image_path = (
                raw_image_path
                if raw_image_path.is_absolute()
                else source_path.parent / raw_image_path
            ).resolve()
            if not source_image_path.is_file():
                raise ValueError(f"manifest image does not exist: {source_image_path}")
            value["image_path"] = os.path.relpath(source_image_path, output_path.parent.resolve())
            selected_lines.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
            seen_ids.add(state_id)
    missing = selected_ids - seen_ids
    if missing:
        raise ValueError(f"capped states are absent from source manifest: {sorted(missing)[:5]}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(selected_lines) + "\n", encoding="utf-8")
    manifest_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    rollouts_path = Path(source_rollouts).resolve()
    provenance: dict[str, object] = {
        "scientific_status": "targeted decoding sensitivity subset",
        "selection_rule": "at least one sibling output reached the generated-token cap",
        "token_cap": token_cap,
        "states": len(selected_ids),
        "source_manifest": str(source_path),
        "source_manifest_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_rollouts": str(rollouts_path),
        "source_rollouts_sha256": hashlib.sha256(rollouts_path.read_bytes()).hexdigest(),
        "output_manifest": str(output_path.resolve()),
        "output_manifest_sha256": manifest_sha256,
        "state_ids": sorted(selected_ids),
    }
    provenance_path = output_path.with_suffix(".provenance.json")
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return provenance
