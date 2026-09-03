from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .benchmarks import load_manifest
from .dataset import group_by_decision, read_jsonl
from .predictability_features import (
    PREDICTABILITY_FEATURE_FORMAT_VERSION,
    load_predictability_feature_dataset,
    validate_predictability_feature_dataset,
)
from .predictability_audit import collapse_fixed_entropy_tool
from .predictability_baselines import validate_fixed_tool_outcomes
from .rollout_shards import shard_directory_name
from .sharding import SHARD_ALGORITHM, stable_shard_index


_DYNAMIC_METADATA_FIELDS = frozenset(
    {
        "rollouts",
        "rollouts_sha256",
        "shard_index",
        "shard_states",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_torch_save(value: object, destination: Path) -> None:
    import torch  # type: ignore[import-not-found]

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"feature merge staging file exists: {temporary}")
    try:
        torch.save(value, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_save(value: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"feature merge report staging file exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _normalized_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: item
        for name, item in value.items()
        if name not in _DYNAMIC_METADATA_FIELDS
    }


def merge_predictability_feature_shards(
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    merged_rollouts_path: Path,
    expected_merged_rollouts_sha256: str,
    run_root: Path,
    shard_count: int,
    shard_key: str,
    shard_namespace: str,
    expected_code_revision: str,
    dataset_role: str,
    output_path: Path,
    report_path: Path,
    feature_name: str = "features.pt",
) -> dict[str, Any]:
    """Validate and atomically merge a deterministic formal feature shard set."""

    if shard_count < 2:
        raise ValueError("feature merge requires at least two shards")
    if shard_key not in {"state_id", "source_id"}:
        raise ValueError("feature shard key must be state_id or source_id")
    if not expected_code_revision:
        raise ValueError("feature merge requires an expected code revision")
    if output_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite merged feature artifacts")
    manifest_file = manifest_path.resolve()
    actual_manifest_sha256 = _sha256_file(manifest_file)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError("feature merge manifest SHA-256 mismatch")
    full_rollouts = merged_rollouts_path.resolve()
    actual_rollouts_sha256 = _sha256_file(full_rollouts)
    if actual_rollouts_sha256 != expected_merged_rollouts_sha256:
        raise ValueError("feature merge full rollout SHA-256 mismatch")

    manifest_examples = load_manifest(manifest_file)
    state_order = [item.state.state_id for item in manifest_examples]
    if len(set(state_order)) != len(state_order):
        raise ValueError("feature merge manifest state IDs must be unique")
    source_by_state = {
        item.state.state_id: item.state.source_id for item in manifest_examples
    }
    expected_states = {
        shard_index: {
            state_id
            for state_id in state_order
            if stable_shard_index(
                source_by_state[state_id] if shard_key == "source_id" else state_id,
                shard_count,
                namespace=shard_namespace,
            )
            == shard_index
        }
        for shard_index in range(shard_count)
    }
    empty = [index for index, states in expected_states.items() if not states]
    if empty:
        raise ValueError(f"feature partition contains empty shards: {empty}")

    full_grouped = group_by_decision(read_jsonl(full_rollouts))
    full_decisions = set(full_grouped)
    rows_by_decision: dict[tuple[str, str], dict[str, Any]] = {}
    first_metadata: dict[str, Any] | None = None
    shard_reports: list[dict[str, Any]] = []
    for shard_index in range(shard_count):
        shard_dir = run_root.resolve() / shard_directory_name(shard_index, shard_count)
        shard_rollouts = shard_dir / "rollouts.jsonl"
        shard_features = shard_dir / feature_name
        if not shard_rollouts.is_file() or not shard_features.is_file():
            raise FileNotFoundError(f"feature shard {shard_index} is incomplete")
        shard_records = read_jsonl(shard_rollouts)
        shard_grouped = group_by_decision(shard_records)
        shard_decisions = set(shard_grouped)
        shard_states = {state_id for state_id, _ in shard_decisions}
        if shard_states != expected_states[shard_index]:
            raise ValueError(f"feature shard {shard_index} state coverage mismatch")
        for decision_id, siblings in shard_grouped.items():
            if sorted(siblings, key=lambda item: item.action_id) != sorted(
                full_grouped.get(decision_id, ()), key=lambda item: item.action_id
            ):
                raise ValueError(
                    f"feature shard {shard_index} rollouts differ from merged input"
                )

        payload, examples = load_predictability_feature_dataset(shard_features)
        validate_fixed_tool_outcomes(
            [item.outcome for item in examples],
            collapse_fixed_entropy_tool(shard_records),
        )
        if payload.get("format_version") != PREDICTABILITY_FEATURE_FORMAT_VERSION:
            raise ValueError(f"feature shard {shard_index} format mismatch")
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"feature shard {shard_index} metadata is missing")
        required = {
            "dataset_role": dataset_role,
            "manifest_sha256": actual_manifest_sha256,
            "rollouts": str(shard_rollouts.resolve()),
            "rollouts_sha256": _sha256_file(shard_rollouts),
            "code_revision": expected_code_revision,
            "shard_algorithm": SHARD_ALGORITHM,
            "shard_count": shard_count,
            "shard_index": shard_index,
            "shard_key": shard_key,
            "shard_namespace": shard_namespace,
            "manifest_examples_before_sharding": len(state_order),
            "shard_states": len(expected_states[shard_index]),
        }
        for name, expected in required.items():
            if metadata.get(name) != expected:
                raise ValueError(
                    f"feature shard {shard_index} metadata mismatch for {name}"
                )
        normalized = _normalized_metadata(metadata)
        if first_metadata is None:
            first_metadata = dict(metadata)
        elif normalized != _normalized_metadata(first_metadata):
            raise ValueError(f"feature shard {shard_index} invariant metadata differs")

        raw_rows = payload.get("rows")
        if not isinstance(raw_rows, list):
            raise ValueError(f"feature shard {shard_index} rows are missing")
        row_decisions = {
            (str(row["state_id"]), str(row["replicate_id"])) for row in raw_rows
        }
        if len(row_decisions) != len(raw_rows) or row_decisions != shard_decisions:
            raise ValueError(f"feature shard {shard_index} decision coverage mismatch")
        for row in raw_rows:
            decision_id = (str(row["state_id"]), str(row["replicate_id"]))
            if decision_id in rows_by_decision:
                raise ValueError(f"duplicate feature decision: {decision_id!r}")
            rows_by_decision[decision_id] = row
        shard_reports.append(
            {
                "shard_index": shard_index,
                "directory": str(shard_dir),
                "states": len(shard_states),
                "decisions": len(shard_decisions),
                "rollouts_sha256": required["rollouts_sha256"],
                "features_sha256": _sha256_file(shard_features),
            }
        )

    if set(rows_by_decision) != full_decisions:
        raise ValueError("merged feature shards do not exactly cover full rollouts")
    assert first_metadata is not None
    merged_metadata = dict(first_metadata)
    merged_metadata["rollouts"] = str(full_rollouts)
    merged_metadata["rollouts_sha256"] = actual_rollouts_sha256
    merged_metadata.pop("shard_index", None)
    merged_metadata.pop("shard_states", None)
    merged_metadata["shard_merge"] = {
        "algorithm": SHARD_ALGORITHM,
        "shard_count": shard_count,
        "shard_key": shard_key,
        "shard_namespace": shard_namespace,
        "complete": True,
    }
    state_position = {state_id: index for index, state_id in enumerate(state_order)}
    merged_rows = [
        rows_by_decision[decision_id]
        for decision_id in sorted(
            rows_by_decision,
            key=lambda item: (state_position[item[0]], item[1]),
        )
    ]
    merged_payload = {
        "format_version": PREDICTABILITY_FEATURE_FORMAT_VERSION,
        "metadata": merged_metadata,
        "rows": merged_rows,
    }
    validate_predictability_feature_dataset(merged_payload)
    _atomic_torch_save(merged_payload, output_path)
    output_sha256 = _sha256_file(output_path)
    report: dict[str, Any] = {
        "schema": "predictability_feature_shard_merge_v1",
        "passed": True,
        "manifest": str(manifest_file),
        "manifest_sha256": actual_manifest_sha256,
        "merged_rollouts": str(full_rollouts),
        "merged_rollouts_sha256": actual_rollouts_sha256,
        "code_revision": expected_code_revision,
        "dataset_role": dataset_role,
        "states": len(state_order),
        "decisions": len(merged_rows),
        "shard_count": shard_count,
        "shard_key": shard_key,
        "shard_namespace": shard_namespace,
        "shards": shard_reports,
        "output": str(output_path.resolve()),
        "output_sha256": output_sha256,
    }
    _atomic_json_save(report, report_path)
    return report
