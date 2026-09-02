from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .dataset import group_by_decision, read_jsonl
from .infographicvqa_literature_attention_extraction import (
    LITERATURE_ATTENTION_METADATA_KEY,
)
from .qwen_semantic import (
    load_semantic_feature_dataset,
    validate_semantic_feature_dataset,
)

_DYNAMIC_METADATA = frozenset(
    {
        "source_features",
        "source_features_sha256",
        "source_rollouts",
        "source_rollouts_sha256",
        "completed_decisions",
        "total_decisions",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _without(mapping: Mapping[str, Any], names: frozenset[str]) -> dict[str, Any]:
    return {name: value for name, value in mapping.items() if name not in names}


def _recursively_equal(left: Any, right: Any) -> bool:
    import torch  # type: ignore[import-not-found]

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_recursively_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        return (
            isinstance(right, Sequence)
            and not isinstance(right, (str, bytes))
            and len(left) == len(right)
            and all(_recursively_equal(a, b) for a, b in zip(left, right))
        )
    return bool(left == right)


def _atomic_torch_save(value: object, destination: Path) -> None:
    import torch  # type: ignore[import-not-found]

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"literature merge staging file exists: {temporary}")
    try:
        torch.save(value, temporary)
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _atomic_json_save(value: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(
            f"literature merge report staging file exists: {temporary}"
        )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def merge_literature_attention_shards(
    *,
    full_rollouts_path: str | Path,
    expected_full_rollouts_sha256: str,
    source_features_path: str | Path,
    expected_source_features_sha256: str,
    shard_rollout_paths: Sequence[str | Path],
    shard_feature_paths: Sequence[str | Path],
    expected_code_revision: str,
    output_path: str | Path,
    report_path: str | Path,
    resume: bool = False,
) -> dict[str, Any]:
    if (
        len(shard_rollout_paths) < 2
        or len(shard_rollout_paths) != len(shard_feature_paths)
        or not expected_code_revision
    ):
        raise ValueError("literature merge requires matching shards and a revision")
    full_rollouts = Path(full_rollouts_path).resolve()
    source_features = Path(source_features_path).resolve()
    full_sha = sha256_file(full_rollouts)
    source_sha = sha256_file(source_features)
    if full_sha != expected_full_rollouts_sha256:
        raise ValueError("literature merge full rollout SHA-256 changed")
    if source_sha != expected_source_features_sha256:
        raise ValueError("literature merge source feature SHA-256 changed")
    full_records = read_jsonl(full_rollouts)
    full_grouped = group_by_decision(full_records)
    full_keys = set(full_grouped)
    source_payload = load_semantic_feature_dataset(source_features)
    validate_semantic_feature_dataset(
        source_payload, full_records, require_outcomes=False
    )
    if not full_keys or bool(source_payload["metadata"].get("outcomes_included", True)):
        raise ValueError("literature merge source population is empty or privileged")

    seen_keys: set[tuple[str, str]] = set()
    seen_sources: set[str] = set()
    decisions: dict[tuple[str, str], dict[str, Any]] = {}
    normalized_metadata: dict[str, Any] | None = None
    first_metadata: dict[str, Any] | None = None
    shard_reports: list[dict[str, Any]] = []
    format_version: Any = None
    for index, (raw_rollouts, raw_features) in enumerate(
        zip(shard_rollout_paths, shard_feature_paths, strict=True)
    ):
        rollouts = Path(raw_rollouts).resolve()
        features = Path(raw_features).resolve()
        records = read_jsonl(rollouts)
        grouped = group_by_decision(records)
        keys = set(grouped)
        sources = {record.source_id for record in records}
        if seen_keys.intersection(keys) or seen_sources.intersection(sources):
            raise ValueError(f"literature merge shard {index} is not source-disjoint")
        payload = load_semantic_feature_dataset(features)
        validate_semantic_feature_dataset(payload, records, require_outcomes=False)
        if format_version is None:
            format_version = payload["format_version"]
        elif payload["format_version"] != format_version:
            raise ValueError("literature merge feature format changed across shards")
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"literature merge shard {index} lacks metadata")
        augmentation = metadata.get(LITERATURE_ATTENTION_METADATA_KEY)
        if not isinstance(augmentation, Mapping):
            raise ValueError(f"literature merge shard {index} lacks stage metadata")
        rollouts_sha = sha256_file(rollouts)
        features_sha = sha256_file(features)
        if (
            augmentation.get("source_rollouts") != str(rollouts)
            or augmentation.get("source_rollouts_sha256") != rollouts_sha
            or augmentation.get("source_features_sha256") is None
            or augmentation.get("code_revision") != expected_code_revision
            or augmentation.get("outcomes_included") is not False
            or augmentation.get("validation_or_test_inputs_used") is not False
            or augmentation.get("completed_decisions") != len(keys)
            or augmentation.get("total_decisions") != len(keys)
        ):
            raise ValueError(f"literature merge shard {index} binding changed")
        normalized = _without(augmentation, _DYNAMIC_METADATA)
        if normalized_metadata is None:
            normalized_metadata = normalized
            first_metadata = dict(augmentation)
        elif normalized != normalized_metadata:
            raise ValueError("literature merge stage metadata changed across shards")
        shard_decisions = payload.get("decisions")
        if not isinstance(shard_decisions, list) or len(shard_decisions) != len(keys):
            raise ValueError(f"literature merge shard {index} coverage changed")
        for decision in shard_decisions:
            key = (str(decision["state_id"]), str(decision["replicate_id"]))
            if key not in keys or key in decisions:
                raise ValueError(f"literature merge shard {index} identity changed")
            decisions[key] = dict(decision)
        seen_keys.update(keys)
        seen_sources.update(sources)
        shard_reports.append(
            {
                "index": index,
                "rollouts": str(rollouts),
                "rollouts_sha256": rollouts_sha,
                "features": str(features),
                "features_sha256": features_sha,
                "decisions": len(keys),
                "sources": len(sources),
            }
        )
    if seen_keys != full_keys or set(decisions) != full_keys:
        raise ValueError("literature merge full decision coverage changed")
    if first_metadata is None:
        raise RuntimeError("literature merge found no stage metadata")
    merged_stage = dict(first_metadata)
    merged_stage.update(
        {
            "source_features": str(source_features),
            "source_features_sha256": source_sha,
            "source_rollouts": str(full_rollouts),
            "source_rollouts_sha256": full_sha,
            "completed_decisions": len(full_keys),
            "total_decisions": len(full_keys),
        }
    )
    merged_metadata = dict(source_payload["metadata"])
    merged_metadata[LITERATURE_ATTENTION_METADATA_KEY] = merged_stage
    merged = {
        "format_version": format_version,
        "metadata": merged_metadata,
        "decisions": [decisions[key] for key in sorted(full_keys)],
    }
    validate_semantic_feature_dataset(merged, full_records, require_outcomes=False)

    output = Path(output_path).resolve()
    report = Path(report_path).resolve()
    if output.exists():
        if not resume:
            raise FileExistsError(f"literature merged output exists: {output}")
        existing = load_semantic_feature_dataset(output)
        validate_semantic_feature_dataset(
            existing, full_records, require_outcomes=False
        )
        if not _recursively_equal(existing, merged):
            raise ValueError("literature merged output differs from recomputation")
    else:
        _atomic_torch_save(merged, output)
    output_sha = sha256_file(output)
    report_payload = {
        "schema": "infographicvqa_literature_attention_where_merge_v1",
        "passed": True,
        "code_revision": expected_code_revision,
        "full_rollouts": str(full_rollouts),
        "full_rollouts_sha256": full_sha,
        "source_features": str(source_features),
        "source_features_sha256": source_sha,
        "source_disjoint": True,
        "output": str(output),
        "output_sha256": output_sha,
        "decisions": len(full_keys),
        "sources": len(seen_sources),
        "shards": shard_reports,
        "outcomes_included": False,
        "validation_or_test_inputs_used": False,
    }
    if report.exists():
        if not resume:
            raise FileExistsError(f"literature merge report exists: {report}")
        existing_report = json.loads(report.read_text(encoding="utf-8"))
        if existing_report != report_payload:
            raise ValueError("literature merge report differs on resume")
    else:
        _atomic_json_save(report_payload, report)
    return report_payload
