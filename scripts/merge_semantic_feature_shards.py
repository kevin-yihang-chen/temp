from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from beyond_entropy.dataset import group_by_decision, read_jsonl
from beyond_entropy.qwen_semantic import (
    load_semantic_feature_dataset,
    validate_semantic_feature_dataset,
)


ROLLOUT_BINDINGS = frozenset({"source_rollouts", "source_rollouts_sha256"})
STAGE_BINDINGS = frozenset(
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


def require(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"semantic shard merge mismatch for {name}")


def without(mapping: Mapping[str, Any], names: frozenset[str]) -> dict[str, Any]:
    return {name: value for name, value in mapping.items() if name not in names}


def stage_metadata(
    metadata: Mapping[str, Any], name: str
) -> dict[str, Any]:
    value = metadata.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"semantic shard lacks {name} metadata")
    return dict(value)


def decision_key(decision: Mapping[str, Any]) -> tuple[str, str]:
    return str(decision["state_id"]), str(decision["replicate_id"])


def recursively_equal(left: Any, right: Any) -> bool:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("semantic shard merge requires torch") from exc
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
            and all(recursively_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        return (
            isinstance(right, Sequence)
            and not isinstance(right, (str, bytes))
            and len(left) == len(right)
            and all(recursively_equal(a, b) for a, b in zip(left, right))
        )
    return bool(left == right)


def atomic_torch_save(value: object, destination: Path) -> None:
    import torch  # type: ignore[import-not-found]

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"semantic shard merge staging file exists: {temporary}")
    try:
        torch.save(value, temporary)
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def atomic_json_save(value: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"semantic shard report staging file exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def merge_metadata(
    *,
    stage: str,
    shard_metadata: list[dict[str, Any]],
    full_rollouts: Path,
    full_rollouts_sha256: str,
    total_decisions: int,
    source_features: Path | None,
    source_features_sha256: str | None,
    source_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    first = shard_metadata[0]
    for position, current in enumerate(shard_metadata[1:], start=1):
        dynamic = set(ROLLOUT_BINDINGS)
        if stage == "multimodal":
            dynamic.update({"question_reembedding"})
        elif stage == "attention":
            dynamic.update({"question_reembedding", "question_region_attention"})
        require(
            without(current, frozenset(dynamic)),
            without(first, frozenset(dynamic)),
            f"shard {position} common metadata",
        )

    if stage == "base":
        metadata = dict(first)
        metadata["source_rollouts"] = str(full_rollouts)
        metadata["source_rollouts_sha256"] = full_rollouts_sha256
        return metadata

    if source_payload is None or source_features is None or source_features_sha256 is None:
        raise ValueError(f"semantic {stage} merge requires canonical source features")
    raw_source_metadata = source_payload.get("metadata")
    if not isinstance(raw_source_metadata, Mapping):
        raise ValueError("canonical source features lack metadata")
    metadata = dict(raw_source_metadata)
    if stage == "multimodal":
        metadata["question_feature_mode"] = first.get("question_feature_mode")
        metadata["question_feature"] = first.get("question_feature")
        upgrades = [stage_metadata(item, "question_reembedding") for item in shard_metadata]
        normalized = without(upgrades[0], STAGE_BINDINGS)
        for position, upgrade in enumerate(upgrades[1:], start=1):
            require(
                without(upgrade, STAGE_BINDINGS),
                normalized,
                f"shard {position} question-reembedding metadata",
            )
        upgrade = dict(upgrades[0])
        upgrade.update(
            {
                "source_features": str(source_features),
                "source_features_sha256": source_features_sha256,
                "source_rollouts": str(full_rollouts),
                "source_rollouts_sha256": full_rollouts_sha256,
                "completed_decisions": total_decisions,
                "total_decisions": total_decisions,
            }
        )
        metadata["question_reembedding"] = upgrade
        return metadata

    if stage != "attention":
        raise ValueError(f"unsupported semantic merge stage: {stage}")
    attentions = [stage_metadata(item, "question_region_attention") for item in shard_metadata]
    normalized = without(attentions[0], STAGE_BINDINGS)
    for position, attention in enumerate(attentions[1:], start=1):
        require(
            without(attention, STAGE_BINDINGS),
            normalized,
            f"shard {position} question-region-attention metadata",
        )
    attention = dict(attentions[0])
    attention.update(
        {
            "source_features": str(source_features),
            "source_features_sha256": source_features_sha256,
            "source_rollouts": str(full_rollouts),
            "source_rollouts_sha256": full_rollouts_sha256,
            "completed_decisions": total_decisions,
            "total_decisions": total_decisions,
        }
    )
    metadata["question_region_attention"] = attention
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge source-disjoint semantic feature shards canonically"
    )
    parser.add_argument("--stage", choices=("base", "multimodal", "attention"), required=True)
    parser.add_argument("--full-rollouts", type=Path, required=True)
    parser.add_argument("--expected-full-rollouts-sha256", required=True)
    parser.add_argument("--shard-rollouts", type=Path, action="append", required=True)
    parser.add_argument("--shard-features", type=Path, action="append", required=True)
    parser.add_argument("--shard-plan", type=Path)
    parser.add_argument("--expected-shard-plan-sha256")
    parser.add_argument("--source-features", type=Path)
    parser.add_argument("--expected-source-features-sha256")
    parser.add_argument("--expected-code-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.shard_rollouts) < 2 or len(args.shard_rollouts) != len(
        args.shard_features
    ):
        raise ValueError("semantic merge requires matching rollout/feature shards")
    full_rollouts = args.full_rollouts.resolve()
    full_rollouts_sha256 = sha256_file(full_rollouts)
    require(
        full_rollouts_sha256,
        args.expected_full_rollouts_sha256,
        "full rollout SHA-256",
    )
    full_records = read_jsonl(full_rollouts)
    full_grouped = group_by_decision(full_records)
    full_keys = set(full_grouped)
    if not full_keys:
        raise ValueError("semantic merge full rollouts are empty")

    shard_plan: Path | None = None
    shard_plan_sha256: str | None = None
    plan_payload: dict[str, Any] | None = None
    if (args.shard_plan is None) != (args.expected_shard_plan_sha256 is None):
        raise ValueError("semantic merge shard plan and expected hash must be paired")
    if args.shard_plan is not None:
        shard_plan = args.shard_plan.resolve()
        shard_plan_sha256 = sha256_file(shard_plan)
        require(
            shard_plan_sha256,
            args.expected_shard_plan_sha256,
            "shard plan SHA-256",
        )
        plan_payload = json.loads(shard_plan.read_text(encoding="utf-8"))
        require(
            plan_payload.get("assignment_unit"),
            "global_sorted_decision_batch",
            "shard plan assignment unit",
        )
        require(
            plan_payload.get("assignment_outcome_fields_used"),
            False,
            "shard plan outcome exclusion",
        )
        require(
            plan_payload.get("source_rollouts"),
            str(full_rollouts),
            "shard plan full rollouts",
        )
        require(
            plan_payload.get("source_rollouts_sha256"),
            full_rollouts_sha256,
            "shard plan full rollout hash",
        )
        require(
            plan_payload.get("code_revision"),
            args.expected_code_revision,
            "shard plan code revision",
        )
        require(
            plan_payload.get("shard_count"),
            len(args.shard_rollouts),
            "shard plan count",
        )

    source_features: Path | None = None
    source_features_sha256: str | None = None
    source_payload: dict[str, Any] | None = None
    if args.stage == "base":
        if args.source_features is not None or args.expected_source_features_sha256:
            raise ValueError("base semantic merge cannot accept source features")
    else:
        if args.source_features is None or not args.expected_source_features_sha256:
            raise ValueError(f"{args.stage} semantic merge requires source features")
        source_features = args.source_features.resolve()
        source_features_sha256 = sha256_file(source_features)
        require(
            source_features_sha256,
            args.expected_source_features_sha256,
            "canonical source-feature SHA-256",
        )
        source_payload = load_semantic_feature_dataset(source_features)
        validate_semantic_feature_dataset(source_payload, full_records)

    seen_keys: set[tuple[str, str]] = set()
    seen_sources: set[str] = set()
    source_disjoint = True
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    metadata: list[dict[str, Any]] = []
    shard_documents: list[dict[str, Any]] = []
    format_version: Any = None
    for index, (raw_rollouts, raw_features) in enumerate(
        zip(args.shard_rollouts, args.shard_features)
    ):
        rollout_path = raw_rollouts.resolve()
        feature_path = raw_features.resolve()
        records = read_jsonl(rollout_path)
        grouped = group_by_decision(records)
        keys = set(grouped)
        sources = {record.source_id for record in records}
        overlap = seen_keys.intersection(keys)
        source_overlap = seen_sources.intersection(sources)
        if overlap or (source_overlap and plan_payload is None):
            raise ValueError(
                f"semantic shard {index} overlaps prior shards: "
                f"decisions={len(overlap)} sources={len(source_overlap)}"
            )
        if source_overlap:
            source_disjoint = False
        if plan_payload is not None:
            plan_shards = plan_payload.get("shards")
            if not isinstance(plan_shards, list) or index >= len(plan_shards):
                raise ValueError(f"semantic shard plan lacks shard {index}")
            plan_shard = plan_shards[index]
            if not isinstance(plan_shard, Mapping):
                raise ValueError(f"semantic shard plan shard {index} is invalid")
            require(plan_shard.get("index"), index, f"shard plan index {index}")
            require(
                plan_shard.get("rollouts"),
                str(rollout_path),
                f"shard plan rollout path {index}",
            )
            require(
                plan_shard.get("rollouts_sha256"),
                sha256_file(rollout_path),
                f"shard plan rollout hash {index}",
            )
            require(
                plan_shard.get("decisions"),
                len(keys),
                f"shard plan decision count {index}",
            )
        payload = load_semantic_feature_dataset(feature_path)
        validate_semantic_feature_dataset(payload, records)
        if format_version is None:
            format_version = payload["format_version"]
        require(payload["format_version"], format_version, f"shard {index} format")
        raw_metadata = payload.get("metadata")
        if not isinstance(raw_metadata, Mapping):
            raise ValueError(f"semantic shard {index} lacks metadata")
        require(
            raw_metadata.get("source_rollouts"),
            str(rollout_path),
            f"shard {index} rollout path",
        )
        require(
            raw_metadata.get("source_rollouts_sha256"),
            sha256_file(rollout_path),
            f"shard {index} rollout hash",
        )
        require(
            raw_metadata.get("code_revision"),
            args.expected_code_revision,
            f"shard {index} code revision",
        )
        require(
            raw_metadata.get("outcomes_included"),
            False,
            f"shard {index} outcome exclusion",
        )
        for decision in payload["decisions"]:
            key = decision_key(decision)
            if key in lookup:
                raise ValueError(f"duplicate semantic feature decision {key!r}")
            lookup[key] = decision
        require(set(lookup).intersection(keys), keys, f"shard {index} feature coverage")
        metadata.append(dict(raw_metadata))
        seen_keys.update(keys)
        seen_sources.update(sources)
        shard_documents.append(
            {
                "index": index,
                "rollouts": str(rollout_path),
                "rollouts_sha256": sha256_file(rollout_path),
                "features": str(feature_path),
                "features_sha256": sha256_file(feature_path),
                "decisions": len(keys),
                "sources": len(sources),
            }
        )
    require(seen_keys, full_keys, "full decision coverage")
    require(set(lookup), full_keys, "full feature lookup coverage")

    if plan_payload is not None:
        batch_size = int(plan_payload.get("batch_size", 0))
        if batch_size <= 0:
            raise ValueError("semantic shard plan batch size must be positive")
        ordered_full_keys = sorted(full_keys)
        expected_batches = [
            set(ordered_full_keys[start : start + batch_size])
            for start in range(0, len(ordered_full_keys), batch_size)
        ]
        actual_shard_keys = []
        for raw_rollouts in args.shard_rollouts:
            actual_shard_keys.append(
                set(group_by_decision(read_jsonl(raw_rollouts.resolve())))
            )
        actual_batch_indices: list[list[int]] = [
            [] for _ in range(len(actual_shard_keys))
        ]
        for batch_index, batch_keys in enumerate(expected_batches):
            owners = [
                index
                for index, shard_keys in enumerate(actual_shard_keys)
                if batch_keys.intersection(shard_keys)
            ]
            if len(owners) != 1 or not batch_keys.issubset(actual_shard_keys[owners[0]]):
                raise ValueError(
                    f"semantic shard plan split global inference batch {batch_index}"
                )
            actual_batch_indices[owners[0]].append(batch_index)
        for index, indices in enumerate(actual_batch_indices):
            require(
                plan_payload["shards"][index].get("batch_indices"),
                indices,
                f"shard plan batch indices {index}",
            )

    ordered_keys = sorted(full_keys)
    merged_metadata = merge_metadata(
        stage=args.stage,
        shard_metadata=metadata,
        full_rollouts=full_rollouts,
        full_rollouts_sha256=full_rollouts_sha256,
        total_decisions=len(ordered_keys),
        source_features=source_features,
        source_features_sha256=source_features_sha256,
        source_payload=source_payload,
    )
    merged = {
        "format_version": format_version,
        "metadata": merged_metadata,
        "decisions": [lookup[key] for key in ordered_keys],
    }
    validate_semantic_feature_dataset(merged, full_records)

    output = args.output.resolve()
    report = args.report.resolve()
    if output.exists():
        if not args.resume:
            raise FileExistsError(f"semantic merged output exists: {output}")
        existing = load_semantic_feature_dataset(output)
        validate_semantic_feature_dataset(existing, full_records)
        if not recursively_equal(existing, merged):
            raise ValueError("existing semantic merged output differs from recomputation")
    else:
        atomic_torch_save(merged, output)
    output_sha256 = sha256_file(output)
    report_payload = {
        "passed": True,
        "stage": args.stage,
        "code_revision": args.expected_code_revision,
        "full_rollouts": str(full_rollouts),
        "full_rollouts_sha256": full_rollouts_sha256,
        "source_features": None if source_features is None else str(source_features),
        "source_features_sha256": source_features_sha256,
        "shard_plan": None if shard_plan is None else str(shard_plan),
        "shard_plan_sha256": shard_plan_sha256,
        "source_disjoint": source_disjoint,
        "output": str(output),
        "output_sha256": output_sha256,
        "decisions": len(ordered_keys),
        "sources": len(seen_sources),
        "shards": shard_documents,
        "outcomes_included": False,
    }
    if report.exists():
        if not args.resume:
            raise FileExistsError(f"semantic merge report exists: {report}")
        existing_report = json.loads(report.read_text(encoding="utf-8"))
        require(existing_report, report_payload, "resume report")
    else:
        atomic_json_save(report_payload, report)
    print(json.dumps(report_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
