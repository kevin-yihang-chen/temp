from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DecisionKey = tuple[str, str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"semantic batch shard mismatch for {name}")


def atomic_write_lines(path: Path, lines: list[bytes]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"semantic batch shard staging file exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            for line in lines:
                handle.write(line)
                if not line.endswith(b"\n"):
                    handle.write(b"\n")
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"semantic batch plan staging file exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def key_digest(keys: list[DecisionKey]) -> str:
    payload = "\n".join(f"{state_id}\t{replicate_id}" for state_id, replicate_id in keys)
    return hashlib.sha256(payload.encode()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Shard semantic-feature rollouts without changing global inference "
            "batch boundaries"
        )
    )
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--expected-rollouts-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--expected-code-revision", required=True)
    parser.add_argument("--expected-candidate-count", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.shards < 2 or args.batch_size <= 0:
        raise ValueError("semantic batch sharding requires >=2 shards and positive batch size")
    if args.expected_candidate_count <= 0:
        raise ValueError("expected candidate count must be positive")
    rollouts = args.rollouts.resolve()
    require(sha256_file(rollouts), args.expected_rollouts_sha256, "rollout SHA-256")
    raw_lines = rollouts.read_bytes().splitlines(keepends=True)
    if not raw_lines:
        raise ValueError("semantic batch shard source rollouts are empty")

    decision_records: Counter[DecisionKey] = Counter()
    decision_sources: dict[DecisionKey, str] = {}
    decision_actions: dict[DecisionKey, set[str]] = defaultdict(set)
    decision_types: dict[DecisionKey, Counter[str]] = defaultdict(Counter)
    parsed_keys: list[DecisionKey] = []
    for position, line in enumerate(raw_lines, start=1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"rollout line {position} is not an object")
        key = (str(value.get("state_id", "")), str(value.get("replicate_id", "")))
        source_id = str(value.get("source_id", ""))
        action_id = str(value.get("action_id", ""))
        action_type = str(value.get("action_type", ""))
        if not all((*key, source_id, action_id, action_type)):
            raise ValueError(f"rollout line {position} lacks batch-shard identity fields")
        previous_source = decision_sources.setdefault(key, source_id)
        require(previous_source, source_id, f"decision {key!r} source")
        if action_id in decision_actions[key]:
            raise ValueError(f"decision {key!r} repeats action {action_id!r}")
        decision_actions[key].add(action_id)
        decision_types[key][action_type] += 1
        decision_records[key] += 1
        parsed_keys.append(key)

    expected_records = args.expected_candidate_count + 1
    for key in sorted(decision_records):
        require(decision_records[key], expected_records, f"decision {key!r} records")
        require(decision_types[key]["ANSWER"], 1, f"decision {key!r} ANSWER count")
        require(
            decision_types[key]["ZOOM"],
            args.expected_candidate_count,
            f"decision {key!r} ZOOM count",
        )
        require(set(decision_types[key]), {"ANSWER", "ZOOM"}, f"decision {key!r} types")

    ordered_keys = sorted(decision_records)
    batches = [
        ordered_keys[start : start + args.batch_size]
        for start in range(0, len(ordered_keys), args.batch_size)
    ]
    if len(batches) < args.shards:
        raise ValueError("semantic batch sharding would produce an empty shard")
    batch_assignments: list[list[int]] = [[] for _ in range(args.shards)]
    key_to_shard: dict[DecisionKey, int] = {}
    for batch_index, keys in enumerate(batches):
        shard_index = batch_index % args.shards
        batch_assignments[shard_index].append(batch_index)
        for key in keys:
            key_to_shard[key] = shard_index

    shard_lines: list[list[bytes]] = [[] for _ in range(args.shards)]
    for line, key in zip(raw_lines, parsed_keys):
        shard_lines[key_to_shard[key]].append(line)
    output_dir = args.output_dir.resolve()
    plan_path = output_dir / "plan.json"
    if output_dir.exists() and not args.resume:
        raise FileExistsError(
            f"semantic batch shard output exists: {output_dir}; pass --resume"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_documents: list[dict[str, Any]] = []
    source_shards: dict[str, set[int]] = defaultdict(set)
    seen_keys: set[DecisionKey] = set()
    for index, lines in enumerate(shard_lines):
        path = output_dir / f"rollouts.shard-{index:02d}.jsonl"
        if path.exists():
            if not args.resume:
                raise FileExistsError(f"semantic batch rollout shard exists: {path}")
        else:
            atomic_write_lines(path, lines)
        expected_bytes = b"".join(
            line if line.endswith(b"\n") else line + b"\n" for line in lines
        )
        expected_sha256 = hashlib.sha256(expected_bytes).hexdigest()
        require(sha256_file(path), expected_sha256, f"shard {index} bytes")
        keys = [
            key
            for batch_index in batch_assignments[index]
            for key in batches[batch_index]
        ]
        if seen_keys.intersection(keys):
            raise ValueError(f"semantic batch shard {index} decisions overlap")
        seen_keys.update(keys)
        require(len(lines), len(keys) * expected_records, f"shard {index} records")
        sources = {decision_sources[key] for key in keys}
        for source in sources:
            source_shards[source].add(index)
        shard_documents.append(
            {
                "index": index,
                "rollouts": str(path),
                "rollouts_sha256": expected_sha256,
                "records": len(lines),
                "decisions": len(keys),
                "sources": len(sources),
                "batch_indices": batch_assignments[index],
                "decision_keys_sha256": key_digest(keys),
            }
        )
    require(seen_keys, set(ordered_keys), "full decision coverage")
    overlapping_sources = sorted(
        source for source, shard_indices in source_shards.items() if len(shard_indices) > 1
    )
    plan = {
        "schema_version": 1,
        "scientific_status": (
            "deterministic label-free sharding that preserves every global sorted "
            "inference batch"
        ),
        "assignment_unit": "global_sorted_decision_batch",
        "assignment_allowed_fields": ["state_id", "replicate_id", "source_id"],
        "assignment_outcome_fields_used": False,
        "code_revision": args.expected_code_revision,
        "candidate_count": args.expected_candidate_count,
        "batch_size": args.batch_size,
        "global_batches": len(batches),
        "tail_batch_size": len(batches[-1]),
        "source_rollouts": str(rollouts),
        "source_rollouts_sha256": args.expected_rollouts_sha256,
        "records": len(raw_lines),
        "decisions": len(ordered_keys),
        "sources": len(set(decision_sources.values())),
        "shard_count": args.shards,
        "source_disjoint": not overlapping_sources,
        "overlapping_sources": len(overlapping_sources),
        "overlapping_source_ids_sha256": key_digest(
            [(source, "") for source in overlapping_sources]
        ),
        "shards": shard_documents,
    }
    if plan_path.exists():
        if not args.resume:
            raise FileExistsError(f"semantic batch shard plan exists: {plan_path}")
        stored = json.loads(plan_path.read_text(encoding="utf-8"))
        require(stored, plan, "resume plan")
    else:
        atomic_write_json(plan_path, plan)
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
