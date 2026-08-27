from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Literal, Sequence

from .schema import ActionRecord


def read_jsonl(path: str | Path, *, validate: bool = True) -> list[ActionRecord]:
    records: list[ActionRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(ActionRecord.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid record at {path}:{line_number}: {exc}") from exc
    if validate:
        validate_sibling_groups(records)
    return records


def write_jsonl(records: Iterable[ActionRecord], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


DecisionKey = tuple[str, str]
SplitGroup = Literal["source_id", "image_id", "state_id"]


def group_by_state(records: Iterable[ActionRecord]) -> dict[str, list[ActionRecord]]:
    grouped: dict[str, list[ActionRecord]] = defaultdict(list)
    for record in records:
        grouped[record.state_id].append(record)
    return dict(grouped)


def group_by_decision(
    records: Iterable[ActionRecord],
) -> dict[DecisionKey, list[ActionRecord]]:
    """Group sibling actions from one state and one stochastic replicate."""

    grouped: dict[DecisionKey, list[ActionRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.state_id, record.replicate_id)].append(record)
    return dict(grouped)


def validate_sibling_groups(records: Iterable[ActionRecord]) -> None:
    materialized = list(records)
    grouped = group_by_decision(materialized)
    if not grouped:
        raise ValueError("dataset is empty")
    for (state_id, replicate_id), siblings in grouped.items():
        answers = [record for record in siblings if record.action_type == "ANSWER"]
        zooms = [record for record in siblings if record.action_type == "ZOOM"]
        if len(answers) != 1:
            raise ValueError(
                f"decision {(state_id, replicate_id)!r} must have exactly one ANSWER sibling"
            )
        if not zooms:
            raise ValueError(
                f"decision {(state_id, replicate_id)!r} must have at least one ZOOM sibling"
            )
        baseline = answers[0]
        action_ids: set[str] = set()
        for record in siblings:
            if record.action_id in action_ids:
                raise ValueError(
                    f"state {state_id!r} contains duplicate action_id {record.action_id!r}"
                )
            action_ids.add(record.action_id)
            if record.question != baseline.question or record.original_image != baseline.original_image:
                raise ValueError(f"decision {(state_id, replicate_id)!r} has inconsistent content")
            if record.image_id != baseline.image_id or record.source_id != baseline.source_id:
                raise ValueError(
                    f"decision {(state_id, replicate_id)!r} has inconsistent group identifiers"
                )
            if record.generation_seed != baseline.generation_seed:
                raise ValueError(
                    f"decision {(state_id, replicate_id)!r} has inconsistent generation_seed"
                )
            if abs(record.entropy_before - baseline.entropy_before) > 1e-9:
                raise ValueError(
                    f"decision {(state_id, replicate_id)!r} has inconsistent entropy_before"
                )
            if abs(record.correct_before - baseline.correct_before) > 1e-9:
                raise ValueError(
                    f"decision {(state_id, replicate_id)!r} has inconsistent correct_before"
                )
        if abs(baseline.delta_success) > 1e-9:
            raise ValueError(
                f"decision {(state_id, replicate_id)!r} ANSWER sibling must preserve correctness"
            )

    for state_id, state_records in group_by_state(materialized).items():
        exemplar = state_records[0]
        for record in state_records[1:]:
            if (
                record.image_id != exemplar.image_id
                or record.source_id != exemplar.source_id
                or record.question != exemplar.question
                or record.original_image != exemplar.original_image
            ):
                raise ValueError(f"state {state_id!r} changes across replicates")


def split_by_group(
    records: Sequence[ActionRecord],
    *,
    group: SplitGroup = "image_id",
    train_fraction: float = 0.7,
    seed: int = 0,
) -> tuple[list[ActionRecord], list[ActionRecord]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if group not in ("source_id", "image_id", "state_id"):
        raise ValueError(f"unsupported split group: {group}")
    group_ids = sorted({getattr(record, group) for record in records})
    if len(group_ids) < 2:
        raise ValueError(f"at least two {group} groups are required for a split")
    random.Random(seed).shuffle(group_ids)
    cut = min(max(1, round(len(group_ids) * train_fraction)), len(group_ids) - 1)
    train_ids = set(group_ids[:cut])
    train = [record for record in records if getattr(record, group) in train_ids]
    test = [record for record in records if getattr(record, group) not in train_ids]
    return train, test


def split_by_state(
    records: Sequence[ActionRecord],
    *,
    train_fraction: float = 0.7,
    seed: int = 0,
) -> tuple[list[ActionRecord], list[ActionRecord]]:
    """Compatibility wrapper; real experiments should split by image/source."""

    return split_by_group(
        records,
        group="state_id",
        train_fraction=train_fraction,
        seed=seed,
    )
