from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

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
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def group_by_state(records: Iterable[ActionRecord]) -> dict[str, list[ActionRecord]]:
    grouped: dict[str, list[ActionRecord]] = defaultdict(list)
    for record in records:
        grouped[record.state_id].append(record)
    return dict(grouped)


def validate_sibling_groups(records: Iterable[ActionRecord]) -> None:
    grouped = group_by_state(records)
    if not grouped:
        raise ValueError("dataset is empty")
    for state_id, siblings in grouped.items():
        answers = [record for record in siblings if record.action_type == "ANSWER"]
        zooms = [record for record in siblings if record.action_type == "ZOOM"]
        if len(answers) != 1:
            raise ValueError(f"state {state_id!r} must have exactly one ANSWER sibling")
        if not zooms:
            raise ValueError(f"state {state_id!r} must have at least one ZOOM sibling")
        baseline = answers[0]
        action_ids: set[str] = set()
        for record in siblings:
            if record.action_id in action_ids:
                raise ValueError(
                    f"state {state_id!r} contains duplicate action_id {record.action_id!r}"
                )
            action_ids.add(record.action_id)
            if record.question != baseline.question or record.original_image != baseline.original_image:
                raise ValueError(f"state {state_id!r} has inconsistent state content")
            if abs(record.entropy_before - baseline.entropy_before) > 1e-9:
                raise ValueError(f"state {state_id!r} has inconsistent entropy_before")
            if abs(record.correct_before - baseline.correct_before) > 1e-9:
                raise ValueError(f"state {state_id!r} has inconsistent correct_before")
        if abs(baseline.delta_success) > 1e-9:
            raise ValueError(f"state {state_id!r} ANSWER sibling must preserve correctness")


def split_by_state(
    records: Sequence[ActionRecord],
    *,
    train_fraction: float = 0.7,
    seed: int = 0,
) -> tuple[list[ActionRecord], list[ActionRecord]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    grouped = group_by_state(records)
    state_ids = sorted(grouped)
    if len(state_ids) < 2:
        raise ValueError("at least two states are required for a split")
    random.Random(seed).shuffle(state_ids)
    cut = min(max(1, round(len(state_ids) * train_fraction)), len(state_ids) - 1)
    train_ids = set(state_ids[:cut])
    train = [record for record in records if record.state_id in train_ids]
    test = [record for record in records if record.state_id not in train_ids]
    return train, test
