"""Ledger-first, hash-bound access control for a fresh sequential held-out test."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REQUIRED_FREEZE_FIELDS = frozenset(
    {
        "schema",
        "one_shot",
        "test_authorized",
        "benchmark",
        "manifest_path",
        "expected_manifest_sha256",
        "rollouts_output",
        "features_output",
        "config_path",
        "config_sha256",
        "critics_path",
        "critics_sha256",
        "model",
        "model_revision",
        "generation_seeds",
        "proposer",
        "candidate_count",
        "visual_cost_per_crop",
        "dtype",
        "attention_implementation",
        "max_new_tokens",
        "min_pixels",
        "max_pixels",
        "manifest_limit",
        "shard_count",
        "shard_index",
        "bootstrap_samples",
        "bootstrap_seed",
        "code_revision",
    }
)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_test_freeze(value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) != REQUIRED_FREEZE_FIELDS:
        raise ValueError("sequential test freeze fields differ from exact contract")
    if (
        value["schema"] != "sequential_test_freeze_v1"
        or value["one_shot"] is not True
        or value["test_authorized"] is not True
        or value["proposer"] != "sequential-opposite-ug-v1"
        or int(value["candidate_count"]) != 4
        or float(value["visual_cost_per_crop"]) != 1.0
        or int(value["max_new_tokens"]) <= 0
        or int(value["min_pixels"]) <= 0
        or int(value["max_pixels"]) < int(value["min_pixels"])
        or int(value["shard_count"]) <= 0
        or not 0 <= int(value["shard_index"]) < int(value["shard_count"])
        or int(value["bootstrap_samples"]) < 10_000
    ):
        raise ValueError("invalid sequential test freeze semantics")
    for name in (
        "expected_manifest_sha256",
        "config_sha256",
        "critics_sha256",
    ):
        raw = str(value[name])
        if len(raw) != 64 or any(char not in "0123456789abcdef" for char in raw):
            raise ValueError(f"invalid {name}")
    seeds = value["generation_seeds"]
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("frozen generation seeds must be non-empty and unique")
    if any(not str(value[name]) for name in REQUIRED_FREEZE_FIELDS - {"one_shot", "test_authorized"}):
        raise ValueError("sequential test freeze contains an empty field")
    return dict(value)


def load_test_freeze(path: str | Path) -> tuple[dict[str, Any], str]:
    freeze_path = Path(path).resolve()
    value = validate_test_freeze(json.loads(freeze_path.read_text()))
    return value, sha256_file(freeze_path)


def start_test_transaction(
    freeze_path: str | Path,
    ledger_path: str | Path,
    *,
    benchmark: str,
    manifest_path: str | Path,
    rollouts_output: str | Path,
    features_output: str | Path,
    model: str,
    model_revision: str,
    generation_seeds: list[int],
    code_revision: str,
    dtype: str,
    attention_implementation: str,
    max_new_tokens: int,
    min_pixels: int,
    max_pixels: int,
    manifest_limit: int | None,
    shard_count: int,
    shard_index: int,
) -> dict[str, Any]:
    """Create the irreversible access ledger before reading test manifest bytes."""

    freeze, freeze_sha256 = load_test_freeze(freeze_path)
    expected = {
        "benchmark": benchmark,
        "manifest_path": str(Path(manifest_path).resolve()),
        "rollouts_output": str(Path(rollouts_output).resolve()),
        "features_output": str(Path(features_output).resolve()),
        "model": model,
        "model_revision": model_revision,
        "generation_seeds": generation_seeds,
        "code_revision": code_revision,
        "dtype": dtype,
        "attention_implementation": attention_implementation,
        "max_new_tokens": max_new_tokens,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "manifest_limit": manifest_limit,
        "shard_count": shard_count,
        "shard_index": shard_index,
    }
    for key, actual in expected.items():
        if freeze[key] != actual:
            raise ValueError(f"test execution differs from freeze for {key}")
    ledger = Path(ledger_path).resolve()
    ledger.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "sequential_test_access_v1",
        "status": "started_irreversible",
        "freeze_path": str(Path(freeze_path).resolve()),
        "freeze_sha256": freeze_sha256,
        "benchmark": benchmark,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with ledger.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return payload


def validate_test_access(
    freeze_path: str | Path, ledger_path: str | Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    freeze, freeze_sha256 = load_test_freeze(freeze_path)
    ledger = json.loads(Path(ledger_path).read_text())
    if (
        ledger.get("schema") != "sequential_test_access_v1"
        or ledger.get("status") != "started_irreversible"
        or ledger.get("freeze_sha256") != freeze_sha256
        or ledger.get("benchmark") != freeze["benchmark"]
    ):
        raise ValueError("invalid or drifted sequential test access ledger")
    return freeze, ledger
