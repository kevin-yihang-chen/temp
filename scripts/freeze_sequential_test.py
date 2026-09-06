#!/usr/bin/env python3
"""Create a one-shot sequential test freeze without opening test data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from beyond_entropy.sequential_test_transaction import validate_test_freeze, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("chartqa", "docvqa", "hrbench"), required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--rollouts-output", required=True)
    parser.add_argument("--features-output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--critics", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--generation-seed", type=int, action="append", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention-implementation", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=768 * 28 * 28)
    parser.add_argument("--manifest-limit", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260906)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = Path(args.output).resolve()
    value = {
        "schema": "sequential_test_freeze_v1",
        "one_shot": True,
        "test_authorized": True,
        "benchmark": args.benchmark,
        # Paths are resolved without reading manifest contents.
        "manifest_path": str(Path(args.manifest_path).resolve()),
        "expected_manifest_sha256": args.expected_manifest_sha256,
        "rollouts_output": str(Path(args.rollouts_output).resolve()),
        "features_output": str(Path(args.features_output).resolve()),
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": sha256_file(args.config),
        "critics_path": str(Path(args.critics).resolve()),
        "critics_sha256": sha256_file(args.critics),
        "model": args.model,
        "model_revision": args.model_revision,
        "generation_seeds": args.generation_seed,
        "proposer": "sequential-opposite-ug-v1",
        "candidate_count": 4,
        "visual_cost_per_crop": 1.0,
        "dtype": args.dtype,
        "attention_implementation": args.attention_implementation,
        "max_new_tokens": args.max_new_tokens,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "manifest_limit": args.manifest_limit,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "code_revision": args.code_revision,
    }
    validate_test_freeze(value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
