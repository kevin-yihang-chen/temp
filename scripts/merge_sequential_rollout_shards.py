#!/usr/bin/env python3
"""Validate and merge a complete deterministic sequential rollout shard set."""
from __future__ import annotations

import argparse
import json

from beyond_entropy.sequential_rollout_shards import merge_sequential_rollout_shards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-code-revision", required=True)
    parser.add_argument("--benchmark", choices=("chartqa", "docvqa", "hrbench"), required=True)
    parser.add_argument("--dataset-role", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--generation-seed", type=int, action="append", default=[])
    args = parser.parse_args()
    result = merge_sequential_rollout_shards(
        manifest_path=args.manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
        run_root=args.run_root,
        shard_count=args.shard_count,
        output_dir=args.output_dir,
        expected_code_revision=args.expected_code_revision,
        benchmark=args.benchmark,
        dataset_role=args.dataset_role,
        generation_seeds=tuple(args.generation_seed or [0]),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
