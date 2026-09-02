#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from beyond_entropy.infographicvqa_literature_attention_merge import (
    merge_literature_attention_shards,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge frozen InfographicVQA literature-attention feature shards"
    )
    parser.add_argument("--full-rollouts", type=Path, required=True)
    parser.add_argument("--expected-full-rollouts-sha256", required=True)
    parser.add_argument("--source-features", type=Path, required=True)
    parser.add_argument("--expected-source-features-sha256", required=True)
    parser.add_argument("--shard-rollouts", type=Path, action="append", required=True)
    parser.add_argument("--shard-features", type=Path, action="append", required=True)
    parser.add_argument("--expected-code-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = merge_literature_attention_shards(
        full_rollouts_path=args.full_rollouts,
        expected_full_rollouts_sha256=args.expected_full_rollouts_sha256,
        source_features_path=args.source_features,
        expected_source_features_sha256=args.expected_source_features_sha256,
        shard_rollout_paths=args.shard_rollouts,
        shard_feature_paths=args.shard_features,
        expected_code_revision=args.expected_code_revision,
        output_path=args.output,
        report_path=args.report,
        resume=args.resume,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
