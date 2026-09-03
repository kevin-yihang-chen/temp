from __future__ import annotations

import argparse
import json
from pathlib import Path

from beyond_entropy.predictability_feature_shards import (
    merge_predictability_feature_shards,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and merge deterministic predictability feature shards"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--merged-rollouts", type=Path, required=True)
    parser.add_argument("--expected-merged-rollouts-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--shard-key", choices=("state_id", "source_id"), required=True)
    parser.add_argument("--shard-namespace", default="")
    parser.add_argument("--expected-code-revision", required=True)
    parser.add_argument(
        "--dataset-role", choices=("train", "validation", "test"), required=True
    )
    parser.add_argument("--feature-name", default="features.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = merge_predictability_feature_shards(
        manifest_path=args.manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
        merged_rollouts_path=args.merged_rollouts,
        expected_merged_rollouts_sha256=args.expected_merged_rollouts_sha256,
        run_root=args.run_root,
        shard_count=args.shard_count,
        shard_key=args.shard_key,
        shard_namespace=args.shard_namespace,
        expected_code_revision=args.expected_code_revision,
        dataset_role=args.dataset_role,
        output_path=args.output,
        report_path=args.report,
        feature_name=args.feature_name,
    )
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
