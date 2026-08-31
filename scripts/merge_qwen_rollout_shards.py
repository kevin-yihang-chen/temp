from __future__ import annotations

import argparse
import json
from pathlib import Path

from beyond_entropy.rollout_shards import merge_qwen_rollout_shards


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and merge deterministic collect-qwen rollout shards"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--expected-code-revision")
    parser.add_argument("--expected-scorer")
    parser.add_argument("--require-resume-audit", action="store_true")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()
    result = merge_qwen_rollout_shards(
        manifest_path=args.manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
        run_root=args.run_root,
        shard_count=args.shard_count,
        output_path=args.output,
        limit=args.limit,
        expected_code_revision=args.expected_code_revision,
        expected_scorer=args.expected_scorer,
        require_resume_audit=args.require_resume_audit,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
