#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from beyond_entropy.rollout_runtime_recovery import (
    prepare_runtime_replay_plan,
    repair_runtime_from_replays,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover Qwen rollout runtime provenance by exact replay")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--rollout-root", type=Path, required=True)
    prepare.add_argument("--replay-root", type=Path, required=True)
    prepare.add_argument("--expected-manifest-sha256", required=True)
    prepare.add_argument("--shard-count", type=int, default=4)
    repair = subparsers.add_parser("repair")
    repair.add_argument("--plan", type=Path, required=True)
    repair.add_argument("--code-revision", required=True)
    repair.add_argument("--prior-job-id", action="append", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_runtime_replay_plan(
            manifest=args.manifest,
            rollout_root=args.rollout_root,
            replay_root=args.replay_root,
            expected_manifest_sha256=args.expected_manifest_sha256,
            shard_count=args.shard_count,
        )
    else:
        result = repair_runtime_from_replays(
            plan=args.plan,
            code_revision=args.code_revision,
            prior_job_ids=args.prior_job_id,
        )
    print(json.dumps({"schema": result["schema"]}, sort_keys=True))


if __name__ == "__main__":
    main()
