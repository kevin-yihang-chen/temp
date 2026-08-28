#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from beyond_entropy.action_bank import summarize_action_bank
from beyond_entropy.dataset import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize one complete sibling counterfactual action bank"
    )
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rollouts-sha256")
    parser.add_argument("--lambda-cost", type=float, default=0.05)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--cluster-by",
        choices=("state_id", "image_id", "source_id"),
        default="source_id",
    )
    args = parser.parse_args()

    rollouts = args.rollouts.resolve()
    digest = hashlib.sha256(rollouts.read_bytes()).hexdigest()
    if args.expected_rollouts_sha256 and digest != args.expected_rollouts_sha256:
        raise ValueError("rollout SHA-256 mismatch")
    report = summarize_action_bank(
        read_jsonl(rollouts),
        lambda_cost=args.lambda_cost,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        cluster_by=args.cluster_by,
    )
    report["run"] = {
        "code_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "rollouts": str(rollouts),
        "rollouts_sha256": digest,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
        "cluster_by": args.cluster_by,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
