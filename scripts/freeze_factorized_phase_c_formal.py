#!/usr/bin/env python3
"""Freeze completed selectors and the one-shot Phase-C formal transaction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from beyond_entropy.phase_c_formal_transaction import build_phase_c_formal_plan
from beyond_entropy.phase_c_training import SEEDS


def _seed_job(value: str) -> tuple[int, str]:
    try:
        seed_text, job_id = value.split("=", 1)
        seed = int(seed_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("seed job must be SEED=JOB_ID") from exc
    if seed not in SEEDS or not job_id.isdigit():
        raise argparse.ArgumentTypeError("seed job is outside the frozen Phase-C matrix")
    return seed, job_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--seed-job", type=_seed_job, action="append", required=True)
    parser.add_argument("--transaction-id", required=True)
    parser.add_argument("--runtime-smoke", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    seed_jobs = dict(args.seed_job)
    if len(seed_jobs) != len(args.seed_job):
        raise ValueError("duplicate seed-job mapping")
    plan = build_phase_c_formal_plan(
        config_path=args.config,
        repository_root=args.repository_root,
        seed_jobs=seed_jobs,
        transaction_id=args.transaction_id,
        output_path=args.output,
        runtime_smoke_path=args.runtime_smoke,
    )
    print(json.dumps({
        "plan": str(Path(args.output).resolve()),
        "plan_sha256": plan["plan_sha256"],
        "code_revision": plan["code_revision"],
        "test_accessed": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
