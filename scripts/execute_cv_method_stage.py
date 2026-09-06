#!/usr/bin/env python3
"""Verify a frozen plan, run two one-GPU arms, then evaluate them."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from beyond_entropy.predictability_matrix_artifacts import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    plan_path = Path(args.plan).resolve()
    if sha256_file(plan_path) != args.sha256:
        raise ValueError("stage plan hash mismatch")
    plan = json.loads(plan_path.read_text())
    if plan.get("schema") != "cv_method_stage_plan_v1" or plan.get("test_authorized") is not False:
        raise ValueError("invalid development-only stage plan")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if revision != plan["code_revision"]:
        raise ValueError("repository revision changed after stage freeze")
    for relative, expected in plan["code_hashes"].items():
        if sha256_file(root / relative) != expected:
            raise ValueError(f"code changed after stage freeze: {relative}")
    for spec in plan["configs"].values():
        if sha256_file(spec["path"]) != spec["sha256"]:
            raise ValueError("configuration changed after stage freeze")
    if shutil.disk_usage(root).free < 8 * 1024**3:
        raise RuntimeError("two-arm stage requires at least 8 GiB free")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("stage executor requires Slurm")
    output_root = Path(plan["output_root"])
    job = f"job-{os.environ['SLURM_JOB_ID']}"
    processes = []
    for gpu, method in enumerate(("outcome_only", "counterfactual_utility")):
        destination = output_root / method.replace("_", "-") / job
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        command = [
            sys.executable, str(root / "scripts/train_sequential_post_training.py"),
            "--config", plan["configs"][method]["path"],
            "--output", str(destination),
        ]
        processes.append((method, destination, subprocess.Popen(command, env=environment)))
    failures = []
    for method, destination, process in processes:
        return_code = process.wait()
        if return_code:
            failures.append((method, return_code, str(destination)))
    if failures:
        raise RuntimeError(f"one or more post-training arms failed: {failures}")
    destinations = {method: destination for method, destination, _ in processes}
    evaluation = output_root / "evaluation" / job
    subprocess.run([
        sys.executable, str(root / "scripts/evaluate_cv_method_stage.py"),
        "--outcome-report", str(destinations["outcome_only"] / "report.json"),
        "--counterfactual-report", str(destinations["counterfactual_utility"] / "report.json"),
        "--config", plan["configs"]["evaluation"]["path"],
        "--output", str(evaluation),
    ], check=True)


if __name__ == "__main__":
    main()
