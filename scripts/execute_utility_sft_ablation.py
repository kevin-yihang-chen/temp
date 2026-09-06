"""Verify and execute one frozen Utility-SFT semantic-ablation plan."""
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
    if sha256_file(args.plan) != args.sha256:
        raise ValueError("semantic-ablation plan changed")
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[1]
    if (
        plan.get("schema") != "utility_sft_semantic_ablation_plan_v1"
        or plan.get("test_authorized") is not False
        or plan.get("formal_claim_eligible") is not False
    ):
        raise ValueError("invalid semantic-ablation plan")
    for relative, expected in plan["code_hashes"].items():
        if sha256_file(root / relative) != expected:
            raise ValueError(f"code changed after ablation freeze: {relative}")
    for name in ("config", "report", "selector", "validation_freeze"):
        if sha256_file(plan[name]["path"]) != plan[name]["sha256"]:
            raise ValueError(f"frozen {name} changed")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("semantic ablation requires Slurm")
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    )
    if "H800" not in devices:
        raise RuntimeError("semantic ablation requires H800")
    if shutil.disk_usage(root).free < 4 * 1024**3:
        raise RuntimeError("semantic ablation requires 4 GiB free")
    run = Path(plan["output_root"]) / f"job-{os.environ['SLURM_JOB_ID']}"
    output = run / "report.json"
    subprocess.run([
        sys.executable, str(root / "scripts/run_utility_sft_ablation.py"),
        "--config", plan["config"]["path"],
        "--report", plan["report"]["path"],
        "--selector", plan["selector"]["path"],
        "--selector-sha256", plan["selector"]["sha256"],
        "--validation-freeze", plan["validation_freeze"]["path"],
        "--validation-freeze-sha256", plan["validation_freeze"]["sha256"],
        "--output", str(output),
    ], check=True)


if __name__ == "__main__":
    main()
