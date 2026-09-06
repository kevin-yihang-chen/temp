"""Execute one immutable, bounded train-only smoke plan inside Slurm."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    if sha(args.plan) != args.sha256:
        raise ValueError("smoke plan changed")
    plan = json.loads(Path(args.plan).read_text())
    if plan["schema"] != "utility_sft_smoke_plan_v1" or plan["test_authorized"] is not False:
        raise ValueError("invalid development-only plan")
    root = Path(__file__).resolve().parents[1]
    for relative, expected in plan["code_hashes"].items():
        if sha(root / relative) != expected:
            raise ValueError(f"code changed after smoke freeze: {relative}")
    for label in ("config", "train_data"):
        if sha(plan[label]["path"]) != plan[label]["sha256"]:
            raise ValueError(f"{label} changed after smoke freeze")
    if shutil.disk_usage(root).free < 8*1024**3:
        raise RuntimeError("smoke needs 8 GiB free for two bounded checkpoint files")
    devices = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True)
    if "H800" not in devices:
        raise RuntimeError("smoke plan requires H800")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm allocation required")
    run = Path(plan["output_root"]) / f"job-{os.environ['SLURM_JOB_ID']}"
    command = [sys.executable, str(root / "scripts/train_utility_sft.py"),
               "--config", plan["config"]["path"], "--train-data", plan["train_data"]["path"],
               "--output", str(run)]
    if plan["engineering_steps"]:
        command.extend(["--engineering-steps", str(plan["engineering_steps"])])
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
