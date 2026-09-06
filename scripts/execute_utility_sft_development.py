"""Verify and execute one hash-bound development pilot arm inside Slurm."""
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
    d=hashlib.sha256()
    with Path(path).open("rb") as h:
        for block in iter(lambda:h.read(1024*1024),b""): d.update(block)
    return d.hexdigest()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--plan",required=True); parser.add_argument("--sha256",required=True)
    args=parser.parse_args()
    if sha(args.plan)!=args.sha256: raise ValueError("development plan changed")
    plan=json.loads(Path(args.plan).read_text()); root=Path(__file__).resolve().parents[1]
    if plan.get("schema") not in (
        "utility_sft_development_pilot_plan_v1",
        "utility_sft_development_correction_plan_v1",
    ) or plan.get("test_authorized") is not False:
        raise ValueError("invalid development plan")
    for rel,expected in plan["code_hashes"].items():
        if sha(root/rel)!=expected: raise ValueError(f"code changed after freeze: {rel}")
    for name in ("config","bundle"):
        if sha(plan[name]["path"])!=plan[name]["sha256"]: raise ValueError(f"{name} changed")
    if shutil.disk_usage(root).free<8*1024**3: raise RuntimeError("development arm needs 8 GiB free")
    devices=subprocess.check_output(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],text=True)
    if "H800" not in devices or not os.environ.get("SLURM_JOB_ID"): raise RuntimeError("H800 Slurm required")
    run=Path(plan["output_root"])/f"job-{os.environ['SLURM_JOB_ID']}"
    subprocess.run([sys.executable,str(root/"scripts/train_utility_sft_development.py"),
        "--config",plan["config"]["path"],"--development-bundle",plan["bundle"]["path"],
        "--output",str(run)],check=True)


if __name__=="__main__": main()
