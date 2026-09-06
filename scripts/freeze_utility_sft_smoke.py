from __future__ import annotations

import argparse
from pathlib import Path

from beyond_entropy.predictability_matrix_artifacts import atomic_json_write_exclusive, sha256_file


def main():
    parser = argparse.ArgumentParser(description="Freeze bounded Utility-SFT training inputs and code before sbatch")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--engineering-steps", type=int, default=2)
    args = parser.parse_args()
    if args.engineering_steps < 0:
        raise ValueError("negative step limit")
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "src/beyond_entropy").glob("*.py"))
    paths += [root / "scripts" / name for name in (
        "train_utility_sft.py", "execute_utility_sft_smoke.py", "slurm_utility_sft_smoke.sh",
        "slurm_utility_sft_controls.sh", "freeze_utility_sft_smoke.py"
    )]
    payload = {
        "schema": "utility_sft_smoke_plan_v1", "test_authorized": False,
        "config": {"path": str(Path(args.config).resolve()), "sha256": sha256_file(args.config)},
        "train_data": {"path": str(Path(args.train_data).resolve()), "sha256": sha256_file(args.train_data)},
        "code_hashes": {str(p.relative_to(root)): sha256_file(p) for p in paths},
        "engineering_steps": args.engineering_steps, "output_root": str(Path(args.output_root).resolve()),
        "gpu": "1 H800", "wall_time_limit_minutes": 30, "maximum_gpu_hours": .5,
        "resource_rationale": (
            "Two-step correctness gate has serial load/backward diagnostics; multiple GPUs add initialization and queue requirements without useful parallel speedup."
            if args.engineering_steps else
            "Single-arm 80-step overfit gate, measured 2.42-2.96 sec/step, estimated 4-6 minutes plus queue. Data-parallelizing a four-example microbatch adds synchronization and queue requirements; reconsider independent-arm parallelism after this sanity gate."
        ),
    }
    atomic_json_write_exclusive(args.plan, payload)
    print(sha256_file(args.plan))


if __name__ == "__main__":
    main()
