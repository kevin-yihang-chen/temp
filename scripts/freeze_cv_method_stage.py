#!/usr/bin/env python3
"""Freeze one immutable two-arm CV-method development stage."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from beyond_entropy.predictability_matrix_artifacts import atomic_json_write_exclusive, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcome-config", required=True)
    parser.add_argument("--counterfactual-config", required=True)
    parser.add_argument("--evaluation-config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    configs = {}
    for method, path in (("outcome_only", args.outcome_config),
                         ("counterfactual_utility", args.counterfactual_config)):
        value = json.loads(Path(path).read_text())
        if (value.get("schema") != "cv_method_post_training_config_v1"
                or value.get("method") != method or value.get("test_authorized") is not False):
            raise ValueError(f"invalid {method} configuration")
        configs[method] = value
    stage = configs["outcome_only"]["stage"]
    if configs["counterfactual_utility"]["stage"] != stage:
        raise ValueError("arm stages differ")
    left, right = dict(configs["outcome_only"]), dict(configs["counterfactual_utility"])
    left.pop("method")
    right.pop("method")
    if left != right:
        raise ValueError("post-training arms differ in more than objective method")
    evaluation = json.loads(Path(args.evaluation_config).read_text())
    if (evaluation.get("schema") != "cv_method_evaluation_config_v1"
            or evaluation.get("stage") != stage
            or evaluation.get("test_authorized") is not False):
        raise ValueError("evaluation configuration differs from training stage")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"], cwd=root,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError("tracked worktree must be clean before stage freeze")
    script_names = (
        "train_sequential_post_training.py", "evaluate_cv_method_stage.py",
        "execute_cv_method_stage.py", "freeze_cv_method_stage.py",
        "slurm_cv_method_stage.sh",
    )
    paths = sorted((root / "src/beyond_entropy").glob("*.py")) + [
        root / "scripts" / name for name in script_names
    ]
    payload = {
        "schema": "cv_method_stage_plan_v1", "stage": stage,
        "test_authorized": False, "code_revision": revision,
        "code_hashes": {str(path.relative_to(root)): sha256_file(path) for path in paths},
        "configs": {
            "outcome_only": {"path": str(Path(args.outcome_config).resolve()),
                             "sha256": sha256_file(args.outcome_config)},
            "counterfactual_utility": {
                "path": str(Path(args.counterfactual_config).resolve()),
                "sha256": sha256_file(args.counterfactual_config),
            },
            "evaluation": {"path": str(Path(args.evaluation_config).resolve()),
                           "sha256": sha256_file(args.evaluation_config)},
        },
        "output_root": str(Path(args.output_root).resolve()),
        "resources": {
            "parallel_arms": 2, "gpu_per_arm": 1,
            "rationale": "Independent matched arms run concurrently; no model-state merge. This preserves aggregate GPU-hours while reducing wall-clock time.",
        },
    }
    atomic_json_write_exclusive(args.plan, payload)
    print(sha256_file(args.plan))


if __name__ == "__main__":
    main()
