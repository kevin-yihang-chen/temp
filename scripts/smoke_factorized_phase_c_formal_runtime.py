#!/usr/bin/env python3
"""Real Qwen selector-load and semantic-control smoke before formal access."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import torch

from beyond_entropy.phase_c_formal_scoring import ablated_policy_inputs
from beyond_entropy.phase_c_formal_transaction import FORMAL_MODES
from beyond_entropy.phase_c_training import METHODS, SEEDS, validate_phase_c_training_matrix
from beyond_entropy.predictability_matrix_artifacts import (
    atomic_json_write_exclusive,
    sha256_file,
)
from beyond_entropy.sequential_post_training import (
    load_sequential_training_examples,
    state_hash_subset,
)
from scripts.score_factorized_phase_c_formal import _load_model


def _git_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or args.seed not in SEEDS:
        raise RuntimeError("runtime smoke requires Slurm and a frozen Phase-C seed")
    root = Path(__file__).resolve().parents[1]
    revision = _git_revision(root)
    if subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
        check=True, capture_output=True, text=True,
    ).stdout.strip():
        raise ValueError("runtime smoke requires a clean tracked worktree")
    matrix_path = Path(args.matrix).resolve()
    matrix = json.loads(matrix_path.read_text())
    datasets = validate_phase_c_training_matrix(matrix, root)
    # Two previously opened development examples are enough for every
    # derangement. The subset is selected from IDs, never outcomes.
    validation = datasets["chartqa"]["validation"]
    examples = state_hash_subset(
        load_sequential_training_examples(
            validation["rollouts"]["path"], validation["manifest"]["path"],
        ),
        maximum_states=2, seed=args.seed,
        namespace="phase-c-formal-runtime-smoke-v1",
    )

    training_root = Path(args.training_root).resolve()
    selectors: dict[str, dict[str, dict[str, Any]]] = {
        method: {} for method in METHODS
    }
    config_reference = None
    for method in METHODS:
        run = training_root / method.replace("_", "-") / f"job-{args.job_id}"
        report_path, selector_path = run / "report.json", run / "selector.pt"
        report = json.loads(report_path.read_text())
        if (
            report.get("stage") != "phase_c_training"
            or report.get("method") != method
            or report.get("test_accessed") is not False
            or report.get("provenance", {}).get("config", {}).get("seed") != args.seed
            or report.get("selector_sha256") != sha256_file(selector_path)
        ):
            raise ValueError(f"invalid runtime-smoke selector: {method}")
        config = report["provenance"]["config"]
        current_reference = (config["model"], config["revision"])
        if config_reference is None:
            config_reference = current_reference
        elif current_reference != config_reference:
            raise ValueError("runtime-smoke selector model revisions differ")
        selectors[method][str(args.seed)] = {
            "selector": {"path": str(selector_path), "sha256": sha256_file(selector_path)},
            "training_report": {"path": str(report_path), "sha256": sha256_file(report_path)},
        }
    assert config_reference is not None
    plan = {
        "model": config_reference[0], "model_revision": config_reference[1],
        "selectors": selectors,
    }

    traces = []
    for method in METHODS:
        model = _load_model(plan, method, args.seed)
        modes = FORMAL_MODES if method == "factorized_potential_outcomes" else ("original",)
        with torch.no_grad():
            for mode in modes:
                views = ablated_policy_inputs(
                    examples, mode=mode, seed=20260913,
                    namespace=f"phase-c-formal-runtime-smoke:{mode}",
                )
                value = model(views[examples[0].decision_id])
                score = float(value["continue_score"][0].float().cpu())
                if not math.isfinite(score):
                    raise RuntimeError("runtime-smoke selector score is non-finite")
                traces.append({
                    "method": method, "mode": mode, "continue_score": score,
                    "measurement": dict(model.last_measurement),
                })
        del model
        gc.collect()
        torch.cuda.empty_cache()
    checks = {
        "all_methods_loaded": {item["method"] for item in traces} == set(METHODS),
        "all_semantic_modes_executed": {
            item["mode"] for item in traces
            if item["method"] == "factorized_potential_outcomes"
        } == set(FORMAL_MODES),
        "all_scores_finite": all(math.isfinite(item["continue_score"]) for item in traces),
        "no_proposed_crop_execution": all(
            item["measurement"].get("proposed_crop_executions") == 0 for item in traces
        ),
        "two_images_observed": all(
            item["measurement"].get("observed_images") == 2 for item in traces
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"formal runtime smoke failed: {checks}")
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=False)
    atomic_json_write_exclusive(output, {
        "schema": "factorized_phase_c_formal_runtime_smoke_v1",
        "completed": True, "test_accessed": False,
        "formal_claim_eligible": False,
        "code_revision": revision, "job_id": os.environ["SLURM_JOB_ID"],
        "selector_training_job_id": str(args.job_id), "seed": args.seed,
        "matrix": {"path": str(matrix_path), "sha256": sha256_file(matrix_path)},
        "selectors": selectors, "examples": len(examples),
        "checks": checks, "traces": traces,
    })
    print(json.dumps({"output": str(output), "sha256": sha256_file(output)}))


if __name__ == "__main__":
    main()
