#!/usr/bin/env python3
"""Score one frozen selector seed on all Phase-C held-out domains and controls."""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import time
from pathlib import Path

import torch

from beyond_entropy.phase_c_formal_scoring import ablated_policy_inputs
from beyond_entropy.phase_c_formal_transaction import (
    FORMAL_MODES,
    validate_formal_access,
)
from beyond_entropy.phase_c_training import BENCHMARKS, METHODS, SEEDS
from beyond_entropy.predictability_matrix_artifacts import (
    atomic_json_write_exclusive,
    sha256_file,
)
from beyond_entropy.sequential_post_training import (
    QwenSequentialPolicy,
    load_sequential_training_examples,
)


def _git_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _load_examples(plan: dict, benchmark: str):
    spec = plan["benchmarks"][benchmark]
    report_path = Path(spec["merged_output"]) / "report.json"
    report = json.loads(report_path.read_text())
    rollouts = Path(spec["merged_output"]) / "rollouts.jsonl"
    if (
        report.get("schema") != "merged_sequential_rollout_bank_v1"
        or report.get("completed") is not True
        or report.get("test_accessed") is not True
        or report.get("benchmark") != benchmark
        or report.get("dataset_role") != "test"
        or report.get("states") != spec["states"]
        or report.get("manifest_sha256") != spec["manifest_sha256"]
        or report.get("code_revision") != plan["code_revision"]
        or report.get("rollouts_sha256") != sha256_file(rollouts)
    ):
        raise ValueError(f"invalid formal rollout merge for {benchmark}")
    examples = load_sequential_training_examples(rollouts, spec["manifest"])
    if len(examples) != spec["states"]:
        raise ValueError(f"formal example count mismatch for {benchmark}")
    return examples, {"report": report_path, "rollouts": rollouts}


def _load_model(plan: dict, method: str, seed: int):
    evidence = plan["selectors"][method][str(seed)]
    report = json.loads(Path(evidence["training_report"]["path"]).read_text())
    selector = torch.load(
        evidence["selector"]["path"], map_location="cpu", weights_only=False,
    )
    if (
        selector.get("provenance") != report.get("provenance")
        or sha256_file(evidence["selector"]["path"]) != evidence["selector"]["sha256"]
    ):
        raise ValueError("selector payload/provenance differs from frozen evidence")
    config = report["provenance"]["config"]
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        plan["model"], revision=plan["model_revision"], local_files_only=True,
        dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(
        plan["model"], revision=plan["model_revision"], local_files_only=True,
        min_pixels=config["min_pixels"], max_pixels=config["max_pixels"],
    )
    model = QwenSequentialPolicy(
        backbone, processor, head_dim=config["head_dim"],
        min_pixels=config["min_pixels"], max_pixels=config["max_pixels"],
        # Recreate the exact trainable topology before loading the sparse
        # selector state.  ``False`` would silently omit the trained visual
        # merger and final language block from the expected key set.
        train_backbone=True,
        head_outputs=3 if method == "factorized_potential_outcomes" else 2,
    )
    expected = set(model.trainable_state_dict())
    parameters = selector.get("parameters")
    if not isinstance(parameters, dict) or set(parameters) != expected:
        raise ValueError("selector trainable parameter keys differ from model")
    incompatible = model.load_state_dict(parameters, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError("selector contains unexpected parameters")
    model.requires_grad_(False)
    if hasattr(model.backbone, "gradient_checkpointing_disable"):
        model.backbone.gradient_checkpointing_disable()
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.seed not in SEEDS or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("formal scoring requires a frozen seed and Slurm allocation")
    plan, ledger = validate_formal_access(
        args.plan, args.plan_sha256, args.ledger
    )
    root = Path(__file__).resolve().parents[1]
    if _git_revision(root) != plan["code_revision"]:
        raise ValueError("formal scoring code revision drifted")
    output = Path(args.output).resolve()
    if str(output) != plan["predictions"][str(args.seed)] or output.exists():
        raise ValueError("formal prediction output differs from one-shot plan")

    examples_by_benchmark = {}
    rollout_evidence = {}
    for benchmark in BENCHMARKS:
        examples_by_benchmark[benchmark], paths = _load_examples(plan, benchmark)
        rollout_evidence[benchmark] = {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        }

    started = time.monotonic()
    predictions = []
    for method in METHODS:
        model = _load_model(plan, method, args.seed)
        modes = FORMAL_MODES if method == "factorized_potential_outcomes" else ("original",)
        with torch.no_grad():
            for benchmark in BENCHMARKS:
                examples = examples_by_benchmark[benchmark]
                for mode in modes:
                    views = ablated_policy_inputs(
                        examples, mode=mode, seed=plan["ablations"]["seed"],
                        namespace=f"phase-c-formal:{benchmark}:{mode}",
                    )
                    for example in examples:
                        output_value = model(views[example.decision_id])
                        row = {
                            "method": method,
                            "seed": args.seed,
                            "mode": mode,
                            "benchmark": benchmark,
                            "state_id": example.inputs.state_id,
                            "replicate_id": example.replicate_id,
                            "source_id": example.inputs.source_id,
                            "continue_score": float(
                                output_value["continue_score"][0].float().cpu()
                            ),
                            "action_logits": output_value[
                                "action_logits"
                            ][0].float().cpu().tolist(),
                            "measurement": dict(model.last_measurement),
                        }
                        if method == "factorized_potential_outcomes":
                            row["factorized_probabilities"] = {
                                key: float(output_value[key][0].float().cpu())
                                for key in (
                                    "error_probability",
                                    "rescue_probability_given_error",
                                    "harm_probability_given_correct",
                                    "expected_gain",
                                )
                            }
                        predictions.append(row)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if any(
        row["measurement"]["proposed_crop_executions"] != 0
        for row in predictions
    ):
        raise RuntimeError("formal selector executed a proposed crop")
    report = {
        "schema": "factorized_phase_c_formal_predictions_v1",
        "one_shot": True,
        "test_accessed": True,
        "formal_claim_eligible": True,
        "plan_sha256": args.plan_sha256,
        "access_ledger_sha256": sha256_file(args.ledger),
        "seed": args.seed,
        "methods": list(METHODS),
        "modes": list(FORMAL_MODES),
        "rollout_evidence": rollout_evidence,
        "predictions": predictions,
        "elapsed_seconds": time.monotonic() - started,
        "job_id": os.environ["SLURM_JOB_ID"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write_exclusive(output, report)
    print(json.dumps({
        "output": str(output), "sha256": sha256_file(output),
        "predictions": len(predictions),
    }))


if __name__ == "__main__":
    main()
