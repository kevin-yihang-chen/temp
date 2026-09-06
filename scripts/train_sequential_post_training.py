#!/usr/bin/env python3
"""Train one bounded ChartQA+DocVQA STOP/CONTINUE post-training arm."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from statistics import mean

import torch

from beyond_entropy.predictability_matrix_artifacts import atomic_json_write_exclusive, sha256_file
from beyond_entropy.qwen_semantic import _atomic_torch_save
from beyond_entropy.sequential_post_training import (
    QwenSequentialPolicy,
    deterministic_joint_schedule,
    load_sequential_training_examples,
    sequential_post_training_loss,
    state_hash_subset,
)
from beyond_entropy.utility_training import optimizer_to_device


BENCHMARKS = ("chartqa", "docvqa")


def tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def validate_config(config: dict) -> None:
    if (config.get("schema") != "cv_method_post_training_config_v1"
            or config.get("stage") not in ("phase_a_smoke", "phase_b_pilot", "phase_c_confirmation")
            or config.get("method") not in (
                "outcome_only", "counterfactual_utility",
                "factorized_potential_outcomes",
            )
            or config.get("test_authorized") is not False
            or config.get("trainable_backbone") != "visual_merger_and_last_language_block"
            or set(config.get("datasets", {})) != set(BENCHMARKS)):
        raise ValueError("unsupported CV-method post-training config")
    for key in ("steps", "gradient_accumulation", "checkpoint_interval",
                "head_dim", "min_pixels", "max_pixels"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f"invalid {key}")
    if config["stage"] == "phase_a_smoke" and not 0.05 <= config["train_fraction"] <= 0.10:
        raise ValueError("Phase A must use 5-10% of each train bank")
    if config["stage"] != "phase_a_smoke" and config["train_fraction"] != 1.0:
        raise ValueError("post-smoke stages must use the full frozen train bank")
    if config["gradient_accumulation"] != 1:
        raise ValueError("v1 protocol freezes one state per optimizer step")


def load_data(config: dict) -> dict[str, dict[str, list]]:
    result = {"train": {}, "validation": {}}
    train_sources, validation_sources = set(), set()
    train_images, validation_images = set(), set()
    for benchmark in BENCHMARKS:
        for role in ("train", "validation"):
            spec = config["datasets"][benchmark][role]
            for field in ("manifest", "rollouts"):
                if sha256_file(spec[field]["path"]) != spec[field]["sha256"]:
                    raise ValueError(f"{benchmark}.{role}.{field} hash mismatch")
            values = load_sequential_training_examples(
                spec["rollouts"]["path"], spec["manifest"]["path"]
            )
            expected = int(spec["states"])
            if len(values) != expected:
                raise ValueError(f"{benchmark}.{role} expected {expected} states")
            if config["stage"] == "phase_a_smoke":
                maximum = max(1, int(expected * config["train_fraction"]))
                values = state_hash_subset(
                    values, maximum_states=maximum, seed=config["seed"],
                    namespace=f"cv-method-v1:{benchmark}:{role}",
                )
            result[role][benchmark] = values
        train_sources.update(item.inputs.source_id for item in result["train"][benchmark])
        validation_sources.update(item.inputs.source_id for item in result["validation"][benchmark])
        train_images.update(item.inputs.image_id for item in result["train"][benchmark])
        validation_images.update(item.inputs.image_id for item in result["validation"][benchmark])
    if train_sources & validation_sources or train_images & validation_images:
        raise ValueError("train/validation source or image leakage")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    validate_config(config)
    if not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("real post-training must run inside a GPU Slurm allocation")
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    run = Path(args.output).resolve()
    code_paths = sorted((root / "src/beyond_entropy").glob("*.py")) + [Path(__file__).resolve()]
    provenance = {
        "config": config, "config_path": str(config_path),
        "config_sha256": sha256_file(config_path), "code_revision": revision,
        "code_hashes": {str(path.relative_to(root)): sha256_file(path) for path in code_paths},
        "test_accessed": False,
    }
    if args.resume:
        if json.loads((run / "started.json").read_text()) != provenance or (run / "report.json").exists():
            raise ValueError("resume provenance mismatch or run already complete")
    else:
        if run.exists():
            raise FileExistsError("refusing to reuse an existing run directory")
        atomic_json_write_exclusive(run / "started.json", provenance)
    data = load_data(config)
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    started = time.monotonic()
    backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config["model"], revision=config["revision"], local_files_only=True,
        dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(
        config["model"], revision=config["revision"], local_files_only=True,
        min_pixels=config["min_pixels"], max_pixels=config["max_pixels"],
    )
    model = QwenSequentialPolicy(
        backbone, processor, head_dim=config["head_dim"],
        min_pixels=config["min_pixels"], max_pixels=config["max_pixels"],
        head_outputs=3 if config["method"] == "factorized_potential_outcomes" else 2,
    )
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW([
        {"params": [p for name, p in model.named_parameters()
                    if p.requires_grad and name.startswith("backbone.")],
         "lr": config["learning_rate"]},
        {"params": model.head.parameters(), "lr": config["head_learning_rate"]},
    ], weight_decay=config["weight_decay"])
    schedule = deterministic_joint_schedule(
        data["train"], draws=config["steps"], seed=seed,
        namespace=f"cv-method-v1:{config['stage']}:matched-schedule",
    )
    schedule_ids = [f"{domain}:{item.inputs.state_id}:{item.replicate_id}"
                    for domain, item in schedule]
    schedule_sha256 = hashlib.sha256("\n".join(schedule_ids).encode()).hexdigest()
    audit_examples = []
    for domain in BENCHMARKS:
        candidates = [
            item for item in data["train"][domain]
            if (
                item.gain != 0
                if config["method"] == "counterfactual_utility"
                else (
                    True
                    if config["method"] == "factorized_potential_outcomes"
                    else item.stop_reward + item.continue_reward > 0
                )
            )
        ]
        audit_examples.extend((domain, item) for item in candidates[:4])
    if not audit_examples:
        raise ValueError("training loss audit has no informative examples")

    @torch.no_grad()
    def audit_loss() -> float:
        model.eval()
        values = []
        for _domain, item in audit_examples:
            output = model(item.inputs)
            values.append(float(sequential_post_training_loss(
                output["action_logits"], item, method=config["method"]
            )))
        return mean(values)

    before = {name: tensor_hash(value) for name, value in model.named_parameters()
              if value.requires_grad}
    trace, start_step = [], 0
    peak_gradients = {"head": 0.0, "visual_merger": 0.0, "language_last": 0.0}
    if args.resume:
        saved = torch.load(run / "resume.pt", map_location="cpu", weights_only=False)
        if saved["provenance"] != provenance or saved["schedule_sha256"] != schedule_sha256:
            raise ValueError("resume checkpoint provenance/schedule mismatch")
        model.load_state_dict(saved["parameters"], strict=False)
        optimizer.load_state_dict(saved["optimizer"])
        optimizer_to_device(optimizer, device)
        before, trace, start_step = saved["before"], saved["trace"], saved["step"]
        initial_audit_loss = saved["initial_audit_loss"]
        peak_gradients = saved["peak_gradients"]
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    else:
        initial_audit_loss = audit_loss()
    model.train()
    for step in range(start_step, config["steps"]):
        optimizer.zero_grad(set_to_none=True)
        domain, example = schedule[step]
        tick = time.monotonic()
        output = model(example.inputs)
        loss = sequential_post_training_loss(
            output["action_logits"], example, method=config["method"]
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training loss")
        loss.backward()
        gradients = model.gradient_report()
        for group, value in gradients.items():
            peak_gradients[group] = max(peak_gradients[group], value)
        norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            config["gradient_clip"], error_if_nonfinite=True,
        )
        optimizer.step()
        row = {
            "step": step + 1, "benchmark": domain,
            "state_id": example.inputs.state_id, "gain": example.gain,
            "loss": float(loss.detach()), "gradient_norm": float(norm),
            "seconds": time.monotonic() - tick, "gradients": gradients,
        }
        trace.append(row)
        print(json.dumps(row), flush=True)
        if (step + 1) % config["checkpoint_interval"] == 0 or step + 1 == config["steps"]:
            _atomic_torch_save({
                "provenance": provenance, "parameters": model.trainable_state_dict(),
                "optimizer": optimizer.state_dict(), "step": step + 1,
                "trace": trace, "before": before, "peak_gradients": peak_gradients,
                "schedule_sha256": schedule_sha256,
                "initial_audit_loss": initial_audit_loss,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
            }, run / "resume.pt")

    final_audit_loss = audit_loss()
    predictions = []
    model.eval()
    with torch.no_grad():
        for benchmark in BENCHMARKS:
            for example in data["validation"][benchmark]:
                output = model(example.inputs)
                logits = output["action_logits"][0].float().cpu().tolist()
                continue_score = float(output["continue_score"][0].float().cpu())
                prediction = {
                    "benchmark": benchmark, "state_id": example.inputs.state_id,
                    "replicate_id": example.replicate_id,
                    "source_id": example.inputs.source_id,
                    "stop_reward": example.stop_reward,
                    "continue_reward": example.continue_reward,
                    "gain": example.gain, "action_logits": logits,
                    "continue_score": continue_score,
                    "natural_action": "CONTINUE" if continue_score > 0 else "STOP",
                    "measurement": dict(model.last_measurement),
                }
                if config["method"] == "factorized_potential_outcomes":
                    prediction["factorized_probabilities"] = {
                        key: float(output[key][0].float().cpu())
                        for key in (
                            "error_probability",
                            "rescue_probability_given_error",
                            "harm_probability_given_correct",
                            "expected_gain",
                        )
                    }
                else:
                    prediction["continue_probability"] = float(torch.softmax(
                        output["action_logits"].float(), dim=-1
                    )[0, 1].cpu())
                predictions.append(prediction)
    _atomic_torch_save(
        {"provenance": provenance, "parameters": model.trainable_state_dict()},
        run / "selector.pt",
    )
    changed = [name for name, value in model.named_parameters()
               if value.requires_grad and tensor_hash(value) != before[name]]
    score_values = [row["continue_score"] for row in predictions]
    checks = {
        "paired_reward_gain_contract": all(
            item.gain == item.continue_reward - item.stop_reward
            for role in data.values() for values in role.values() for item in values
        ),
        "binary_action_support_valid": all(
            item.inputs.proposed_action_id
            not in {observed.action_id for observed in item.inputs.acquired_observations}
            for role in data.values() for values in role.values() for item in values
        ),
        "finite_trace": all(math.isfinite(row["loss"]) and math.isfinite(row["gradient_norm"])
                            for row in trace),
        "loss_positive_and_decreased_on_fixed_audit": (
            initial_audit_loss > 0 and final_audit_loss < initial_audit_loss
        ),
        "all_trainable_groups_received_gradient": all(value > 0 for value in peak_gradients.values()),
        "all_trainable_groups_updated": (
            any(name.startswith("head.") for name in changed)
            and any("visual.merger" in name for name in changed)
            and any("language_model.layers" in name for name in changed)
        ),
        "no_proposed_crop_execution": all(
            row["measurement"]["proposed_crop_executions"] == 0 for row in predictions
        ),
        "finite_nonconstant_validation_scores": (
            all(math.isfinite(value) for value in score_values)
            and len(set(round(value, 8) for value in score_values)) > 1
        ),
        "no_action_collapse": 0 < sum(value > 0 for value in score_values) < len(score_values),
    }
    report = {
        "schema": "cv_method_post_training_report_v1",
        "stage": config["stage"], "method": config["method"],
        **({
            "factorization": {
                "reward_domain": "bounded_[0,1]",
                "error_mass_target": "1 - stop_reward",
                "rescue_fraction_target": (
                    "max(continue_reward-stop_reward,0)/(1-stop_reward)"
                ),
                "harm_fraction_target": (
                    "max(stop_reward-continue_reward,0)/stop_reward"
                ),
                "identity": (
                    "gain = error_mass*rescue_fraction "
                    "- correct_mass*harm_fraction"
                ),
                "conditional_loss_weights": "error_mass_and_correct_mass",
            }
        } if config["method"] == "factorized_potential_outcomes" else {}),
        "scientific_status": "engineering_smoke" if config["stage"] == "phase_a_smoke" else "development_pilot",
        "test_accessed": False, "formal_claim_eligible": config["stage"] == "phase_c_confirmation",
        "provenance": provenance, "schedule_sha256": schedule_sha256,
        "train": {
            "states": {name: len(data["train"][name]) for name in BENCHMARKS},
            "gain_counts": {name: {
                "beneficial": sum(item.gain > 0 for item in data["train"][name]),
                "harmful": sum(item.gain < 0 for item in data["train"][name]),
                "neutral": sum(item.gain == 0 for item in data["train"][name]),
            } for name in BENCHMARKS},
            "schedule_draws": {name: sum(domain == name for domain, _ in schedule)
                               for name in BENCHMARKS},
        },
        "validation": {
            "states": {name: len(data["validation"][name]) for name in BENCHMARKS},
            "predictions": predictions,
        },
        "checks": checks,
        "smoke_passed": config["stage"] != "phase_a_smoke" or all(checks.values()),
        "trace": trace, "peak_gradients": peak_gradients,
        "fixed_train_loss_audit": {
            "examples": [f"{domain}:{item.inputs.state_id}:{item.replicate_id}"
                         for domain, item in audit_examples],
            "initial": initial_audit_loss, "final": final_audit_loss,
        },
        "changed_parameter_names": changed,
        "selector_sha256": sha256_file(run / "selector.pt"),
        "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "gpu": torch.cuda.get_device_name(), "job_id": os.environ["SLURM_JOB_ID"],
        "versions": {package: importlib.metadata.version(package)
                     for package in ("torch", "transformers", "Pillow")},
    }
    atomic_json_write_exclusive(run / "report.json", report)
    print(json.dumps({"report": str(run / "report.json"), "checks": checks,
                      "smoke_passed": report["smoke_passed"]}), flush=True)


if __name__ == "__main__":
    main()
