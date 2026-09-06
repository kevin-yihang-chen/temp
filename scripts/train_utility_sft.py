"""Bounded train-only Qwen2.5-VL Utility-SFT sanity experiment (no test API)."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import time
from pathlib import Path
from statistics import mean

import torch

from beyond_entropy.predictability_matrix_artifacts import atomic_json_write_exclusive, sha256_file
from beyond_entropy.qwen_semantic import _atomic_torch_save
from beyond_entropy.utility_dataset import load_utility_development
from beyond_entropy.utility_head import utility_sft_loss
from beyond_entropy.utility_qwen import QwenSpatialUtility
from beyond_entropy.utility_training import optimizer_to_device, sanity_passed, supervision_kwargs


def tensor_hash(tensor):
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def select_sanity_subset(samples, maximum):
    """TRAIN-only diagnostic sampling; shared by all arms, never formal evidence."""
    samples = sorted(samples, key=lambda s: s.inputs.state.state_id)
    positive = next((s for s in samples if max(s.gains) > 0), None)
    negative = next((s for s in samples if min(s.gains) < 0), None)
    if positive is None or negative is None or maximum < 2:
        raise ValueError("overfit gate needs both positive- and negative-gain TRAIN actions")
    chosen = {}
    for s in (positive, negative, *samples):
        chosen.setdefault(s.inputs.state.state_id, s)
        if len(chosen) == maximum:
            break
    return list(chosen.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--engineering-steps", type=int, default=0,
                        help="Shorter implementation gate; NEVER a passed overfit result")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if (config.get("schema") != "utility_sft_train_config_v1"
            or config.get("scope") != "train_only_overfit_sanity"
            or config.get("test_authorized") is not False
            or config.get("method") not in ("format", "best_action", "utility")
            or config.get("trainable_backbone") != "visual_merger_and_last_language_block"):
        raise ValueError("unsupported or non-development configuration")
    if not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("real 3B experiment must run inside a GPU Slurm allocation")
    for key in ("steps", "gradient_accumulation", "max_states", "expected_zoom_actions"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"invalid {key}")
    if args.engineering_steps < 0 or args.engineering_steps > config["steps"]:
        raise ValueError("engineering step budget outside configured bounds")
    samples = select_sanity_subset(load_utility_development(args.train_data, role="train"), config["max_states"])
    if any(len(s.gains) != config["expected_zoom_actions"]+1 for s in samples):
        raise ValueError("candidate coverage mismatch")
    run = Path(args.output).resolve()
    root = Path(__file__).resolve().parents[1]
    code_paths = sorted((root / "src/beyond_entropy").glob("*.py")) + [Path(__file__).resolve()]
    provenance = {
        "config": config, "config_sha256": sha256_file(args.config),
        "train_data_sha256": sha256_file(args.train_data),
        "code_hashes": {str(p.relative_to(root)): sha256_file(p) for p in code_paths},
        "engineering_steps": args.engineering_steps,
        "state_ids": [s.inputs.state.state_id for s in samples],
        "selection": "train-only first positive/negative then ID order; shared all arms",
        "test_accessed": False,
    }
    if args.resume:
        if json.loads((run / "started.json").read_text()) != provenance or (run / "report.json").exists():
            raise ValueError("resume configuration/code/data mismatch or run already complete")
    else:
        if run.exists():
            raise FileExistsError("refusing to reuse an existing run directory")
        atomic_json_write_exclusive(run / "started.json", provenance)
    seed = config["seed"]
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
    model = QwenSpatialUtility(
        backbone, processor, temperature=config["temperature"], head_dim=config["head_dim"],
        min_pixels=config["min_pixels"], max_pixels=config["max_pixels"],
    )
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone.")], "lr": config["learning_rate"]},
        {"params": model.head.parameters(), "lr": config["head_learning_rate"]},
    ], weight_decay=config["weight_decay"])
    before_hashes = {n: tensor_hash(p) for n, p in model.named_parameters() if p.requires_grad}
    peak_gradients = {"head": 0., "visual_merger": 0., "language_last": 0.}
    train_trace, start_step = [], 0

    def objective(out, sample):
        return utility_sft_loss(out["action_logits"], **supervision_kwargs(
            sample, method=config["method"], temperature=config["temperature"],
            device=next(model.parameters()).device,
        ))

    @torch.no_grad()
    def evaluate():
        model.eval()
        rows = []
        for s in samples:
            out = model(s.inputs)
            predicted = out["predicted_gain"][0].float().cpu().tolist()
            selected = s.inputs.action_space.select(predicted, lambda_cost=0.)
            rows.append({
                "state_id": s.inputs.state.state_id, "source_id": s.inputs.state.source_id,
                "gains": s.gains, "predicted_gain": predicted,
                "action_logits": out["action_logits"][0].float().cpu().tolist(),
                "selected": selected, "loss": float(objective(out, s).item()),
                "regret": max(s.gains)-s.gains[selected],
                "support_correct": selected == s.support_action,
                "measurement": dict(model.last_measurement),
            })
        return rows

    if args.resume:
        saved = torch.load(run / "resume.pt", map_location="cpu", weights_only=False)
        if saved["provenance"] != provenance:
            raise ValueError("checkpoint provenance mismatch")
        model.load_state_dict(saved["parameters"], strict=False)
        optimizer.load_state_dict(saved["optimizer"])
        optimizer_to_device(optimizer, next(model.parameters()).device)
        before_hashes, initial = saved["before_hashes"], saved["initial"]
        peak_gradients, train_trace, start_step = saved["peak_gradients"], saved["trace"], saved["step"]
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    else:
        initial = evaluate()
    steps = args.engineering_steps or config["steps"]
    for step in range(start_step, steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        rng = random.Random(seed*100000 + step)
        batch = [samples[rng.randrange(len(samples))] for _ in range(config["gradient_accumulation"])]
        losses = []
        tick = time.monotonic()
        for s in batch:
            out = model(s.inputs)
            loss = objective(out, s)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            (loss / len(batch)).backward()
            losses.append(float(loss.detach().item()))
        gradients = model.gradient_report()
        for group, value in gradients.items():
            peak_gradients[group] = max(value, peak_gradients[group])
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        item = {"step": step+1, "loss": mean(losses), "gradient_norm": float(norm),
                "seconds": time.monotonic()-tick, "gradients": gradients}
        train_trace.append(item)
        print(json.dumps(item), flush=True)
        if (step+1) % 10 == 0 or step+1 == steps:
            _atomic_torch_save({
                "provenance": provenance, "parameters": model.trainable_state_dict(),
                "optimizer": optimizer.state_dict(), "step": step+1, "trace": train_trace,
                "before_hashes": before_hashes, "initial": initial, "peak_gradients": peak_gradients,
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
            }, run / "resume.pt")
    final = evaluate()
    after_hashes = {n: tensor_hash(p) for n, p in model.named_parameters() if p.requires_grad}
    changed = [n for n in before_hashes if before_hashes[n] != after_hashes[n]]
    positive = [p for r in final for g, p in zip(r["gains"], r["predicted_gain"]) if g > 0]
    negative = [p for r in final for g, p in zip(r["gains"], r["predicted_gain"]) if g < 0]
    checks = {
        "finite_loss_decreased": mean(r["loss"] for r in final) < mean(r["loss"] for r in initial),
        "backbone_gradients_nonzero": all(v > 0 for v in peak_gradients.values()),
        "vision_merger_updated": any("visual.merger" in n for n in changed),
        "language_block_updated": any("language_model.layers" in n for n in changed),
        "head_updated": any(n.startswith("head.") for n in changed),
        "single_original_image_no_candidate_execution": all(r["measurement"]["vision_encoder_calls"] == 1 and r["measurement"]["candidate_crop_executions"] == 0 for r in final),
        "positive_negative_separated": bool(positive and negative and min(positive) > max(negative)),
        "overfit_regret_zero": mean(r["regret"] for r in final) <= 1e-8,
        "support_memorized": all(r["support_correct"] for r in final),
    }
    _atomic_torch_save({"provenance": provenance, "parameters": model.trainable_state_dict()}, run / "selector.pt")
    report = {
        "schema": "utility_sft_train_sanity_v1", "scientific_status": "engineering_only" if args.engineering_steps else "train_overfit_only_not_generalization",
        "method": config["method"], "test_accessed": False, "formal_claim_eligible": False,
        "checks": checks, "overfit_passed": sanity_passed(
            config["method"], checks, engineering=bool(args.engineering_steps)
        ),
        "provenance": provenance, "initial": initial, "final": final, "trace": train_trace,
        "changed_parameter_names": changed, "peak_gradients": peak_gradients,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "elapsed_seconds": time.monotonic()-started, "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "gpu": torch.cuda.get_device_name(), "job_id": os.environ["SLURM_JOB_ID"],
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "Pillow")},
        "selector_sha256": sha256_file(run / "selector.pt"),
    }
    atomic_json_write_exclusive(run / "report.json", report)
    print(json.dumps({"report": str(run / "report.json"), "checks": checks, "overfit_passed": report["overfit_passed"]}), flush=True)


if __name__ == "__main__":
    main()
