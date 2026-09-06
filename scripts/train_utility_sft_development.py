"""Train one SFT arm on a bounded three-domain development pilot; no test API."""
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
from beyond_entropy.utility_training import (
    optimizer_to_device,
    source_cycle_samples,
    source_hash_subset,
    supervision_kwargs,
)


BENCHMARKS = ("chartqa", "docvqa", "hrbench")


def tensor_hash(tensor):
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def load_pilot(bundle_path, *, train_sources, validation_sources, seed):
    bundle = json.loads(Path(bundle_path).read_text())
    if (bundle.get("schema") != "utility_sft_development_bundle_v1"
            or bundle.get("test_data_present") is not False
            or bundle.get("formal_test_eligible") is not False
            or bundle.get("split_audit", {}).get("passed") is not True):
        raise ValueError("invalid development-only bundle")
    result = {"train": {}, "validation": {}}
    for benchmark in BENCHMARKS:
        for role, limit in (("train", train_sources), ("validation", validation_sources)):
            entry = bundle["inventory"][f"{benchmark}.{role}"]
            if sha256_file(entry["path"]) != entry["sha256"]:
                raise ValueError("development dataset changed after bundle freeze")
            loaded = load_utility_development(entry["path"], role=role)
            if not loaded or {x.benchmark for x in loaded} != {benchmark}:
                raise ValueError("development benchmark mismatch")
            selected = source_hash_subset(
                loaded, maximum_sources=limit, seed=seed,
                namespace=f"utility-development-pilot-v1:{benchmark}:{role}",
            )
            if len({x.inputs.state.source_id for x in selected}) != min(limit, entry["sources"]):
                raise ValueError("whole-source pilot selection coverage mismatch")
            result[role][benchmark] = selected
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--development-bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if (config.get("schema") != "utility_sft_train_config_v1"
            or config.get("scope") not in (
                "three_domain_development_pilot",
                "three_domain_development_correction",
            )
            or config.get("test_authorized") is not False
            or config.get("method") not in ("format", "best_action", "utility")
            or config.get("domain_sampling") not in (
                "uniform_domain_then_source",
                "uniform_domain_then_source_cycle",
            )):
        raise ValueError("invalid development config")
    if ((config["scope"] == "three_domain_development_pilot")
            != (config["domain_sampling"] == "uniform_domain_then_source")):
        raise ValueError("scope and domain sampling protocol disagree")
    if not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("real development pilot requires GPU Slurm")
    for key in ("steps", "gradient_accumulation", "train_sources_per_benchmark",
                "validation_sources_per_benchmark", "checkpoint_interval"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"invalid {key}")
    run = Path(args.output).resolve()
    root = Path(__file__).resolve().parents[1]
    code_paths = sorted((root / "src/beyond_entropy").glob("*.py")) + [Path(__file__).resolve()]
    provenance = {
        "config": config, "config_sha256": sha256_file(args.config),
        "development_bundle": str(Path(args.development_bundle).resolve()),
        "development_bundle_sha256": sha256_file(args.development_bundle),
        "code_hashes": {str(p.relative_to(root)): sha256_file(p) for p in code_paths},
        "test_accessed": False,
    }
    if args.resume:
        if json.loads((run/"started.json").read_text()) != provenance or (run/"report.json").exists():
            raise ValueError("resume mismatch or completed run")
    else:
        if run.exists():
            raise FileExistsError("refusing to reuse development run")
        atomic_json_write_exclusive(run/"started.json", provenance)
    data = load_pilot(
        args.development_bundle,
        train_sources=config["train_sources_per_benchmark"],
        validation_sources=config["validation_sources_per_benchmark"], seed=config["seed"],
    )
    if any(not any(max(s.gains)>0 for s in data["train"][b]) for b in BENCHMARKS):
        raise ValueError("each training domain must contain positive-gain support")
    if any(
        len(s.gains) != int(config["expected_zoom_actions"])+1
        for role in ("train", "validation")
        for benchmark in BENCHMARKS for s in data[role][benchmark]
    ):
        raise ValueError("candidate action coverage mismatch")
    seed = config["seed"]
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    tick = time.monotonic()
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
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW([
        {"params": [p for n,p in model.named_parameters() if p.requires_grad and n.startswith("backbone.")], "lr": config["learning_rate"]},
        {"params": model.head.parameters(), "lr": config["head_learning_rate"]},
    ], weight_decay=config["weight_decay"])
    before = {n:tensor_hash(p) for n,p in model.named_parameters() if p.requires_grad}
    trace, start_step = [], 0
    peak_gradients = {"head":0., "visual_merger":0., "language_last":0.}
    if args.resume:
        saved = torch.load(run/"resume.pt", map_location="cpu", weights_only=False)
        if saved["provenance"] != provenance:
            raise ValueError("resume provenance mismatch")
        model.load_state_dict(saved["parameters"], strict=False)
        optimizer.load_state_dict(saved["optimizer"]); optimizer_to_device(optimizer, device)
        before, trace, start_step = saved["before"], saved["trace"], saved["step"]
        peak_gradients = saved["peak_gradients"]
        torch.set_rng_state(saved["torch_rng"]); torch.cuda.set_rng_state_all(saved["cuda_rng"])
    domain_sources = {}
    for benchmark in BENCHMARKS:
        grouped = {}
        for sample in data["train"][benchmark]:
            grouped.setdefault(sample.inputs.state.source_id, []).append(sample)
        domain_sources[benchmark] = grouped
    scheduled = None
    schedule_cursor = {benchmark: 0 for benchmark in BENCHMARKS}
    if config["domain_sampling"] == "uniform_domain_then_source_cycle":
        draw_counts = {benchmark: 0 for benchmark in BENCHMARKS}
        for step in range(config["steps"]):
            for slot in range(config["gradient_accumulation"]):
                benchmark = BENCHMARKS[
                    (step * config["gradient_accumulation"] + slot) % len(BENCHMARKS)
                ]
                draw_counts[benchmark] += 1
        scheduled = {
            benchmark: source_cycle_samples(
                domain_sources[benchmark], draws=draw_counts[benchmark], seed=seed,
                namespace=f"utility-development-correction-v1:{benchmark}",
            )
            for benchmark in BENCHMARKS
        }
        for previous_step in range(start_step):
            for slot in range(config["gradient_accumulation"]):
                benchmark = BENCHMARKS[
                    (previous_step * config["gradient_accumulation"] + slot)
                    % len(BENCHMARKS)
                ]
                schedule_cursor[benchmark] += 1
    for step in range(start_step, config["steps"]):
        model.train(); optimizer.zero_grad(set_to_none=True)
        losses=[]; step_tick=time.monotonic()
        rng=random.Random(seed*100000+step)
        for slot in range(config["gradient_accumulation"]):
            benchmark=BENCHMARKS[(step*config["gradient_accumulation"]+slot)%len(BENCHMARKS)]
            if scheduled is None:
                source_ids=sorted(domain_sources[benchmark])
                source=source_ids[rng.randrange(len(source_ids))]
                source_states=domain_sources[benchmark][source]
                sample=source_states[rng.randrange(len(source_states))]
            else:
                sample = scheduled[benchmark][schedule_cursor[benchmark]]
                schedule_cursor[benchmark] += 1
            out=model(sample.inputs)
            loss=utility_sft_loss(out["action_logits"], **supervision_kwargs(
                sample, method=config["method"], temperature=config["temperature"], device=device
            ))
            if not torch.isfinite(loss): raise FloatingPointError("nonfinite loss")
            (loss/config["gradient_accumulation"]).backward(); losses.append(float(loss.detach()))
        gradients=model.gradient_report()
        for group,value in gradients.items(): peak_gradients[group]=max(peak_gradients[group],value)
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        row={"step":step+1,"loss":mean(losses),"gradient_norm":float(norm),
             "seconds":time.monotonic()-step_tick,"gradients":gradients}
        trace.append(row); print(json.dumps(row),flush=True)
        if (step+1)%config["checkpoint_interval"]==0 or step+1==config["steps"]:
            _atomic_torch_save({"provenance":provenance,"parameters":model.trainable_state_dict(),
                "optimizer":optimizer.state_dict(),"step":step+1,"trace":trace,"before":before,
                "peak_gradients":peak_gradients,"torch_rng":torch.get_rng_state(),
                "cuda_rng":torch.cuda.get_rng_state_all()}, run/"resume.pt")
    validation=[]
    model.eval()
    with torch.no_grad():
        for benchmark in BENCHMARKS:
            for sample in data["validation"][benchmark]:
                out=model(sample.inputs)
                predicted=out["predicted_gain"][0].float().cpu().tolist()
                validation.append({"benchmark":benchmark,"state_id":sample.inputs.state.state_id,
                    "source_id":sample.inputs.state.source_id,"gains":list(sample.gains),
                    "predicted_gain":predicted,"action_logits":out["action_logits"][0].float().cpu().tolist(),
                    "best_action":sample.best_action,"support_action":sample.support_action,
                    "measurement":dict(model.last_measurement)})
    _atomic_torch_save({"provenance":provenance,"parameters":model.trainable_state_dict()},run/"selector.pt")
    changed=[n for n,p in model.named_parameters() if p.requires_grad and tensor_hash(p)!=before[n]]
    schedule_audit = {}
    if scheduled is not None:
        for benchmark, samples in scheduled.items():
            schedule_audit[benchmark] = {
                "draws": len(samples),
                "unique_sources": len({s.inputs.state.source_id for s in samples}),
                "unique_states": len({s.inputs.state.state_id for s in samples}),
                "positive_gain_draws": sum(max(s.gains) > 0 for s in samples),
                "harmful_gain_draws": sum(min(s.gains) < 0 for s in samples),
            }
    report_schema = (
        "utility_sft_development_correction_v1"
        if config["scope"] == "three_domain_development_correction"
        else "utility_sft_development_pilot_v1"
    )
    report={"schema":report_schema,"formal_claim_eligible":False,
        "test_accessed":False,"method":config["method"],"provenance":provenance,
        "train":{"states":{b:len(data['train'][b]) for b in BENCHMARKS},
                 "sources":{b:len({s.inputs.state.source_id for s in data['train'][b]}) for b in BENCHMARKS}},
        "validation":{"states":{b:len(data['validation'][b]) for b in BENCHMARKS},
                      "sources":{b:len({s.inputs.state.source_id for s in data['validation'][b]}) for b in BENCHMARKS},
                      "predictions":validation},
        "training_schedule_audit": schedule_audit,
        "trace":trace,"peak_gradients":peak_gradients,"changed_parameter_names":changed,
        "selector_sha256":sha256_file(run/"selector.pt"),"elapsed_seconds":time.monotonic()-tick,
        "peak_gpu_bytes":torch.cuda.max_memory_allocated(),"gpu":torch.cuda.get_device_name(),
        "job_id":os.environ["SLURM_JOB_ID"],
        "versions":{p:importlib.metadata.version(p) for p in ("torch","transformers","Pillow")}}
    atomic_json_write_exclusive(run/"report.json",report)
    print(json.dumps({"report":str(run/"report.json"),"selector_sha256":report["selector_sha256"],
                      "validation_states":len(validation)}),flush=True)


if __name__ == "__main__": main()
