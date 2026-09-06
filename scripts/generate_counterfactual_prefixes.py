#!/usr/bin/env python3
"""Generate real Qwen paired STOP/CONTINUE rollouts from fixed visual prefixes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from beyond_entropy.benchmarks import load_manifest, scorer_by_name
from beyond_entropy.qwen_backend import Qwen25VLBackend
from beyond_entropy.sequential_rollout import (
    FixedOppositeUGPrefix,
    collect_counterfactual_prefixes,
)
from beyond_entropy.sequential_schema import SequentialRolloutRecord
from beyond_entropy.sequential_metrics import sequential_diagnostic
from beyond_entropy.sharding import SHARD_ALGORITHM, stable_shard_index


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def read_existing(path: Path) -> list[SequentialRolloutRecord]:
    if not path.exists():
        return []
    result = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                result.append(SequentialRolloutRecord.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"invalid checkpoint row {path}:{line_number}") from exc
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--benchmark", choices=("chartqa", "docvqa", "hrbench"), required=True)
    parser.add_argument("--dataset-role", choices=("train", "validation"), required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention-implementation", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=768 * 28 * 28)
    parser.add_argument("--generation-seed", type=int, action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--checkpoint-interval", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--features-output")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("real sequential rollout generation must run under Slurm")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    if args.checkpoint_interval <= 0:
        raise ValueError("checkpoint interval must be positive")
    seeds = tuple(args.generation_seed or [0])
    if len(set(seeds)) != len(seeds):
        raise ValueError("generation seeds must be unique")

    root = Path(__file__).resolve().parents[1]
    manifest = Path(args.manifest).resolve()
    output = Path(args.output).resolve()
    provenance_path = output.with_suffix(output.suffix + ".provenance.json")
    completion_path = output.with_suffix(output.suffix + ".complete.json")
    examples = load_manifest(manifest, limit=args.limit)
    examples = [
        item
        for item in examples
        if stable_shard_index(
            item.state.state_id,
            args.shard_count,
            namespace="sequential-prefix-v1",
        )
        == args.shard_index
    ]
    if not examples:
        raise ValueError("selected sequential shard is empty")
    provenance = {
        "schema": "sequential_prefix_rollout_provenance_v1",
        "scientific_status": "development_only_test_unopened",
        "test_accessed": False,
        "dataset_role": args.dataset_role,
        "benchmark": args.benchmark,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "manifest_limit_before_sharding": args.limit,
        "states": len(examples),
        "model": args.model,
        "model_revision": args.revision,
        "dtype": args.dtype,
        "attention_implementation": args.attention_implementation,
        "max_new_tokens": args.max_new_tokens,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "generation_seeds": list(seeds),
        "proposer": "sequential-opposite-ug-v1",
        "candidate_count": 4,
        "visual_crop_ratio": 2.0,
        "visual_cost_per_crop": 1.0,
        "shard_algorithm": SHARD_ALGORITHM,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "code_revision": git_revision(root),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
    }
    if completion_path.exists():
        raise FileExistsError(f"completed run cannot be reopened: {completion_path}")
    if provenance_path.exists():
        if not args.resume or json.loads(provenance_path.read_text()) != provenance:
            raise ValueError("resume provenance differs from current request")
    else:
        if output.exists():
            raise FileExistsError("rollout file exists without provenance")
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        with provenance_path.open("x", encoding="utf-8") as handle:
            json.dump(provenance, handle, indent=2, sort_keys=True)
            handle.write("\n")

    existing = read_existing(output)
    completed = {item.decision_id for item in existing}
    expected = {
        (item.state.state_id, f"replicate-{index:03d}")
        for item in examples
        for index in range(len(seeds))
    }
    if completed - expected:
        raise ValueError("checkpoint contains decisions outside the frozen shard")
    pending = [
        item
        for item in examples
        if any(
            (item.state.state_id, f"replicate-{index:03d}") not in completed
            for index in range(len(seeds))
        )
    ]
    backend = Qwen25VLBackend(
        args.model,
        revision=args.revision,
        device_map=args.device_map,
        dtype=args.dtype,
        attention_implementation=args.attention_implementation,
        max_new_tokens=args.max_new_tokens,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        local_files_only=True,
    )
    proposer = FixedOppositeUGPrefix()
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if output.exists() else "x"
    with output.open(mode, encoding="utf-8") as handle:
        for position, example in enumerate(pending, start=1):
            rows = collect_counterfactual_prefixes(
                (example,),
                prefixes=proposer,
                backend=backend,
                scorer=scorer_by_name(args.benchmark),
                generation_seeds=seeds,
            )
            for row in rows:
                if row.decision_id not in completed:
                    handle.write(json.dumps(row.to_dict(), allow_nan=False, sort_keys=True) + "\n")
                    completed.add(row.decision_id)
            if position % args.checkpoint_interval == 0:
                handle.flush()
                os.fsync(handle.fileno())
                print(json.dumps({"completed": len(completed), "expected": len(expected)}), flush=True)
    if completed != expected:
        raise RuntimeError("sequential rollout checkpoint coverage is incomplete")
    all_records = read_existing(output)

    feature_summary = None
    if args.features_output:
        # Release the generation backend before loading the frozen feature extractor.
        del backend
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        from beyond_entropy.acquisition_critic import extract_sequential_feature_dataset

        states = {item.state.state_id: item.state for item in examples}
        feature_summary = extract_sequential_feature_dataset(
            records=all_records,
            manifest_states=states,
            output_path=args.features_output,
            dataset_role=args.dataset_role,
            benchmark=args.benchmark,
            model_name_or_path=args.model,
            revision=args.revision,
            device_map=args.device_map,
            dtype=args.dtype,
            attention_implementation=args.attention_implementation,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            checkpoint_interval=args.checkpoint_interval,
        )
    completion = {
        **provenance,
        "schema": "sequential_prefix_rollout_completion_v1",
        "completed": True,
        "record_count": len(all_records),
        "rollouts_sha256": sha256_file(output),
        "headroom_diagnostic": sequential_diagnostic(all_records),
        "runtime_measurement": getattr(backend, "runtime_measurement", lambda: None)()
        if not args.features_output
        else None,
        "features": (
            None
            if not args.features_output
            else {
                "path": str(Path(args.features_output).resolve()),
                "sha256": feature_summary["output_sha256"],
                "rows": len(feature_summary["rows"]),
            }
        ),
    }
    with completion_path.open("x", encoding="utf-8") as handle:
        json.dump(completion, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"completion": str(completion_path), "records": len(all_records)}), flush=True)


if __name__ == "__main__":
    main()
