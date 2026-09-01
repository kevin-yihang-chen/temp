#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any, Callable

from beyond_entropy.infographicvqa_decar import (
    DECAR_ACTION_IDS,
    DECAR_SCALAR_NAMES,
    fit_when,
    fit_where,
)


SCHEMA = "infographicvqa_decar_full_shape_fit_benchmark_v1"
REGISTERED_EPOCHS = 200
FIT_COUNTS = {
    "where_inner": 40,
    "where_outer": 10,
    "when_ternary_outer": 10,
    "when_binary_outer": 5,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the frozen DECAR fit schedule on synthetic tensors."
    )
    parser.add_argument("--decisions", type=int, default=23_946)
    parser.add_argument("--sources", type=int, default=2_204)
    parser.add_argument("--embedding-dim", type=int, default=3_584)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20_260_917)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _synchronize(torch: Any, device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def _timed_fit(
    torch: Any,
    device: str,
    fit: Callable[[], Any],
) -> tuple[float, int]:
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(torch, device)
    start = time.monotonic()
    result = fit()
    _synchronize(torch, device)
    seconds = time.monotonic() - start
    peak = (
        int(torch.cuda.max_memory_allocated(device)) if device.startswith("cuda") else 0
    )
    del result
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return seconds, peak


def main() -> None:
    args = _parser().parse_args()
    if (
        args.decisions < 20
        or args.sources < 5
        or args.sources > args.decisions
        or args.embedding_dim <= 0
        or args.epochs <= 0
        or args.epochs > REGISTERED_EPOCHS
        or args.output.exists()
    ):
        raise ValueError("invalid or overwriting DECAR benchmark configuration")

    import torch  # type: ignore[import-not-found]

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("DECAR benchmark requested CUDA but CUDA is unavailable")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    outer_rows = math.ceil(0.8 * args.decisions)
    inner_rows = math.ceil(0.6 * args.decisions)
    candidates = len(DECAR_ACTION_IDS)
    scalar_dim = len(DECAR_SCALAR_NAMES)

    allocation_start = time.monotonic()
    question = torch.randn(
        outer_rows, args.embedding_dim, generator=generator, dtype=torch.float32
    )
    global_visual = torch.randn(
        outer_rows, args.embedding_dim, generator=generator, dtype=torch.float32
    )
    region = torch.randn(
        outer_rows,
        candidates,
        args.embedding_dim,
        generator=generator,
        dtype=torch.float32,
    )
    scalars = torch.randn(
        outer_rows,
        candidates,
        scalar_dim,
        generator=generator,
        dtype=torch.float32,
    )
    target_base = torch.linspace(-0.3, 0.3, outer_rows, dtype=torch.float32)
    targets = torch.stack(
        tuple(target_base + 0.05 * index for index in range(candidates)),
        dim=1,
    )
    selected_deltas = torch.tensor((-0.2, 0.0, 0.2), dtype=torch.float32).repeat(
        math.ceil(outer_rows / 3)
    )[:outer_rows]
    predicted_gaps = torch.linspace(-1.0, 1.0, outer_rows, dtype=torch.float32)
    predicted_margins = torch.linspace(0.0, 0.5, outer_rows, dtype=torch.float32)
    source_ids = tuple(
        f"synthetic-source-{index % args.sources:05d}" for index in range(outer_rows)
    )
    allocation_seconds = time.monotonic() - allocation_start

    timings: dict[str, float] = {}
    peaks: dict[str, int] = {}

    timings["where_inner"], peaks["where_inner"] = _timed_fit(
        torch,
        args.device,
        lambda: fit_where(
            question[:inner_rows],
            global_visual[:inner_rows],
            region[:inner_rows],
            scalars[:inner_rows],
            targets[:inner_rows],
            source_ids[:inner_rows],
            seed=args.seed,
            device=args.device,
            epochs=args.epochs,
        ),
    )
    timings["where_outer"], peaks["where_outer"] = _timed_fit(
        torch,
        args.device,
        lambda: fit_where(
            question,
            global_visual,
            region,
            scalars,
            targets,
            source_ids,
            seed=args.seed + 1,
            device=args.device,
            epochs=args.epochs,
        ),
    )

    when_common = {
        "question": question,
        "global_visual": global_visual,
        "selected_region": region[:, 0, :],
        "selected_scalars": scalars[:, 0, :],
        "predicted_gaps": predicted_gaps,
        "predicted_margins": predicted_margins,
        "selected_deltas": selected_deltas,
        "source_ids": source_ids,
        "device": args.device,
        "epochs": args.epochs,
    }
    timings["when_ternary_outer"], peaks["when_ternary_outer"] = _timed_fit(
        torch,
        args.device,
        lambda: fit_when(seed=args.seed + 2, binary=False, **when_common),
    )
    timings["when_binary_outer"], peaks["when_binary_outer"] = _timed_fit(
        torch,
        args.device,
        lambda: fit_when(seed=args.seed + 3, binary=True, **when_common),
    )

    epoch_scale = REGISTERED_EPOCHS / args.epochs
    projected_fit_seconds = epoch_scale * sum(
        FIT_COUNTS[name] * timings[name] for name in FIT_COUNTS
    )
    projected_with_reserve = 1.25 * projected_fit_seconds
    accelerator = (
        torch.cuda.get_device_name(args.device)
        if args.device.startswith("cuda")
        else platform.processor() or "cpu"
    )
    report = {
        "schema": SCHEMA,
        "scientific_status": "synthetic runtime-only benchmark; no task endpoint",
        "configuration": {
            "decisions": args.decisions,
            "sources": args.sources,
            "embedding_dim": args.embedding_dim,
            "candidates": candidates,
            "scalar_dim": scalar_dim,
            "benchmark_epochs": args.epochs,
            "registered_epochs": REGISTERED_EPOCHS,
            "seed": args.seed,
            "device": args.device,
            "outer_rows": outer_rows,
            "inner_rows": inner_rows,
            "fit_counts": FIT_COUNTS,
        },
        "runtime": {
            "accelerator": accelerator,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "allocation_seconds": allocation_seconds,
            "measured_fit_seconds": timings,
            "peak_allocated_bytes": peaks,
            "projected_registered_fit_seconds": projected_fit_seconds,
            "projected_registered_fit_hours": projected_fit_seconds / 3600.0,
            "projected_with_25pct_reserve_seconds": projected_with_reserve,
            "projected_with_25pct_reserve_hours": projected_with_reserve / 3600.0,
        },
        "contracts": {
            "synthetic_inputs_only": True,
            "task_outcomes_read": False,
            "scientific_endpoints_computed": False,
            "validation_or_test_inputs_used": False,
            "credentials_present": bool(
                os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
