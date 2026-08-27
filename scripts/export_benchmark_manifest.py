from __future__ import annotations

import argparse
import json
from pathlib import Path

from beyond_entropy.manifest_export import (
    BENCHMARK_SPECS,
    export_benchmark_manifest,
    stratified_sample_indices,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a benchmark slice as JSONL")
    parser.add_argument("--task", choices=sorted(BENCHMARK_SPECS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dataset-revision")
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Install the benchmark dependency with: pip install -e '.[benchmark]'"
        ) from exc

    spec = BENCHMARK_SPECS[args.task]
    revision = args.dataset_revision or spec.default_revision
    dataset = load_dataset(
        spec.dataset_id,
        split=spec.split,
        revision=revision,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
    )
    labels = [str(label) for label in dataset[spec.group_field]]
    source_indices = stratified_sample_indices(
        labels,
        count=args.count,
        seed=args.seed,
    )
    rows = [dataset[index] for index in source_indices]
    result = export_benchmark_manifest(
        rows,
        source_indices=source_indices,
        task=args.task,
        dataset_id=spec.dataset_id,
        dataset_revision=revision,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
