from __future__ import annotations

import argparse
import hashlib
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
    parser.add_argument(
        "--arrow-file",
        type=Path,
        help="load a previously materialized datasets Arrow split without Hub access",
    )
    args = parser.parse_args()

    try:
        from datasets import Dataset, load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Install the benchmark dependency with: pip install -e '.[benchmark]'"
        ) from exc

    spec = BENCHMARK_SPECS[args.task]
    revision = args.dataset_revision or spec.default_revision
    if args.arrow_file:
        dataset = Dataset.from_file(str(args.arrow_file.resolve()))
    else:
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
    if args.arrow_file:
        digest = hashlib.sha256()
        with args.arrow_file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result["source_arrow"] = str(args.arrow_file.resolve())
        result["source_arrow_sha256"] = digest.hexdigest()
        provenance_path = args.output_dir / "manifest.provenance.json"
        provenance_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
