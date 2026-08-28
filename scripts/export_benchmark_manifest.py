from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from beyond_entropy.manifest_export import (
    BENCHMARK_SPECS,
    benchmark_source_group,
    benchmark_stratum,
    export_benchmark_manifest,
    hash_ranked_source_group_indices,
    stratified_sample_indices,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a benchmark slice as JSONL")
    parser.add_argument("--task", choices=sorted(BENCHMARK_SPECS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument("--count", type=int, default=64)
    selection_group.add_argument("--source-group-count", type=int)
    parser.add_argument("--source-group-offset", type=int, default=0)
    parser.add_argument(
        "--selection-namespace",
        default="beyond-entropy-source-selection-v1",
    )
    parser.add_argument(
        "--exclude-source-group",
        action="append",
        default=[],
        help="skip a source group after the rank offset and deterministically backfill",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dataset-revision")
    parser.add_argument("--cache-dir", type=Path)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--arrow-file",
        type=Path,
        help="load a previously materialized datasets Arrow split without Hub access",
    )
    input_group.add_argument(
        "--parquet-file",
        type=Path,
        action="append",
        help=(
            "load one or more pinned local Parquet shards without Hub access; "
            "repeat the flag in canonical shard order"
        ),
    )
    args = parser.parse_args()

    try:
        from datasets import Dataset, load_dataset  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SystemExit(
            "Install the benchmark dependency with: pip install -e '.[benchmark]'"
        ) from exc

    spec = BENCHMARK_SPECS[args.task]
    revision = args.dataset_revision or spec.default_revision
    if args.arrow_file:
        dataset = Dataset.from_file(str(args.arrow_file.resolve()))
    elif args.parquet_file:
        parquet_files = [path.resolve() for path in args.parquet_file]
        missing = [path for path in parquet_files if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"local Parquet shard does not exist: {missing[0]}")
        dataset = Dataset.from_parquet([str(path) for path in parquet_files])
    else:
        dataset_args = [spec.dataset_id]
        if spec.dataset_name is not None:
            dataset_args.append(spec.dataset_name)
        dataset = load_dataset(
            *dataset_args,
            split=spec.split,
            revision=revision,
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
        )
    if args.source_group_count is None:
        selection_columns = {
            field: dataset[field] for field in spec.selection_fields
        }
        labels = [
            benchmark_stratum(
                {
                    field: selection_columns[field][index]
                    for field in spec.selection_fields
                },
                task=args.task,
            )
            for index in range(len(dataset))
        ]
        source_indices = stratified_sample_indices(
            labels,
            count=args.count,
            seed=args.seed,
        )
        selection = "seeded round-robin stratified row sample"
        selection_metadata = {"row_count": args.count}
    else:
        if not spec.source_fields:
            raise SystemExit(f"task {args.task} has no source-group selection fields")
        source_columns = {field: dataset[field] for field in spec.source_fields}
        group_ids = [
            benchmark_source_group(
                {field: source_columns[field][index] for field in spec.source_fields},
                task=args.task,
            )
            for index in range(len(dataset))
        ]
        source_indices = hash_ranked_source_group_indices(
            group_ids,
            count=args.source_group_count,
            offset=args.source_group_offset,
            seed=args.seed,
            namespace=args.selection_namespace,
            excluded_groups=args.exclude_source_group,
        )
        selected_group_ids = sorted({group_ids[index] for index in source_indices})
        selection = "SHA-256-ranked whole-source-group slice"
        selection_metadata = {
            "namespace": args.selection_namespace,
            "seed": args.seed,
            "source_group_count": args.source_group_count,
            "source_group_offset": args.source_group_offset,
            "excluded_source_groups": sorted(set(args.exclude_source_group)),
            "source_group_ids": selected_group_ids,
        }
    rows = [dataset[index] for index in source_indices]
    result = export_benchmark_manifest(
        rows,
        source_indices=source_indices,
        task=args.task,
        dataset_id=spec.dataset_id,
        dataset_revision=revision,
        output_dir=args.output_dir,
        seed=args.seed,
        selection=selection,
        selection_metadata=selection_metadata,
    )
    local_sources = []
    if args.arrow_file:
        local_sources = [args.arrow_file.resolve()]
        result["source_arrow"] = str(local_sources[0])
    elif args.parquet_file:
        local_sources = [path.resolve() for path in args.parquet_file]
        result["source_parquet_files"] = [str(path) for path in local_sources]
    if local_sources:
        source_hashes = []
        for path in local_sources:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            source_hashes.append(digest.hexdigest())
        if args.arrow_file:
            result["source_arrow_sha256"] = source_hashes[0]
        else:
            result["source_parquet_sha256"] = source_hashes
        provenance_path = args.output_dir / "manifest.provenance.json"
        provenance_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
