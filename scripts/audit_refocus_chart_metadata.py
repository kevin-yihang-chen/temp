#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from beyond_entropy.refocus_chart_audit import audit_split_rows


SELECTED_COLUMNS = (
    "id",
    "source",
    "split",
    "data_source",
    "ability",
    "agent_name",
    "prompt",
    "reward_model",
    "extra_info",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit pinned Refocus_Chart metadata without reading image columns."
    )
    parser.add_argument("--dataset", default="VTOOL/Refocus_Chart")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--train-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_split(
    dataset: str, revision: str, split: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import fsspec  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    url = (
        f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/{split}.parquet"
    )
    with fsspec.open(url, "rb", block_size=2**20, cache_type="readahead") as stream:
        parquet = pq.ParquetFile(stream)
        selected_compressed_bytes = 0
        for row_group_index in range(parquet.metadata.num_row_groups):
            row_group = parquet.metadata.row_group(row_group_index)
            for column_index in range(row_group.num_columns):
                column = row_group.column(column_index)
                top_level = column.path_in_schema.split(".", 1)[0]
                if top_level in SELECTED_COLUMNS:
                    selected_compressed_bytes += column.total_compressed_size
        table = parquet.read(columns=list(SELECTED_COLUMNS), use_threads=True)
        metadata = {
            "url": url,
            "rows": parquet.metadata.num_rows,
            "row_groups": parquet.metadata.num_row_groups,
            "leaf_columns": parquet.metadata.num_columns,
            "created_by": parquet.metadata.created_by,
            "selected_top_level_columns": list(SELECTED_COLUMNS),
            "explicitly_excluded_top_level_columns": [
                "images",
                "edited_image",
                "thoughts",
                "focus_areas_bbox",
                "x_values",
                "y_values",
                "x_values_bbox",
                "y_values_bbox",
                "figure_bbox",
            ],
            "selected_column_compressed_bytes": selected_compressed_bytes,
        }
    return table.to_pylist(), metadata


def main() -> None:
    args = parse_args()
    train_rows, train_parquet = _read_split(args.dataset, args.revision, "train")
    train = audit_split_rows(
        train_rows,
        split="train",
        dataset_revision=args.revision,
        parquet_sha256=args.train_sha256,
    )
    report = {
        "schema": "refocus_chart_train_metadata_report_v1",
        "dataset": args.dataset,
        "revision": args.revision,
        "license_declared": None,
        "license_observation_source": (
            "separate pinned repository API audit; not inferred from Parquet"
        ),
        "train_parquet": train_parquet,
        "train": train,
        "test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "output": str(args.output),
        "train_rows": train["rows"],
        "train_manifest_sha256": train["manifest_sha256"],
        "test_accessed": False,
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
