#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from beyond_entropy.refocus_chart_audit import (
    canonical_sha256,
    sha256_file,
    structural_chart_signature,
    verify_local_pinned_shard,
)
from beyond_entropy.refocus_g1_dataset import (
    ACTION_SYSTEM_PROMPT_V1,
    AGENT_NAME,
    CONVERTER_SCHEMA,
    GROUP_SPLIT_SEED,
    build_official_tool_metadata,
    convert_official_train_row,
    select_structural_groups,
)


METADATA_COLUMNS = (
    "id",
    "question",
    "answer",
    "source",
    "split",
    "x_values",
    "y_values",
    "x_values_bbox",
    "y_values_bbox",
    "figure_bbox",
)
IMAGE_COLUMN = "image"


@dataclass(frozen=True)
class LocatedRow:
    shard_index: int
    row_group_index: int
    row_offset: int
    global_index: int
    row: dict[str, Any]
    structural_sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert only pinned official ReFocus train rows into a leakage-safe "
            "VTool/verl G1 dataset."
        )
    )
    parser.add_argument("--official-pin", type=Path, required=True)
    parser.add_argument("--official-shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-train-groups", type=int, default=32)
    parser.add_argument("--max-curve-eval-groups", type=int, default=16)
    parser.add_argument(
        "--agent-name",
        choices=(AGENT_NAME, "vtool_agent"),
        default=AGENT_NAME,
        help="Paired proposed/control agent or upstream outcome-only agent.",
    )
    parser.add_argument(
        "--smoke-one-row",
        action="store_true",
        help=(
            "Write one deterministically selected official-train row for an "
            "engineering-only processor smoke; no scientific result may use it."
        ),
    )
    return parser.parse_args()


def _load_pin(path: Path) -> Mapping[str, Any]:
    pin = json.loads(path.read_text(encoding="utf-8"))
    if pin.get("schema") != "refocus_official_train_pin_v1":
        raise ValueError("official train pin schema mismatch")
    if pin.get("dataset") != "ReFocus/ReFocus_Data":
        raise ValueError("official train dataset identity mismatch")
    if pin.get("license") != "apache-2.0":
        raise ValueError("official train license must be apache-2.0")
    revision = str(pin.get("revision", ""))
    if len(revision) != 40:
        raise ValueError("official train revision must be a pinned commit")
    shards = pin.get("train_shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("official train pin must contain train shards")
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise ValueError("official train shard entries must be mappings")
        path_value = str(shard.get("path", ""))
        if not path_value.startswith("data/train-") or not path_value.endswith(
            ".parquet"
        ):
            raise ValueError("converter accepts only pinned data/train shards")
    return pin


def _scan_train_metadata(
    shard_paths: list[Path],
) -> tuple[list[LocatedRow], list[dict[str, Any]]]:
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    located: list[LocatedRow] = []
    shard_reports: list[dict[str, Any]] = []
    global_index = 0
    for shard_index, path in enumerate(shard_paths):
        parquet = pq.ParquetFile(path)
        shard_rows = 0
        for row_group_index in range(parquet.metadata.num_row_groups):
            table = parquet.read_row_group(
                row_group_index,
                columns=list(METADATA_COLUMNS),
                use_threads=False,
            )
            rows = table.to_pylist()
            for row_offset, row in enumerate(rows):
                metadata = build_official_tool_metadata(row)
                located.append(
                    LocatedRow(
                        shard_index=shard_index,
                        row_group_index=row_group_index,
                        row_offset=row_offset,
                        global_index=global_index,
                        row=row,
                        structural_sha256=structural_chart_signature(metadata),
                    )
                )
                global_index += 1
                shard_rows += 1
        shard_reports.append(
            {
                "path": str(path),
                "rows": shard_rows,
                "row_groups": parquet.metadata.num_row_groups,
                "metadata_columns_read": list(METADATA_COLUMNS),
                "image_column_read_during_metadata_scan": False,
            }
        )
    return located, shard_reports


def _load_selected_original_images(
    located_rows: list[LocatedRow], shard_paths: list[Path]
) -> dict[int, dict[str, Any]]:
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    by_location: dict[tuple[int, int], list[LocatedRow]] = defaultdict(list)
    for located in located_rows:
        by_location[(located.shard_index, located.row_group_index)].append(located)

    rows_with_images: dict[int, dict[str, Any]] = {}
    opened: dict[int, Any] = {}
    for (shard_index, row_group_index), selections in sorted(by_location.items()):
        parquet = opened.setdefault(
            shard_index, pq.ParquetFile(shard_paths[shard_index])
        )
        image_rows = parquet.read_row_group(
            row_group_index,
            columns=[IMAGE_COLUMN],
            use_threads=False,
        ).to_pylist()
        for located in selections:
            merged = dict(located.row)
            merged[IMAGE_COLUMN] = image_rows[located.row_offset][IMAGE_COLUMN]
            rows_with_images[located.global_index] = merged
    if len(rows_with_images) != len(located_rows):
        raise AssertionError("selected original-image coverage mismatch")
    return rows_with_images


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    if not rows:
        raise ValueError(f"cannot write empty split: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="zstd", use_dictionary=True)
    return {
        "path": str(path),
        "rows": len(rows),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "schema_sha256": canonical_sha256(str(table.schema)),
        "row_id_manifest_sha256": canonical_sha256(
            sorted(str(row["id"]) for row in rows)
        ),
        "structural_group_manifest_sha256": canonical_sha256(
            sorted({str(row["extra_info"]["structural_chart_sha256"]) for row in rows})
        ),
    }


def main() -> None:
    args = parse_args()
    pin = _load_pin(args.official_pin)
    shard_paths: list[Path] = []
    verified_shards: list[dict[str, Any]] = []
    for shard in pin["train_shards"]:
        path, digest = verify_local_pinned_shard(args.official_shard_root, shard)
        shard_paths.append(path)
        verified_shards.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    located, scan_reports = _scan_train_metadata(shard_paths)
    if not located:
        raise ValueError("official train scan returned no rows")

    if args.smoke_one_row:
        selected = [
            min(
                located,
                key=lambda item: canonical_sha256(
                    [GROUP_SPLIT_SEED, item.row["id"], "one-row-smoke"]
                ),
            )
        ]
        split_by_group = {selected[0].structural_sha256: "g1_smoke"}
        selection_summary: dict[str, Any] = {
            "mode": "engineering_only_one_row_smoke",
            "max_train_groups": 0,
            "max_curve_eval_groups": 0,
        }
    else:
        selection = select_structural_groups(
            (item.row for item in located),
            max_train_groups=args.max_train_groups,
            max_curve_eval_groups=args.max_curve_eval_groups,
        )
        split_by_group = selection.group_to_split
        selected = [
            item for item in located if item.structural_sha256 in split_by_group
        ]
        selection_summary = {
            "mode": "frozen_structural_group_split",
            "max_train_groups": args.max_train_groups,
            "max_curve_eval_groups": args.max_curve_eval_groups,
            "all_train_groups": selection.all_train_groups,
            "all_curve_eval_groups": selection.all_curve_eval_groups,
            "selected_train_groups": selection.selected_train_groups,
            "selected_curve_eval_groups": selection.selected_curve_eval_groups,
        }
    if not selected:
        raise ValueError("group selection returned no rows")

    originals = _load_selected_original_images(selected, shard_paths)
    converted_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for located_row in selected:
        split = split_by_group[located_row.structural_sha256]
        converted_by_split[split].append(
            convert_official_train_row(
                originals[located_row.global_index],
                index=located_row.global_index,
                development_split=split,
                agent_name=args.agent_name,
            )
        )
    for rows in converted_by_split.values():
        rows.sort(key=lambda row: int(row["extra_info"]["index"]))

    output_reports: dict[str, Any] = {}
    for split, rows in sorted(converted_by_split.items()):
        filename = f"{split}.parquet"
        output_reports[split] = _write_parquet(rows, args.output_dir / filename)

    train_groups = {
        row["extra_info"]["structural_chart_sha256"]
        for row in converted_by_split.get("g1_train", [])
    }
    curve_groups = {
        row["extra_info"]["structural_chart_sha256"]
        for row in converted_by_split.get("g1_curve_eval", [])
    }
    overlap = train_groups & curve_groups
    if overlap:
        raise AssertionError("structural chart groups cross train/curve-eval")

    report = {
        "schema": CONVERTER_SCHEMA,
        "decision": "refocus_official_g1_converter_passed",
        "official_dataset": pin["dataset"],
        "official_revision": pin["revision"],
        "official_license": pin["license"],
        "source_split": "train",
        "protected_split_contents_accessed": False,
        "group_split_seed": GROUP_SPLIT_SEED,
        "agent_name": args.agent_name,
        "system_prompt_sha256": canonical_sha256(ACTION_SYSTEM_PROMPT_V1),
        "all_official_train_rows": len(located),
        "all_official_structural_groups": len(
            {item.structural_sha256 for item in located}
        ),
        "selected_rows": len(selected),
        "selected_structural_groups": len(split_by_group),
        "train_curve_eval_structural_overlap": len(overlap),
        "policy_inputs_exclude": [
            "answer",
            "edited_image",
            "focus_areas_bbox",
            "thoughts",
        ],
        "reward_target_only": "answer",
        "verified_train_shards": verified_shards,
        "metadata_scans": scan_reports,
        "selection": selection_summary,
        "outputs": output_reports,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "report": str(args.report),
                "selected_rows": report["selected_rows"],
                "outputs": output_reports,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"decision": "converter_failed", "error": str(exc)}))
        raise
