#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from beyond_entropy.infographicvqa_decar_manifest import (
    FULL_IDENTITY_FIELDS,
    materialize_full_task_row,
    validate_decar_fold_manifests,
)


EXPECTED_QUESTIONS = 23_946
EXPECTED_IMAGES = 4_406
EXPECTED_SOURCES = 2_204
EXPECTED_INNER_ROWS = 8_816
EXPECTED_DOWNLOAD_REVISION = "539088ef8a8ada01ac8e2e6d4e372586748a265e"
EXPECTED_DOWNLOAD_FILES = 24
EXPECTED_DOWNLOAD_BYTES = 1_981_251_656


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked(path: Path, expected: str, name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or _sha256(resolved) != expected:
        raise ValueError(f"InfographicVQA DECAR full {name} SHA-256 mismatch")
    return resolved


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSON at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _verify_download(dataset_dir: Path, manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", []) if isinstance(manifest, dict) else []
    if (
        not isinstance(manifest, dict)
        or manifest.get("revision") != EXPECTED_DOWNLOAD_REVISION
        or manifest.get("file_count") != EXPECTED_DOWNLOAD_FILES
        or manifest.get("aggregate_bytes") != EXPECTED_DOWNLOAD_BYTES
        or bool(manifest.get("validation_files_downloaded", True))
        or bool(manifest.get("test_files_downloaded", True))
        or not isinstance(files, list)
        or len(files) != EXPECTED_DOWNLOAD_FILES
    ):
        raise ValueError("InfographicVQA DECAR full download contract changed")
    for row in files:
        path = dataset_dir / str(row["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or _sha256(path) != str(row["sha256"])
        ):
            raise ValueError("InfographicVQA DECAR full parquet hash changed")


def _materialize_rows(
    dataset_dir: Path,
    identities: list[dict[str, Any]],
    images_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import pyarrow.parquet as pq  # type: ignore[import-not-found,import-untyped]

    identities_by_file: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for identity in identities:
        if set(identity) != FULL_IDENTITY_FIELDS:
            raise ValueError("DECAR full identity field inventory changed")
        filename = str(identity["transport_file"])
        row_index = int(identity["transport_row"])
        if row_index in identities_by_file[filename]:
            raise ValueError("DECAR full has duplicate parquet row locators")
        identities_by_file[filename][row_index] = identity

    columns = ["questionId", "question", "answers", "image", "data_split"]
    task_rows: list[dict[str, Any]] = []
    images_by_encoded: dict[str, dict[str, Any]] = {}
    observed_files: set[str] = set()
    for filename, expected_by_row in sorted(identities_by_file.items()):
        parquet_path = dataset_dir / "InfographicVQA" / filename
        parquet = pq.ParquetFile(parquet_path)
        if set(expected_by_row) != set(range(parquet.metadata.num_rows)):
            raise ValueError("DECAR full parquet row coverage is not exact")
        observed_files.add(filename)
        row_start = 0
        for row_group in range(parquet.num_row_groups):
            table = parquet.read_row_group(row_group, columns=columns)
            for offset in range(table.num_rows):
                absolute_row = row_start + offset
                payload = {name: table[name][offset].as_py() for name in columns}
                task, image_row, raw_image = materialize_full_task_row(
                    expected_by_row[absolute_row], payload
                )
                encoded = str(image_row["encoded_sha256"])
                prior = images_by_encoded.get(encoded)
                if prior is None:
                    image_path = images_dir / f"{encoded}.img"
                    with image_path.open("xb") as handle:
                        handle.write(raw_image)
                        handle.flush()
                        os.fsync(handle.fileno())
                    images_by_encoded[encoded] = image_row
                elif prior != image_row:
                    raise RuntimeError("DECAR full repeated-image metadata changed")
                task_rows.append(task)
            row_start += table.num_rows
        if row_start != parquet.metadata.num_rows:
            raise RuntimeError("DECAR full parquet scan differs from footer")
    if len(observed_files) != EXPECTED_DOWNLOAD_FILES:
        raise ValueError("DECAR full did not cover exactly 24 train parquets")
    task_rows.sort(key=lambda row: (str(row["question_id"]), str(row["state_id"])))
    image_rows = [images_by_encoded[key] for key in sorted(images_by_encoded)]
    return task_rows, image_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize the registered full InfographicVQA DECAR train bank"
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--download-manifest", type=Path, required=True)
    parser.add_argument("--expected-download-manifest-sha256", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--outer-folds", type=Path, required=True)
    parser.add_argument("--expected-outer-folds-sha256", required=True)
    parser.add_argument("--inner-folds", type=Path, required=True)
    parser.add_argument("--expected-inner-folds-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--allocation-result", type=Path, required=True)
    parser.add_argument("--expected-allocation-result-sha256", required=True)
    parser.add_argument("--pilot-result", type=Path, required=True)
    parser.add_argument("--expected-pilot-result-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    download_manifest = _checked(
        args.download_manifest,
        args.expected_download_manifest_sha256,
        "download manifest",
    )
    source_manifest = _checked(
        args.source_manifest,
        args.expected_source_manifest_sha256,
        "source manifest",
    )
    outer_folds = _checked(
        args.outer_folds, args.expected_outer_folds_sha256, "outer folds"
    )
    inner_folds = _checked(
        args.inner_folds, args.expected_inner_folds_sha256, "inner folds"
    )
    protocol = _checked(args.protocol, args.expected_protocol_sha256, "protocol")
    allocation_result = _checked(
        args.allocation_result,
        args.expected_allocation_result_sha256,
        "allocation result",
    )
    pilot_result = _checked(
        args.pilot_result, args.expected_pilot_result_sha256, "pilot result"
    )
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite DECAR full bank: {output_dir}")

    _verify_download(dataset_dir, download_manifest)
    identities = _read_jsonl(source_manifest)
    outer_rows = _read_jsonl(outer_folds)
    inner_rows = _read_jsonl(inner_folds)
    fold_audit = validate_decar_fold_manifests(
        identities, outer_rows, inner_rows
    )
    if (
        fold_audit["questions"] != EXPECTED_QUESTIONS
        or fold_audit["images"] != EXPECTED_IMAGES
        or fold_audit["sources"] != EXPECTED_SOURCES
        or fold_audit["inner_rows"] != EXPECTED_INNER_ROWS
    ):
        raise ValueError("DECAR full population or fold contract changed")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=output_dir.name + ".partial-", dir=output_dir.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        images_dir = temporary / "images"
        images_dir.mkdir()
        task_rows, image_rows = _materialize_rows(
            dataset_dir, identities, images_dir
        )
        questions = {str(row["question_id"]) for row in task_rows}
        states = {str(row["state_id"]) for row in task_rows}
        sources = {str(row["source_id"]) for row in task_rows}
        images = {str(row["image_id"]) for row in task_rows}
        if (
            len(task_rows) != EXPECTED_QUESTIONS
            or len(questions) != EXPECTED_QUESTIONS
            or len(states) != EXPECTED_QUESTIONS
            or len(sources) != EXPECTED_SOURCES
            or len(images) != EXPECTED_IMAGES
            or len(image_rows) != EXPECTED_IMAGES
        ):
            raise ValueError("DECAR full materialized population changed")
        forbidden_task_fields = {
            "normalized_hostname",
            "transport_file",
            "transport_row",
            "outer_fold",
            "inner_fold",
            "answer_type",
            "ocr",
            "operation/reasoning",
        }
        if any(forbidden_task_fields & set(row) for row in task_rows):
            raise RuntimeError("DECAR full task manifest contains forbidden fields")

        task_path = temporary / "task-manifest.jsonl"
        image_manifest_path = temporary / "image-manifest.jsonl"
        _write_jsonl(task_path, task_rows)
        _write_jsonl(image_manifest_path, image_rows)
        report = {
            "schema": "infographicvqa_decar_full_materialization_v1",
            "scientific_status": (
                "registered full official-train OOF bank; validation and test sealed"
            ),
            "population": {
                "questions": len(task_rows),
                "sources": len(sources),
                "images": len(image_rows),
                "answer_references": sum(
                    len(row["target"]["answers"]) for row in task_rows
                ),
            },
            "fold_audit": fold_audit,
            "columns_read": columns_for_report(),
            "columns_not_read": [
                "answer_type",
                "image_url",
                "operation/reasoning",
                "ocr",
            ],
            "task_manifest_fields": sorted(task_rows[0]),
            "task_manifest_forbidden_fields_absent": True,
            "audits": {
                "identity_coverage_exact": True,
                "encoded_and_decoded_image_hashes_exact": True,
                "all_split_markers_train": True,
                "source_disjoint_folds_exact": True,
                "fold_ids_not_serialized_in_task_manifest": True,
                "task_outcomes_computed": False,
                "teacher_likelihood_computed": False,
                "validation_or_test_rows_read": False,
            },
            "run": {
                "code_revision": subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
                "dataset_dir": str(dataset_dir),
                "bindings": {
                    "download_manifest": _binding(download_manifest),
                    "source_manifest": _binding(source_manifest),
                    "outer_folds": _binding(outer_folds),
                    "inner_folds": _binding(inner_folds),
                    "protocol": _binding(protocol),
                    "allocation_result": _binding(allocation_result),
                    "pilot_result": _binding(pilot_result),
                },
            },
        }
        report_path = temporary / "report.json"
        _write_json(report_path, report)
        completion = {
            "report": {"path": "report.json", "sha256": _sha256(report_path)},
            "task_manifest": {
                "path": "task-manifest.jsonl",
                "sha256": _sha256(task_path),
            },
            "image_manifest": {
                "path": "image-manifest.jsonl",
                "sha256": _sha256(image_manifest_path),
            },
            "images": {
                "count": len(image_rows),
                "aggregate_bytes": sum(int(row["bytes"]) for row in image_rows),
            },
        }
        _write_json(temporary / "complete.json", completion)
        temporary.rename(output_dir)
    print(json.dumps(completion, indent=2, sort_keys=True))


def columns_for_report() -> list[str]:
    return ["questionId", "question", "answers", "image", "data_split"]


def _binding(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256(path)}


if __name__ == "__main__":
    main()
