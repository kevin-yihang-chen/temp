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
    build_pilot_task_manifest,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked(path: Path, expected: str, name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or _sha256(resolved) != expected:
        raise ValueError(f"InfographicVQA DECAR {name} SHA-256 mismatch")
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
        or manifest.get("revision") != "539088ef8a8ada01ac8e2e6d4e372586748a265e"
        or manifest.get("file_count") != 24
        or manifest.get("aggregate_bytes") != 1_981_251_656
        or not isinstance(files, list)
        or len(files) != 24
    ):
        raise ValueError("InfographicVQA DECAR download contract changed")
    for row in files:
        path = dataset_dir / str(row["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or _sha256(path) != str(row["sha256"])
        ):
            raise ValueError("InfographicVQA DECAR parquet hash changed")


def _read_selected_payloads(
    dataset_dir: Path, identities: list[dict[str, Any]]
) -> dict[tuple[str, int], dict[str, Any]]:
    import pyarrow.parquet as pq  # type: ignore[import-not-found,import-untyped]

    targets_by_file: dict[str, list[int]] = defaultdict(list)
    for row in identities:
        targets_by_file[str(row["transport_file"])].append(int(row["transport_row"]))
    payloads: dict[tuple[str, int], dict[str, Any]] = {}
    columns = ["questionId", "question", "answers", "image", "data_split"]
    for filename, target_rows in sorted(targets_by_file.items()):
        parquet = pq.ParquetFile(dataset_dir / "InfographicVQA" / filename)
        if len(target_rows) != len(set(target_rows)):
            raise ValueError("DECAR pilot has duplicate parquet row locators")
        remaining = set(target_rows)
        row_start = 0
        for row_group in range(parquet.num_row_groups):
            row_count = parquet.metadata.row_group(row_group).num_rows
            selected = sorted(
                row for row in remaining if row_start <= row < row_start + row_count
            )
            if selected:
                table = parquet.read_row_group(row_group, columns=columns)
                for absolute_row in selected:
                    offset = absolute_row - row_start
                    payloads[(filename, absolute_row)] = {
                        name: table[name][offset].as_py() for name in columns
                    }
                    remaining.remove(absolute_row)
            row_start += row_count
        if remaining:
            raise ValueError("DECAR pilot parquet row locator is outside the file")
    return payloads


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize the registered InfographicVQA DECAR pilot"
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--download-manifest", type=Path, required=True)
    parser.add_argument("--expected-download-manifest-sha256", required=True)
    parser.add_argument("--pilot-question-manifest", type=Path, required=True)
    parser.add_argument("--expected-pilot-question-manifest-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--allocation-result", type=Path, required=True)
    parser.add_argument("--expected-allocation-result-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    download_manifest = _checked(
        args.download_manifest,
        args.expected_download_manifest_sha256,
        "download manifest",
    )
    pilot_path = _checked(
        args.pilot_question_manifest,
        args.expected_pilot_question_manifest_sha256,
        "pilot-question manifest",
    )
    protocol = _checked(args.protocol, args.expected_protocol_sha256, "protocol")
    allocation_result = _checked(
        args.allocation_result,
        args.expected_allocation_result_sha256,
        "allocation result",
    )
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite DECAR pilot: {output_dir}")
    _verify_download(dataset_dir, download_manifest)
    identities = _read_jsonl(pilot_path)
    payloads = _read_selected_payloads(dataset_dir, identities)
    report, task_rows, image_rows, image_bytes = build_pilot_task_manifest(
        identities, payloads
    )
    if report["population"]["questions"] != 512 or report["population"]["sources"] != 512:
        raise ValueError("DECAR pilot population contract changed")
    report["run"] = {
        "code_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "dataset_dir": str(dataset_dir),
        "download_manifest": {
            "path": str(download_manifest),
            "sha256": _sha256(download_manifest),
        },
        "pilot_question_manifest": {
            "path": str(pilot_path),
            "sha256": _sha256(pilot_path),
        },
        "protocol": {"path": str(protocol), "sha256": _sha256(protocol)},
        "allocation_result": {
            "path": str(allocation_result),
            "sha256": _sha256(allocation_result),
        },
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=output_dir.name + ".partial-", dir=output_dir.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        images_dir = temporary / "images"
        images_dir.mkdir()
        for encoded, raw in image_bytes.items():
            path = images_dir / f"{encoded}.img"
            with path.open("xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
        report_path = temporary / "report.json"
        task_path = temporary / "task-manifest.jsonl"
        image_manifest_path = temporary / "image-manifest.jsonl"
        _write_json(report_path, report)
        _write_jsonl(task_path, task_rows)
        _write_jsonl(image_manifest_path, image_rows)
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
                "aggregate_bytes": sum(len(raw) for raw in image_bytes.values()),
            },
        }
        _write_json(temporary / "complete.json", completion)
        temporary.rename(output_dir)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
