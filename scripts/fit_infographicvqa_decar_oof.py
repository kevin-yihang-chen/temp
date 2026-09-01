#!/usr/bin/env python3
"""Fit the frozen InfographicVQA DECAR variants under nested source OOF."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from beyond_entropy.dataset import read_jsonl
from beyond_entropy.infographicvqa_decar import assemble_decar_dataset, fit_nested_oof
from beyond_entropy.qwen_semantic import load_semantic_feature_dataset


EXPECTED_DECISIONS = 23_946
EXPECTED_SOURCES = 2_204
EXPECTED_IMAGES = 4_406


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked(path: Path, expected: str, name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or _sha256(resolved) != expected:
        raise ValueError(f"InfographicVQA DECAR OOF {name} SHA-256 mismatch")
    return resolved


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
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


def _fold_maps(
    outer_rows: list[dict[str, Any]], inner_rows: list[dict[str, Any]]
) -> tuple[dict[str, int], dict[tuple[int, str], int]]:
    outer: dict[str, int] = {}
    for row in outer_rows:
        source_id = str(row["source_id"])
        if source_id in outer:
            raise ValueError("DECAR OOF outer-fold source is duplicated")
        outer[source_id] = int(row["outer_fold"])
    inner: dict[tuple[int, str], int] = {}
    for row in inner_rows:
        key = (int(row["outer_test_fold"]), str(row["source_id"]))
        if key in inner:
            raise ValueError("DECAR OOF inner-fold context is duplicated")
        inner[key] = int(row["inner_fold"])
    return outer, inner


def _image_geometry(rows: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for row in rows:
        if set(row) != {
            "bytes",
            "decoded_rgb_sha256",
            "encoded_sha256",
            "height",
            "path",
            "width",
        }:
            raise ValueError("DECAR OOF image-manifest schema changed")
        image_id = str(row["decoded_rgb_sha256"])
        if image_id in result:
            raise ValueError("DECAR OOF image geometry is duplicated")
        result[image_id] = (int(row["width"]), int(row["height"]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "rollouts",
        "answer-nll",
        "features",
        "image-manifest",
        "outer-folds",
        "inner-folds",
        "protocol",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--expected-{name}-sha256", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    input_names = (
        "rollouts",
        "answer_nll",
        "features",
        "image_manifest",
        "outer_folds",
        "inner_folds",
        "protocol",
    )
    paths = {
        name: _checked(
            getattr(args, name), getattr(args, f"expected_{name}_sha256"), name
        )
        for name in input_names
    }
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite DECAR OOF output: {output_dir}")

    records = read_jsonl(paths["rollouts"])
    nll_rows = _read_jsonl_objects(paths["answer_nll"])
    feature_payload = load_semantic_feature_dataset(paths["features"])
    image_rows = _read_jsonl_objects(paths["image_manifest"])
    outer_rows = _read_jsonl_objects(paths["outer_folds"])
    inner_rows = _read_jsonl_objects(paths["inner_folds"])
    outer, inner = _fold_maps(outer_rows, inner_rows)
    dataset = assemble_decar_dataset(
        records,
        nll_rows,
        feature_payload,
        _image_geometry(image_rows),
    )
    if (
        dataset.decisions != EXPECTED_DECISIONS
        or len(set(dataset.source_ids)) != EXPECTED_SOURCES
        or len(set(dataset.image_ids)) != EXPECTED_IMAGES
    ):
        raise ValueError("DECAR OOF full population changed")

    start = time.monotonic()
    predictions, audit = fit_nested_oof(
        dataset,
        outer,
        inner,
        device=args.device,
        epochs=args.epochs,
    )
    runtime_seconds = time.monotonic() - start
    rows = predictions.pop("predictions")
    if not isinstance(rows, list) or len(rows) != EXPECTED_DECISIONS:
        raise RuntimeError("DECAR OOF output coverage changed")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    run = {
        "code_revision": revision,
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "input_hashes_verified_before_fit": True,
        "validation_or_test_inputs_used": False,
        "prediction_outcomes_included": False,
        "device": args.device,
        "epochs": args.epochs,
        "runtime_seconds": runtime_seconds,
    }
    audit["run"] = run
    report = {
        "schema": "infographicvqa_decar_oof_fit_report_v1",
        "scientific_endpoints_computed": False,
        "scientific_endpoints_read": False,
        "prediction_outcomes_included": False,
        "population": {
            "decisions": dataset.decisions,
            "sources": len(set(dataset.source_ids)),
            "images": len(set(dataset.image_ids)),
        },
        "prediction_metadata": predictions["metadata"],
        "run": run,
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=output_dir.name + ".partial-", dir=output_dir.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        prediction_path = temporary / "predictions.jsonl"
        audit_path = temporary / "audit.json"
        report_path = temporary / "report.json"
        _write_jsonl(prediction_path, rows)
        _write_json(audit_path, audit)
        _write_json(report_path, report)
        completion = {
            "schema": "infographicvqa_decar_oof_fit_complete_v1",
            "predictions": {
                "path": "predictions.jsonl",
                "sha256": _sha256(prediction_path),
            },
            "audit": {"path": "audit.json", "sha256": _sha256(audit_path)},
            "report": {"path": "report.json", "sha256": _sha256(report_path)},
            "prediction_rows": len(rows),
            "prediction_outcomes_included": False,
        }
        _write_json(temporary / "complete.json", completion)
        temporary.replace(output_dir)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
