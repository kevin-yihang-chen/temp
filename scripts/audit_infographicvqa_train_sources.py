#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from beyond_entropy.infographicvqa_source_audit import (
    audit_infographicvqa_train_sources,
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
        raise ValueError(f"InfographicVQA {name} SHA-256 mismatch")
    return resolved


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit pinned InfographicVQA train sources without reading labels"
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--download-manifest", type=Path, required=True)
    parser.add_argument("--expected-download-manifest-sha256", required=True)
    parser.add_argument("--audit-freeze", type=Path, required=True)
    parser.add_argument("--expected-audit-freeze-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    manifest_path = _checked(
        args.download_manifest,
        args.expected_download_manifest_sha256,
        "download manifest",
    )
    audit_freeze = _checked(
        args.audit_freeze, args.expected_audit_freeze_sha256, "audit freeze"
    )
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite InfographicVQA audit: {output_dir}")
    parquet_paths = sorted((dataset_dir / "InfographicVQA").glob("train-*.parquet"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("InfographicVQA download manifest is not an object")
    report, source_rows, pilot_rows = audit_infographicvqa_train_sources(
        parquet_paths, manifest
    )
    report["run"] = {
        "code_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "dataset_dir": str(dataset_dir),
        "download_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
        },
        "audit_freeze": {
            "path": str(audit_freeze),
            "sha256": _sha256(audit_freeze),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "report.json"
    sources_path = output_dir / "source-manifest.jsonl"
    pilot_path = output_dir / "pilot-source-manifest.jsonl"
    _write_json(report_path, report)
    _write_jsonl(sources_path, source_rows)
    _write_jsonl(pilot_path, pilot_rows)
    completion = {
        "report": {"path": str(report_path), "sha256": _sha256(report_path)},
        "source_manifest": {
            "path": str(sources_path),
            "sha256": _sha256(sources_path),
        },
        "pilot_source_manifest": {
            "path": str(pilot_path),
            "sha256": _sha256(pilot_path),
        },
    }
    _write_json(output_dir / "complete.json", completion)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
