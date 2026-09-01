#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from beyond_entropy.infographicvqa_decar_allocation import build_decar_allocation


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
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze outcome-blind InfographicVQA DECAR source folds"
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--pilot-source-manifest", type=Path, required=True)
    parser.add_argument("--expected-pilot-source-manifest-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_path = _checked(
        args.source_manifest,
        args.expected_source_manifest_sha256,
        "source manifest",
    )
    pilot_path = _checked(
        args.pilot_source_manifest,
        args.expected_pilot_source_manifest_sha256,
        "pilot source manifest",
    )
    protocol_path = _checked(args.protocol, args.expected_protocol_sha256, "protocol")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite InfographicVQA DECAR allocation: {output_dir}"
        )

    report, outer_rows, inner_rows, pilot_questions = build_decar_allocation(
        _read_jsonl(source_path), _read_jsonl(pilot_path)
    )
    if report["population"] != {
        "questions": 23_946,
        "sources": 2_204,
        "images": 4_406,
        "pilot_questions": 512,
        "pilot_sources": 512,
    }:
        raise ValueError("InfographicVQA DECAR population contract changed")
    report["run"] = {
        "code_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "source_manifest": {"path": str(source_path), "sha256": _sha256(source_path)},
        "pilot_source_manifest": {
            "path": str(pilot_path),
            "sha256": _sha256(pilot_path),
        },
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=output_dir.name + ".partial-", dir=output_dir.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        report_path = temporary / "report.json"
        outer_path = temporary / "outer-folds.jsonl"
        inner_path = temporary / "inner-folds.jsonl"
        pilot_question_path = temporary / "pilot-question-manifest.jsonl"
        _write_json(report_path, report)
        _write_jsonl(outer_path, outer_rows)
        _write_jsonl(inner_path, inner_rows)
        _write_jsonl(pilot_question_path, pilot_questions)
        completion = {
            "report": {"path": "report.json", "sha256": _sha256(report_path)},
            "outer_folds": {
                "path": "outer-folds.jsonl",
                "sha256": _sha256(outer_path),
            },
            "inner_folds": {
                "path": "inner-folds.jsonl",
                "sha256": _sha256(inner_path),
            },
            "pilot_questions": {
                "path": "pilot-question-manifest.jsonl",
                "sha256": _sha256(pilot_question_path),
            },
        }
        _write_json(temporary / "complete.json", completion)
        temporary.rename(output_dir)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
