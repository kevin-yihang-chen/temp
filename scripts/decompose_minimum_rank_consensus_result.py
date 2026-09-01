#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from beyond_entropy.dataset import read_jsonl
from beyond_entropy.minimum_rank_consensus_decomposition import (
    decompose_minimum_rank_consensus,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            result.append(value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decompose the frozen minimum-rank consensus failure"
    )
    for name in ("rollouts", "scores", "evaluation-report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--expected-{name}-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "rollouts": args.rollouts.expanduser().resolve(),
        "scores": args.scores.expanduser().resolve(),
        "evaluation_report": args.evaluation_report.expanduser().resolve(),
    }
    expected = {
        "rollouts": args.expected_rollouts_sha256,
        "scores": args.expected_scores_sha256,
        "evaluation_report": args.expected_evaluation_report_sha256,
    }
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if hashes != expected:
        raise ValueError("consensus decomposition input SHA-256 mismatch")
    report = decompose_minimum_rank_consensus(
        read_jsonl(paths["rollouts"]),
        _jsonl(paths["scores"]),
        _json(paths["evaluation_report"]),
    )
    report["run"] = {
        "code_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "inputs": {
            name: {"path": str(path), "sha256": hashes[name]}
            for name, path in paths.items()
        },
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "sha256": _sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
