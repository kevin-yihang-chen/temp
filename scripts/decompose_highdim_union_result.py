#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from beyond_entropy.dataset import read_jsonl
from beyond_entropy.highdim_union_decomposition import decompose_highdim_union


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decompose a frozen highdim union result into stopping and action effects"
    )
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--expected-rollouts-sha256", required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--expected-scores-sha256", required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--expected-evaluation-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "rollouts": args.rollouts.resolve(),
        "scores": args.scores.resolve(),
        "evaluation_report": args.evaluation_report.resolve(),
    }
    expected = {
        "rollouts": args.expected_rollouts_sha256,
        "scores": args.expected_scores_sha256,
        "evaluation_report": args.expected_evaluation_report_sha256,
    }
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if hashes != expected:
        raise ValueError("highdim decomposition input SHA-256 mismatch")
    evaluation_report = _read_json(paths["evaluation_report"])
    if evaluation_report.get("decision") != "highdim_diagonal_bilinear_union_not_advanced":
        raise ValueError("highdim decomposition requires the frozen negative decision")
    report = decompose_highdim_union(
        read_jsonl(paths["rollouts"]),
        _read_jsonl_objects(paths["scores"]),
    )
    reproduced = report["comparisons"]["highdim_full"]
    for weighting in ("source_balanced", "question_balanced"):
        for method in ("incumbent", "highdim_full"):
            expected_method = (
                "highdim_diagonal_bilinear" if method == "highdim_full" else method
            )
            for metric, value in reproduced[weighting][method].items():
                original = evaluation_report[weighting][expected_method][metric]
                if value is None or original is None:
                    if value is not original:
                        raise ValueError("highdim decomposition null metric differs")
                elif not math.isclose(
                    float(value), float(original), rel_tol=0.0, abs_tol=1e-15
                ):
                    raise ValueError("highdim decomposition does not reproduce result")
    report["audits"]["frozen_result_metrics_reproduced"] = True
    report["run"] = {
        "code_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "paths": {name: str(path) for name, path in paths.items()},
        "sha256": hashes,
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "sha256": _sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
