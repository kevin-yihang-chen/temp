#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from beyond_entropy.dataset import read_jsonl
from beyond_entropy.minimum_rank_consensus_gate import (
    CONSENSUS_BOOTSTRAP_RESAMPLES,
    CONSENSUS_BOOTSTRAP_SEED,
    evaluate_minimum_rank_consensus_gate,
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
        raise ValueError(f"consensus {name} SHA-256 mismatch")
    return resolved


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen minimum-rank consensus risk gate"
    )
    for name in (
        "rollouts",
        "cost-report",
        "cost-score-report",
        "cost-model",
        "cost-scores",
        "incumbent-report",
        "incumbent-model",
        "incumbent-scores",
        "protocol",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--expected-{name}-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    names = (
        "rollouts",
        "cost_report",
        "cost_score_report",
        "cost_model",
        "cost_scores",
        "incumbent_report",
        "incumbent_model",
        "incumbent_scores",
        "protocol",
    )
    paths = {
        name: _checked(
            getattr(args, name), getattr(args, f"expected_{name}_sha256"), name
        )
        for name in names
    }
    if _json(paths["cost_report"]).get("decision") != (
        "cost_sensitive_direct_action_value_not_advanced"
    ):
        raise ValueError("consensus branch requires the frozen cost-sensitive result")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite consensus output: {output_dir}")

    report, score_report, contract, score_rows = (
        evaluate_minimum_rank_consensus_gate(
            read_jsonl(paths["rollouts"]),
            _jsonl(paths["cost_scores"]),
            _jsonl(paths["incumbent_scores"]),
            bound_inputs_verified=True,
            bootstrap_resamples=CONSENSUS_BOOTSTRAP_RESAMPLES,
            bootstrap_seed=CONSENSUS_BOOTSTRAP_SEED,
        )
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    run = {
        "code_revision": revision,
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "input_hashes_verified_before_score_construction": True,
        "fit_performed": False,
        "screenqa_inputs_used": False,
        "protected_role_inputs_used": False,
    }
    report["run"] = run
    score_report["run"] = run
    contract["run"] = run

    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "report.json"
    score_report_path = output_dir / "score-report.json"
    contract_path = output_dir / "contract.json"
    scores_path = output_dir / "scores.jsonl"
    _write_json(report_path, report)
    _write_json(score_report_path, score_report)
    _write_json(contract_path, contract)
    with scores_path.open("x", encoding="utf-8") as handle:
        for row in score_rows:
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    completion = {
        "decision": report["decision"],
        "report": {"path": str(report_path), "sha256": _sha256(report_path)},
        "score_report": {
            "path": str(score_report_path),
            "sha256": _sha256(score_report_path),
        },
        "contract": {"path": str(contract_path), "sha256": _sha256(contract_path)},
        "scores": {"path": str(scores_path), "sha256": _sha256(scores_path)},
    }
    _write_json(output_dir / "complete.json", completion)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
