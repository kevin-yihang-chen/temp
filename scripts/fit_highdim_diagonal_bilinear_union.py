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
from beyond_entropy.highdim_diagonal_bilinear_union import (
    HIGHDIM_BOOTSTRAP_RESAMPLES,
    HIGHDIM_SEED,
    evaluate_highdim_diagonal_bilinear_union,
)
from beyond_entropy.qwen_semantic import load_semantic_feature_dataset, validate_semantic_feature_dataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked(path: Path, expected: str, name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or _sha256(resolved) != expected:
        raise ValueError(f"highdim {name} SHA-256 mismatch")
    return resolved


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload is not an object: {path}")
    return payload


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            rows.append(row)
    return rows


def _write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the frozen highdim diagonal-bilinear union")
    for name in (
        "rollouts", "features", "predictions", "joint-report", "joint-model",
        "audited-scores", "incumbent-report", "incumbent-model",
        "previous-report", "previous-score-report", "previous-model", "protocol",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--expected-{name}-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    names = (
        "rollouts", "features", "predictions", "joint_report", "joint_model",
        "audited_scores", "incumbent_report", "incumbent_model",
        "previous_report", "previous_score_report", "previous_model", "protocol",
    )
    paths = {name: _checked(getattr(args, name), getattr(args, f"expected_{name}_sha256"), name) for name in names}
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite highdim output: {output_dir}")
    records = read_jsonl(paths["rollouts"])
    feature_payload = load_semantic_feature_dataset(paths["features"])
    validate_semantic_feature_dataset(feature_payload, records)
    semantic = {(str(row["state_id"]), str(row["replicate_id"])): row for row in feature_payload["decisions"]}
    report, score_report, model, score_rows = evaluate_highdim_diagonal_bilinear_union(
        {"docvqa": records},
        _jsonl(paths["predictions"]),
        _jsonl(paths["audited_scores"]),
        semantic_decisions_by_domain={"docvqa": semantic},
        bound_loss_proposer_report=_json(paths["joint_report"]),
        bound_inputs_verified=True,
        bootstrap_resamples=HIGHDIM_BOOTSTRAP_RESAMPLES,
        seed=HIGHDIM_SEED,
    )
    revision = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    run = {
        "code_revision": revision,
        "inputs": {name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()},
        "input_hashes_verified_before_fit": True,
        "screenqa_inputs_used": False,
        "protected_role_inputs_used": False,
    }
    report["run"] = run
    score_report["run"] = run
    model["run"] = run
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "report.json"
    score_report_path = output_dir / "score-report.json"
    model_path = output_dir / "model.json"
    scores_path = output_dir / "scores.jsonl"
    _write_json(report_path, report)
    _write_json(score_report_path, score_report)
    _write_json(model_path, model)
    with scores_path.open("x", encoding="utf-8") as handle:
        for row in score_rows:
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    completion = {
        "decision": report["decision"],
        "report": {"path": str(report_path), "sha256": _sha256(report_path)},
        "score_report": {"path": str(score_report_path), "sha256": _sha256(score_report_path)},
        "model": {"path": str(model_path), "sha256": _sha256(model_path)},
        "scores": {"path": str(scores_path), "sha256": _sha256(scores_path)},
    }
    _write_json(output_dir / "complete.json", completion)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
