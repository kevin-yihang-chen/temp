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
from beyond_entropy.decoupled_loss_gate import (
    DECOUPLED_BOOTSTRAP_RESAMPLES,
    DECOUPLED_SEED,
    evaluate_decoupled_loss_proposal_gate,
)
from beyond_entropy.qwen_semantic import (
    load_semantic_feature_dataset,
    validate_semantic_feature_dataset,
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
        raise ValueError(f"decoupled {name} SHA-256 mismatch")
    return resolved


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen loss-proposal/factorized-gate composition"
    )
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--expected-rollouts-sha256", required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--expected-features-sha256", required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--expected-predictions-sha256", required=True)
    parser.add_argument("--joint-report", type=Path, required=True)
    parser.add_argument("--expected-joint-report-sha256", required=True)
    parser.add_argument("--joint-model", type=Path, required=True)
    parser.add_argument("--expected-joint-model-sha256", required=True)
    parser.add_argument("--incumbent-report", type=Path, required=True)
    parser.add_argument("--expected-incumbent-report-sha256", required=True)
    parser.add_argument("--incumbent-model", type=Path, required=True)
    parser.add_argument("--expected-incumbent-model-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "rollouts": _checked(args.rollouts, args.expected_rollouts_sha256, "rollouts"),
        "features": _checked(args.features, args.expected_features_sha256, "features"),
        "predictions": _checked(
            args.predictions, args.expected_predictions_sha256, "predictions"
        ),
        "joint_report": _checked(
            args.joint_report, args.expected_joint_report_sha256, "joint report"
        ),
        "joint_model": _checked(
            args.joint_model, args.expected_joint_model_sha256, "joint model"
        ),
        "incumbent_report": _checked(
            args.incumbent_report,
            args.expected_incumbent_report_sha256,
            "incumbent report",
        ),
        "incumbent_model": _checked(
            args.incumbent_model,
            args.expected_incumbent_model_sha256,
            "incumbent model",
        ),
        "protocol": _checked(args.protocol, args.expected_protocol_sha256, "protocol"),
    }
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite decoupled output: {output_dir}")

    records = read_jsonl(paths["rollouts"])
    feature_payload = load_semantic_feature_dataset(paths["features"])
    validate_semantic_feature_dataset(feature_payload, records)
    semantic = {
        (str(row["state_id"]), str(row["replicate_id"])): row
        for row in feature_payload["decisions"]
    }
    report, score_report, score_rows = evaluate_decoupled_loss_proposal_gate(
        {"docvqa": records},
        _jsonl(paths["predictions"]),
        semantic_decisions_by_domain={"docvqa": semantic},
        bootstrap_resamples=DECOUPLED_BOOTSTRAP_RESAMPLES,
        bootstrap_seed=DECOUPLED_SEED,
    )
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
        "screenqa_inputs_used": False,
    }
    report["run"] = run
    score_report["run"] = run
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "report.json"
    score_report_path = output_dir / "score-report.json"
    scores_path = output_dir / "scores.jsonl"
    _write_json(report_path, report)
    _write_json(score_report_path, score_report)
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
        "scores": {"path": str(scores_path), "sha256": _sha256(scores_path)},
    }
    _write_json(output_dir / "complete.json", completion)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
