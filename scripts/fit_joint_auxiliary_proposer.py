#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import torch  # type: ignore[import-not-found]

from beyond_entropy.dataset import read_jsonl
from beyond_entropy.joint_auxiliary_proposer import (
    JOINT_BOOTSTRAP_RESAMPLES,
    JOINT_EPOCHS,
    JOINT_FOLDS,
    JOINT_HIDDEN_DIMS,
    JOINT_LEARNING_RATE,
    JOINT_LOSS_WEIGHT,
    JOINT_SEED,
    JOINT_WEIGHT_DECAY,
    fit_joint_auxiliary_action_proposer,
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
        raise ValueError(f"joint auxiliary {name} SHA-256 mismatch")
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


def _revision(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit the frozen non-ScreenQA joint auxiliary action proposer"
    )
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--expected-rollouts-sha256", required=True)
    parser.add_argument("--answer-nll", type=Path, required=True)
    parser.add_argument("--expected-answer-nll-sha256", required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--expected-features-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("the frozen joint auxiliary run requires CUDA")
    accelerator_name = torch.cuda.get_device_name(0)
    if "H800" not in accelerator_name:
        raise RuntimeError(
            f"the frozen joint auxiliary run requires H800, got {accelerator_name!r}"
        )

    repo = Path(__file__).resolve().parents[1]
    rollouts_path = _checked(
        args.rollouts, args.expected_rollouts_sha256, "rollouts"
    )
    nll_path = _checked(args.answer_nll, args.expected_answer_nll_sha256, "NLL")
    features_path = _checked(
        args.features, args.expected_features_sha256, "semantic features"
    )
    protocol_path = _checked(
        args.protocol, args.expected_protocol_sha256, "protocol"
    )
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite joint output: {output_dir}")

    records = read_jsonl(rollouts_path)
    nll_rows = _jsonl(nll_path)
    semantic_payload = load_semantic_feature_dataset(features_path)
    validate_semantic_feature_dataset(semantic_payload, records)
    semantic_decisions = {
        (str(decision["state_id"]), str(decision["replicate_id"])): decision
        for decision in semantic_payload["decisions"]
    }
    report, model, predictions = fit_joint_auxiliary_action_proposer(
        {"docvqa": records},
        {"docvqa": nll_rows},
        semantic_decisions_by_domain={"docvqa": semantic_decisions},
        feature_mode="hybrid-context-semantic",
        n_folds=JOINT_FOLDS,
        seed=JOINT_SEED,
        epochs=JOINT_EPOCHS,
        learning_rate=JOINT_LEARNING_RATE,
        weight_decay=JOINT_WEIGHT_DECAY,
        loss_weight=JOINT_LOSS_WEIGHT,
        hidden_dims=JOINT_HIDDEN_DIMS,
        bootstrap_resamples=JOINT_BOOTSTRAP_RESAMPLES,
        device="cuda",
    )
    expected_counts = {
        "n_sources": 3500,
        "n_decisions": 13580,
        "n_zoom_rows": 54320,
        "target_counts": {
            "positive_gain": 1604,
            "negative_gain": 1535,
            "neutral_gain": 51181,
        },
    }
    for name, expected in expected_counts.items():
        if report.get(name) != expected:
            raise ValueError(f"joint auxiliary frozen count mismatch for {name}")
    run = {
        "code_revision": _revision(repo),
        "protocol": {"path": str(protocol_path), "sha256": args.expected_protocol_sha256},
        "inputs": {
            "rollouts": {
                "path": str(rollouts_path),
                "sha256": args.expected_rollouts_sha256,
            },
            "answer_nll": {
                "path": str(nll_path),
                "sha256": args.expected_answer_nll_sha256,
            },
            "semantic_features": {
                "path": str(features_path),
                "sha256": args.expected_features_sha256,
            },
        },
        "device": "cuda",
        "accelerator_name": accelerator_name,
        "screenqa_inputs_used": False,
        "docvqa_calibration_formal_reserve_inputs_used": False,
    }
    report["run"] = run
    model["run"] = run

    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "report.json"
    model_path = output_dir / "model.json"
    predictions_path = output_dir / "oof-predictions.jsonl"
    _write_json(report_path, report)
    _write_json(model_path, model)
    with predictions_path.open("x", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    completion = {
        "decision": report["decision"],
        "report": {"path": str(report_path), "sha256": _sha256(report_path)},
        "model": {"path": str(model_path), "sha256": _sha256(model_path)},
        "oof_predictions": {
            "path": str(predictions_path),
            "sha256": _sha256(predictions_path),
        },
    }
    _write_json(output_dir / "complete.json", completion)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
