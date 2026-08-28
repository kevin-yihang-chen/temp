from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from beyond_entropy.dataset import read_jsonl
from beyond_entropy.qwen_semantic import (
    load_semantic_feature_dataset,
    validate_semantic_feature_dataset,
)
from beyond_entropy.scaled_action_value import fit_scaled_pairwise_action_value_model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit the preregistered scaled TextVQA action-value model"
    )
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--expected-rollouts-sha256")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--expected-features-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--feature-mode",
        choices=("semantic-context", "hybrid-context-semantic"),
        default="semantic-context",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()

    code_revision = os.environ.get("BE_CODE_REVISION")
    if not code_revision:
        code_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    rollout_sha256 = _sha256(args.rollouts)
    feature_sha256 = _sha256(args.features)
    for name, actual, expected in (
        ("rollouts", rollout_sha256, args.expected_rollouts_sha256),
        ("features", feature_sha256, args.expected_features_sha256),
    ):
        if expected is not None and actual != expected:
            raise ValueError(f"{name} SHA-256 mismatch")
    records = read_jsonl(args.rollouts)
    features = load_semantic_feature_dataset(args.features)
    validate_semantic_feature_dataset(features, records)
    if bool(features["metadata"].get("outcomes_included", True)):
        raise ValueError("primary scaled training requires label-free feature storage")
    semantic_decisions = {
        (str(decision["state_id"]), str(decision["replicate_id"])): decision
        for decision in features["decisions"]
    }
    report, model = fit_scaled_pairwise_action_value_model(
        records,
        feature_mode=args.feature_mode,
        semantic_decisions=semantic_decisions,
        n_folds=5,
        lambda_cost=0.05,
        ranker_c_values=(0.01, 0.1, 1.0),
        call_alpha_values=(1.0, 10.0, 100.0),
        max_thresholds=32,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=20260828,
    )
    run = {
        "code_revision": code_revision,
        "rollouts": str(args.rollouts.resolve()),
        "rollouts_sha256": rollout_sha256,
        "features": str(args.features.resolve()),
        "features_sha256": feature_sha256,
        "risk_calibration_outcomes_used": False,
        "formal_outcomes_used": False,
        "preregistration": "docs/scaled_textvqa_risk_control_preregistration.md",
    }
    report["run"] = run
    model["training_provenance"] = run
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "model.json").write_text(
        json.dumps(model, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
