from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from beyond_entropy.dataset import read_jsonl
from beyond_entropy.qwen_semantic import (
    load_semantic_feature_dataset,
    validate_semantic_feature_dataset,
)
from beyond_entropy.risk_control import RiskConstraint, calibrate_source_risk_threshold
from beyond_entropy.scaled_action_value import (
    acquisition_calibration_rows,
    predict_scaled_action_value,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the preregistered source-level risk calibration"
    )
    parser.add_argument("--ranker-model", type=Path, required=True)
    parser.add_argument("--expected-ranker-model-sha256", required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--expected-rollouts-sha256", required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--expected-features-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    paths_and_hashes = (
        (args.ranker_model, args.expected_ranker_model_sha256, "ranker model"),
        (args.rollouts, args.expected_rollouts_sha256, "rollouts"),
        (args.features, args.expected_features_sha256, "features"),
    )
    actual_hashes = {}
    for path, expected, name in paths_and_hashes:
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"{name} SHA-256 mismatch")
        actual_hashes[name] = actual
    model = json.loads(args.ranker_model.read_text(encoding="utf-8"))
    if model.get("calibrated_threshold") is not None:
        raise ValueError("ranker model is already calibrated")
    thresholds = [float(value) for value in model.get("threshold_grid", [])]
    if not thresholds or len(thresholds) > 32:
        raise ValueError("ranker model does not contain the frozen threshold family")

    records = read_jsonl(args.rollouts)
    features = load_semantic_feature_dataset(args.features)
    validate_semantic_feature_dataset(features, records)
    if bool(features["metadata"].get("outcomes_included", True)):
        raise ValueError("risk calibration requires label-free feature storage")
    semantic_decisions = {
        (str(decision["state_id"]), str(decision["replicate_id"])): decision
        for decision in features["decisions"]
    }
    predictions = predict_scaled_action_value(
        model,
        records,
        semantic_decisions=semantic_decisions,
    )
    rows = acquisition_calibration_rows(predictions, records)
    calibration = calibrate_source_risk_threshold(
        rows,
        thresholds,
        constraints=[
            RiskConstraint("induced_harm", 0.005),
            RiskConstraint("net_negative_call_mass", 0.02),
        ],
        lambda_cost=0.05,
        max_tool_cost=1.0,
        family_error=0.05,
        min_source_call_rate=0.01,
        min_source_utility=0.001,
        selection_objective="source_call_rate",
    )
    code_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    run = {
        "code_revision": code_revision,
        "ranker_model": str(args.ranker_model.resolve()),
        "ranker_model_sha256": actual_hashes["ranker model"],
        "rollouts": str(args.rollouts.resolve()),
        "rollouts_sha256": actual_hashes["rollouts"],
        "features": str(args.features.resolve()),
        "features_sha256": actual_hashes["features"],
        "formal_outcomes_used": False,
        "preregistration": "docs/scaled_textvqa_risk_control_preregistration.md",
    }
    calibration["run"] = run
    frozen_model = json.loads(json.dumps(model))
    frozen_model["calibrated_threshold"] = calibration["selected_threshold"]
    frozen_model["risk_calibration"] = {
        "selection_status": calibration["selection_status"],
        "selected_threshold": calibration["selected_threshold"],
        "method": calibration["method"],
        "constraints": calibration["constraints"],
        "family_error": calibration["family_error"],
        "hypothesis_count": calibration["hypothesis_count"],
        "min_source_call_rate": calibration["min_source_call_rate"],
        "min_source_utility": calibration["min_source_utility"],
        "selection_objective": calibration["selection_objective"],
        "provenance": run,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "calibration.json").write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "model.json").write_text(
        json.dumps(frozen_model, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(calibration, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
