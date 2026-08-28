from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping

from beyond_entropy.action_value import predict_frozen_factorized_action_values
from beyond_entropy.dataset import read_jsonl
from beyond_entropy.qwen_semantic import (
    load_semantic_feature_dataset,
    validate_semantic_feature_dataset,
)
from beyond_entropy.risk_control import threshold_grid_from_target_call_rates


EXPECTED_MODEL_SHA256 = (
    "2509e844de92f4b37e485ab26328d268318f596da93d80e60c0799351e7e52e9"
)
EXPECTED_REPORT_SHA256 = (
    "d9bafdb3fc73af00f1691e39c8974fc491594bd05b7b463a8bd01e410fe379ea"
)
EXPECTED_ROLLOUTS_SHA256 = (
    "1c1d5b67010b5ddfbdabe47072291336b34dcc54928e5db7a12727daa4f14c8e"
)
EXPECTED_FEATURES_SHA256 = (
    "93cdfa91b570fcc67f16bdd4e39d59489fa160e26c2797abf16d684f2f44a504"
)
EXPECTED_TRAINING_REVISION = "ae8e340c3309b14bb9c3b8691cdad7e7c2ff6edf"
TARGET_CALL_RATES = (
    0.0025,
    0.005,
    0.0075,
    0.01,
    0.0125,
    0.015,
    0.0175,
    0.02,
    0.025,
    0.03,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mapping(path: Path, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _require(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"factorized candidate violates frozen {name}")


def _validate_raw_model(model: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    common_expected = {
        "model_type": "multidomain_factorized_action_value",
        "training_protocol": "source_grouped_oof_domain_source_balanced_v2",
        "sample_weighting": "equal_domain_then_equal_source_then_equal_row",
        "feature_mode": "hybrid-context-semantic",
        "seed": 20260828,
        "n_folds": 5,
        "lambda_cost": 0.05,
        "selected_alpha": 1.0,
        "domains": ["textvqa"],
    }
    for name, value in common_expected.items():
        _require(model.get(name), value, f"model {name}")
        _require(report.get(name), value, f"report {name}")
    _require(model.get("state_feature_count"), 27, "model state feature count")
    _require(model.get("action_feature_count"), 46, "model action feature count")
    refit = report.get("refit")
    if not isinstance(refit, Mapping):
        raise ValueError("factorized report is missing refit dimensions")
    _require(refit.get("state_feature_count"), 27, "report state feature count")
    _require(refit.get("action_feature_count"), 46, "report action feature count")
    raw_threshold = model.get("threshold")
    if not isinstance(raw_threshold, (int, float)) or not math.isfinite(
        float(raw_threshold)
    ):
        raise ValueError("raw factorized model has no finite development threshold")
    run = report.get("run")
    if not isinstance(run, Mapping):
        raise ValueError("factorized report is missing run provenance")
    _require(run.get("code_revision"), EXPECTED_TRAINING_REVISION, "training revision")
    _require(run.get("formal_outcomes_used"), False, "formal outcome exclusion")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the sole factorized candidate and threshold sequence"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parents[1]
    tracked_status = subprocess.run(
        ["git", "-C", str(repo_dir), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if tracked_status.strip():
        raise ValueError("tracked worktree must be clean before candidate freeze")
    code_revision = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    expected_hashes = {
        "model": EXPECTED_MODEL_SHA256,
        "report": EXPECTED_REPORT_SHA256,
        "rollouts": EXPECTED_ROLLOUTS_SHA256,
        "features": EXPECTED_FEATURES_SHA256,
    }
    paths = {
        "model": args.model.resolve(),
        "report": args.report.resolve(),
        "rollouts": args.rollouts.resolve(),
        "features": args.features.resolve(),
        "protocol": args.protocol.resolve(),
    }
    for name, expected in expected_hashes.items():
        actual = _sha256(paths[name])
        if actual != expected:
            raise ValueError(f"{name} SHA-256 mismatch")
    protocol_sha256 = _sha256(paths["protocol"])
    model = _load_mapping(paths["model"], "raw model")
    report = _load_mapping(paths["report"], "development report")
    _validate_raw_model(model, report)

    records = read_jsonl(paths["rollouts"])
    features = load_semantic_feature_dataset(paths["features"])
    validate_semantic_feature_dataset(features, records)
    if bool(features["metadata"].get("outcomes_included", True)):
        raise ValueError("candidate freeze requires label-free semantic features")
    semantic_decisions = {
        (str(decision["state_id"]), str(decision["replicate_id"])): decision
        for decision in features["decisions"]
    }
    actions, scores_by_key = predict_frozen_factorized_action_values(
        model,
        records,
        semantic_decisions=semantic_decisions,
    )
    if len(scores_by_key) != 7912 or set(actions) != set(scores_by_key):
        raise RuntimeError("development predictions do not cover 7,912 decisions")
    scores = [scores_by_key[key] for key in sorted(scores_by_key)]
    thresholds = threshold_grid_from_target_call_rates(scores, TARGET_CALL_RATES)
    if any(left <= right for left, right in zip(thresholds, thresholds[1:])):
        raise RuntimeError("frozen threshold sequence is not strictly descending")
    threshold_summaries = [
        {
            "threshold": threshold,
            "development_call_rate": sum(score >= threshold for score in scores)
            / len(scores),
        }
        for threshold in thresholds
    ]

    candidate = dict(model)
    candidate["development_oof_threshold"] = float(candidate["threshold"])
    candidate["threshold"] = None
    candidate["decision_rule"] = (
        "factorized_expected_net_value_above_fixed_sequence_calibrated_margin"
    )
    candidate["threshold_grid"] = thresholds
    candidate["calibration_contract"] = {
        "method": "fixed_sequence_bounded_mean_kl_ltt_v1",
        "threshold_order": "strict_to_permissive_descending",
        "target_development_call_rates": list(TARGET_CALL_RATES),
        "threshold_summaries": threshold_summaries,
        "constraints": [
            {"kind": "induced_harm", "limit": 0.005},
            {"kind": "net_negative_call_mass", "limit": 0.02},
        ],
        "family_error": 0.05,
        "per_step_p_cutoff": 0.025,
        "min_source_call_rate": 0.01,
        "min_source_utility": 0.001,
        "calibration_sources": 3000,
        "formal_sources": 5953,
    }
    candidate["candidate_freeze"] = {
        "code_revision": code_revision,
        "raw_model": str(paths["model"]),
        "raw_model_sha256": EXPECTED_MODEL_SHA256,
        "development_report": str(paths["report"]),
        "development_report_sha256": EXPECTED_REPORT_SHA256,
        "development_rollouts": str(paths["rollouts"]),
        "development_rollouts_sha256": EXPECTED_ROLLOUTS_SHA256,
        "development_features": str(paths["features"]),
        "development_features_sha256": EXPECTED_FEATURES_SHA256,
        "protocol": str(paths["protocol"]),
        "protocol_sha256": protocol_sha256,
        "calibration_outcomes_used": False,
        "formal_outcomes_used": False,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(candidate, handle, indent=2, sort_keys=True)
        handle.write("\n")
    output_sha256 = _sha256(args.output)
    audit = {
        "passed": True,
        "scientific_status": (
            "single post-failure factorized candidate frozen before new calibration"
        ),
        "candidate": str(args.output.resolve()),
        "candidate_sha256": output_sha256,
        "code_revision": code_revision,
        "protocol_sha256": protocol_sha256,
        "development_decisions": len(scores),
        "action_predictions": len(actions),
        "threshold_count": len(thresholds),
        "thresholds": thresholds,
        "threshold_summaries": threshold_summaries,
        "calibration_outcomes_used": False,
        "formal_outcomes_used": False,
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    with args.audit_output.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
