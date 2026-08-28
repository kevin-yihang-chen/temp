from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping


EXPECTED_MODEL_TYPE = "source_crossfit_pairwise_ranker_call_value_v1"
EXPECTED_FEATURE_MODE = "semantic-context"
EXPECTED_MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"


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


def _require_equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"frozen policy contract mismatch for {name}")


def _validate_policy(
    model: Mapping[str, Any],
    calibration: Mapping[str, Any],
    *,
    ranker_model_sha256: str,
) -> None:
    _require_equal(model.get("model_type"), EXPECTED_MODEL_TYPE, "model type")
    _require_equal(model.get("feature_mode"), EXPECTED_FEATURE_MODE, "feature mode")
    _require_equal(float(model["lambda_cost"]), 0.05, "lambda cost")
    _require_equal(int(model["seed"]), 20260828, "model seed")
    _require_equal(int(model["n_folds"]), 5, "source folds")
    if float(model["selected_ranker_c"]) not in {0.01, 0.1, 1.0}:
        raise ValueError("selected ranker C is outside the preregistered family")
    if float(model["selected_call_alpha"]) not in {1.0, 10.0, 100.0}:
        raise ValueError("selected call alpha is outside the preregistered family")
    thresholds = [float(value) for value in model.get("threshold_grid", [])]
    if not thresholds or len(thresholds) > 32 or len(set(thresholds)) != len(thresholds):
        raise ValueError("model threshold family is not the frozen unique <=32 grid")
    threshold = model.get("calibrated_threshold")
    if threshold is None:
        raise ValueError("answer-now calibration failure cannot open the formal role")

    _require_equal(
        calibration.get("selection_status"),
        "selected_non_degenerate_safe_threshold",
        "calibration status",
    )
    _require_equal(calibration.get("selected_threshold"), threshold, "threshold")
    _require_equal(calibration.get("method"), "bonferroni_bounded_mean_kl_ltt_v1", "risk method")
    _require_equal(float(calibration["lambda_cost"]), 0.05, "calibration cost")
    _require_equal(float(calibration["max_tool_cost"]), 1.0, "maximum tool cost")
    _require_equal(float(calibration["family_error"]), 0.05, "family error")
    _require_equal(int(calibration["hypothesis_count"]), len(thresholds) * 2, "hypothesis count")
    _require_equal(
        calibration.get("constraints"),
        [
            {"kind": "induced_harm", "limit": 0.005},
            {"kind": "net_negative_call_mass", "limit": 0.02},
        ],
        "risk constraints",
    )
    _require_equal(float(calibration["min_source_call_rate"]), 0.01, "minimum call rate")
    _require_equal(float(calibration["min_source_utility"]), 0.001, "minimum utility")
    _require_equal(calibration.get("selection_objective"), "source_call_rate", "selection objective")
    selected = calibration.get("selected")
    if not isinstance(selected, Mapping) or not bool(selected.get("risk_accepted")):
        raise ValueError("selected calibration threshold is not risk accepted")
    if float(selected["source_call_rate"]) < 0.01 or float(selected["source_utility"]) < 0.001:
        raise ValueError("selected calibration threshold is degenerate")

    embedded_calibration = model.get("risk_calibration")
    if not isinstance(embedded_calibration, Mapping):
        raise ValueError("calibrated model is missing its risk-calibration contract")
    for key in (
        "selection_status",
        "selected_threshold",
        "method",
        "constraints",
        "family_error",
        "hypothesis_count",
        "min_source_call_rate",
        "min_source_utility",
        "selection_objective",
    ):
        _require_equal(embedded_calibration.get(key), calibration.get(key), f"embedded {key}")
    run = calibration.get("run")
    if not isinstance(run, Mapping):
        raise ValueError("calibration report has no provenance")
    _require_equal(run.get("ranker_model_sha256"), ranker_model_sha256, "ranker model hash")
    _require_equal(run.get("formal_outcomes_used"), False, "formal outcome exclusion")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze and hash a successful scaled TextVQA policy"
    )
    parser.add_argument("--ranker-model", type=Path, required=True)
    parser.add_argument("--ranker-report", type=Path, required=True)
    parser.add_argument("--calibrated-model", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--ranker-rollouts", type=Path, required=True)
    parser.add_argument("--ranker-features", type=Path, required=True)
    parser.add_argument("--calibration-rollouts", type=Path, required=True)
    parser.add_argument("--calibration-features", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parents[1]
    status = subprocess.run(
        ["git", "-C", str(repo_dir), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("tracked worktree must be clean before freezing the policy")
    code_revision = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    artifact_paths = {
        "ranker_model": args.ranker_model.resolve(),
        "ranker_report": args.ranker_report.resolve(),
        "calibrated_model": args.calibrated_model.resolve(),
        "calibration_report": args.calibration_report.resolve(),
        "ranker_rollouts": args.ranker_rollouts.resolve(),
        "ranker_features": args.ranker_features.resolve(),
        "calibration_rollouts": args.calibration_rollouts.resolve(),
        "calibration_features": args.calibration_features.resolve(),
        "protocol": args.protocol.resolve(),
    }
    implementation_paths = {
        "freeze_script": Path(__file__).resolve(),
        "action_value_module": repo_dir / "src/beyond_entropy/scaled_action_value.py",
        "risk_control_module": repo_dir / "src/beyond_entropy/risk_control.py",
        "evaluator_module": repo_dir / "src/beyond_entropy/scaled_evaluation.py",
        "evaluator_script": repo_dir / "scripts/evaluate_scaled_textvqa_action_value.py",
        "collector_cli": repo_dir / "src/beyond_entropy/cli.py",
        "collector_backend": repo_dir / "src/beyond_entropy/qwen_backend.py",
        "rollout_module": repo_dir / "src/beyond_entropy/rollout.py",
        "semantic_module": repo_dir / "src/beyond_entropy/qwen_semantic.py",
        "question_reembed_module": repo_dir / "src/beyond_entropy/question_reembed.py",
        "attention_module": repo_dir / "src/beyond_entropy/attention_features.py",
        "question_reembed_script": repo_dir / "scripts/reembed_contextual_questions.py",
        "attention_script": repo_dir / "scripts/extract_question_region_attention.py",
        "fit_script": repo_dir / "scripts/fit_scaled_textvqa_action_value.py",
        "calibration_script": repo_dir / "scripts/calibrate_scaled_textvqa_action_value.py",
        "formal_export_script": repo_dir / "scripts/export_textvqa_train_scale_formal.py",
        "formal_gate_verifier": repo_dir / "scripts/verify_scaled_textvqa_formal_gate.py",
        "formal_rollout_submitter": repo_dir / "scripts/submit_textvqa_train_scale_formal_rollout.sh",
        "formal_feature_submitter": repo_dir / "scripts/submit_textvqa_train_scale_formal_features.sh",
        "formal_evaluation_job": repo_dir / "scripts/slurm_textvqa_train_scale_formal_evaluate.sh",
        "formal_evaluation_submitter": repo_dir / "scripts/submit_textvqa_train_scale_formal_evaluation.sh",
        "rollout_job": repo_dir / "scripts/slurm_textvqa_train_scale_rollout.sh",
        "feature_job": repo_dir / "scripts/slurm_textvqa_train_scale_features.sh",
    }
    missing = [path for path in (*artifact_paths.values(), *implementation_paths.values()) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"freeze input does not exist: {missing[0]}")
    artifact_hashes = {name: _sha256(path) for name, path in artifact_paths.items()}
    implementation_hashes = {
        name: _sha256(path) for name, path in implementation_paths.items()
    }
    model = _load_mapping(args.calibrated_model, "calibrated model")
    calibration = _load_mapping(args.calibration_report, "calibration report")
    _validate_policy(
        model,
        calibration,
        ranker_model_sha256=artifact_hashes["ranker_model"],
    )
    training = model.get("training_provenance")
    calibration_run = calibration["run"]
    if not isinstance(training, Mapping) or not isinstance(calibration_run, Mapping):
        raise ValueError("model training or calibration provenance is missing")
    for provenance, prefix in ((training, "ranker"), (calibration_run, "calibration")):
        _require_equal(
            provenance.get("rollouts_sha256"),
            artifact_hashes[f"{prefix}_rollouts"],
            f"{prefix} rollout provenance",
        )
        _require_equal(
            provenance.get("features_sha256"),
            artifact_hashes[f"{prefix}_features"],
            f"{prefix} feature provenance",
        )
    payload = {
        "schema_version": 1,
        "formal_gate_status": "ready_for_formal_manifest",
        "code_revision": code_revision,
        "policy": {
            "model_type": model["model_type"],
            "feature_mode": model["feature_mode"],
            "lambda_cost": model["lambda_cost"],
            "selected_ranker_c": model["selected_ranker_c"],
            "selected_call_alpha": model["selected_call_alpha"],
            "threshold_count": len(model["threshold_grid"]),
            "threshold_grid": model["threshold_grid"],
            "selected_threshold": model["calibrated_threshold"],
            "calibration_selection_status": calibration["selection_status"],
            "calibration_selected_summary": calibration["selected"],
        },
        "action_contract": {
            "model": "Qwen/Qwen2.5-VL-3B-Instruct",
            "model_revision": EXPECTED_MODEL_REVISION,
            "generation_seed": 0,
            "candidate_count": 4,
            "proposer": "ug-grid",
            "visual_crop_ratio": 2.0,
            "visual_cost": 1.0,
            "max_new_tokens": 32,
            "min_pixels": 200704,
            "max_pixels": 602112,
            "attention_implementation": "sdpa",
            "system_prompt": "You are a helpful assistant.",
            "scorer": "textvqa",
            "question_feature_mode": "multimodal-original",
            "question_region_attention_top_layers": 4,
        },
        "formal_test": {
            "allocated_sources": 5000,
            "manifest_materialized": False,
            "rollouts_collected": False,
            "bootstrap_resamples": 20000,
            "bootstrap_confidence": 0.975,
            "bootstrap_seed": 20260828,
        },
        "artifacts": {
            name: {"path": str(path), "sha256": artifact_hashes[name]}
            for name, path in artifact_paths.items()
        },
        "implementation": {
            name: {"path": str(path), "sha256": implementation_hashes[name]}
            for name, path in implementation_paths.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
