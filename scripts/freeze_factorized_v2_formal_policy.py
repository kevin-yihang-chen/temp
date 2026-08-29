from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping

from beyond_entropy.factorized_formal import (
    ALLOCATION_AUDIT_SHA256,
    ALLOCATION_SHA256,
    CALIBRATION_DECISIONS,
    CALIBRATION_SOURCES,
    CANDIDATE_SHA256,
    FORMAL_SOURCES,
    PROTOCOL_SHA256,
    check_hash,
    load_mapping,
    sha256_file,
)


CALIBRATION_MANIFEST_PROVENANCE_SHA256 = (
    "3cf60f8474c10bc81b83b5cf47ef22224b010154b0933c2ffb00bec7225e0c45"
)


IMPLEMENTATION_PATHS = {
    "action_value": "src/beyond_entropy/action_value.py",
    "factorized_evaluation": "src/beyond_entropy/factorized_evaluation.py",
    "factorized_formal_contract": "src/beyond_entropy/factorized_formal.py",
    "risk_control": "src/beyond_entropy/risk_control.py",
    "rollout": "src/beyond_entropy/rollout.py",
    "qwen_backend": "src/beyond_entropy/qwen_backend.py",
    "qwen_semantic": "src/beyond_entropy/qwen_semantic.py",
    "rollout_audit": "src/beyond_entropy/rollout_audit.py",
    "manifest_export": "src/beyond_entropy/manifest_export.py",
    "manifest_audit": "src/beyond_entropy/manifest_audit.py",
    "calibration_script": "scripts/calibrate_factorized_textvqa_fixed_sequence.py",
    "freeze_script": "scripts/freeze_factorized_v2_formal_policy.py",
    "freeze_wrapper": "scripts/freeze_factorized_v2_formal_policy.sh",
    "formal_export_script": "scripts/export_textvqa_factorized_v2_formal.py",
    "formal_export_job": "scripts/slurm_textvqa_factorized_v2_formal_export.sh",
    "formal_export_submission": "scripts/submit_textvqa_factorized_v2_formal_export.sh",
    "formal_gate_verifier": "scripts/verify_factorized_v2_formal_gate.py",
    "formal_evaluator_script": "scripts/evaluate_factorized_textvqa_formal.py",
    "formal_renderer": "scripts/render_factorized_textvqa_formal.py",
    "formal_rollout_job": "scripts/slurm_textvqa_factorized_v2_formal_rollout.sh",
    "formal_feature_job": "scripts/slurm_textvqa_factorized_v2_formal_features.sh",
    "formal_evaluation_job": "scripts/slurm_textvqa_factorized_v2_formal_evaluate.sh",
    "formal_submission": "scripts/submit_textvqa_factorized_v2_formal.sh",
    "rollout_audit_script": "scripts/audit_scaled_textvqa_rollouts.py",
    "context_reembedding": "scripts/reembed_contextual_questions.py",
    "attention_extraction": "scripts/extract_question_region_attention.py",
    "label_free_audit": "scripts/audit_label_free_semantic_features.py",
}


def _require(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"policy freeze contract mismatch for {name}")


def _component(path: Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"frozen component does not exist: {path}")
    return {"path": str(path), "sha256": sha256_file(path)}


def _validate_successful_calibration(
    *,
    candidate: Mapping[str, Any],
    calibration: Mapping[str, Any],
    model: Mapping[str, Any],
) -> float:
    _require(
        calibration.get("selection_status"),
        "selected_non_degenerate_safe_threshold",
        "calibration selection",
    )
    _require(calibration.get("n_sources"), CALIBRATION_SOURCES, "sources")
    _require(calibration.get("n_decisions"), CALIBRATION_DECISIONS, "decisions")
    selected_threshold = calibration.get("selected_threshold")
    if not isinstance(selected_threshold, (int, float)) or not math.isfinite(
        float(selected_threshold)
    ):
        raise ValueError("successful calibration must select a finite threshold")
    threshold = float(selected_threshold)
    if threshold not in [float(value) for value in candidate["threshold_grid"]]:
        raise ValueError("calibration selected a threshold outside the frozen sequence")
    for name, value in candidate.items():
        if name == "threshold":
            _require(model.get(name), threshold, "calibrated model threshold")
        else:
            _require(model.get(name), value, f"calibrated model {name}")
    risk = model.get("risk_calibration")
    if not isinstance(risk, Mapping):
        raise ValueError("calibrated model is missing risk calibration")
    _require(
        risk.get("selection_status"),
        "selected_non_degenerate_safe_threshold",
        "model selection",
    )
    _require(risk.get("selected_threshold"), threshold, "model selected threshold")
    run = calibration.get("run")
    if not isinstance(run, Mapping):
        raise ValueError("calibration report is missing run provenance")
    _require(run.get("candidate_sha256"), CANDIDATE_SHA256, "run candidate")
    _require(run.get("allocation_sha256"), ALLOCATION_SHA256, "run allocation")
    _require(run.get("protocol_sha256"), PROTOCOL_SHA256, "run protocol")
    _require(run.get("formal_outcomes_used"), False, "formal outcome exclusion")
    return threshold


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze a successful factorized-v2 calibration before formal export"
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--allocation-audit", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest-provenance", type=Path, required=True)
    parser.add_argument("--rollout-audit", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parents[1]
    fixed_inputs = (
        (args.candidate, CANDIDATE_SHA256, "candidate"),
        (args.allocation, ALLOCATION_SHA256, "allocation"),
        (args.allocation_audit, ALLOCATION_AUDIT_SHA256, "allocation audit"),
        (args.protocol, PROTOCOL_SHA256, "protocol"),
        (
            args.manifest_provenance,
            CALIBRATION_MANIFEST_PROVENANCE_SHA256,
            "calibration manifest provenance",
        ),
    )
    for path, expected, name in fixed_inputs:
        check_hash(path, expected, name)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite policy freeze: {args.output}")
    tracked_status = subprocess.run(
        ["git", "-C", str(repo_dir), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if tracked_status.strip():
        raise ValueError("tracked worktree must be clean before policy freeze")
    code_revision = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    implementation_commit_time = int(
        subprocess.run(
            ["git", "-C", str(repo_dir), "show", "-s", "--format=%ct", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    if implementation_commit_time > int(args.calibration.stat().st_mtime):
        raise ValueError("formal implementation was not committed before calibration output")

    candidate = load_mapping(args.candidate, "candidate")
    allocation = load_mapping(args.allocation, "allocation")
    allocation_audit = load_mapping(args.allocation_audit, "allocation audit")
    calibration = load_mapping(args.calibration, "calibration")
    model = load_mapping(args.model, "calibrated model")
    threshold = _validate_successful_calibration(
        candidate=candidate,
        calibration=calibration,
        model=model,
    )
    contract = allocation.get("selection_contract")
    if not isinstance(contract, Mapping) or (
        contract.get("formal_manifest_exported") is not False
        or contract.get("formal_rollouts_collected") is not False
        or contract.get("selection_target_fields_accessed") is not False
    ):
        raise ValueError("allocation no longer describes a sealed formal role")
    _require(allocation_audit.get("passed"), True, "allocation audit status")
    _require(
        allocation_audit.get("formal_outcomes_collected"),
        False,
        "allocation formal outcome exclusion",
    )

    run = calibration["run"]
    artifacts = {
        "candidate": _component(args.candidate),
        "allocation": _component(args.allocation),
        "allocation_audit": _component(args.allocation_audit),
        "calibration_report": _component(args.calibration),
        "calibrated_model": _component(args.model),
        "protocol": _component(args.protocol),
        "calibration_manifest_provenance": _component(
            args.manifest_provenance
        ),
        "calibration_rollout_audit": _component(args.rollout_audit),
    }
    run_components = {
        "calibration_manifest": ("manifest", "manifest_sha256"),
        "calibration_rollouts": ("rollouts", "rollouts_sha256"),
        "calibration_features": ("features", "features_sha256"),
    }
    for name, (path_key, hash_key) in run_components.items():
        path = Path(str(run.get(path_key, ""))).resolve()
        expected = str(run.get(hash_key, ""))
        check_hash(path, expected, name)
        artifacts[name] = {"path": str(path), "sha256": expected}
    if (
        artifacts["calibration_rollout_audit"]["sha256"]
        != run.get("rollout_audit_sha256")
    ):
        raise ValueError("calibration rollout audit differs from calibration run")
    implementation = {
        name: _component(repo_dir / relative_path)
        for name, relative_path in IMPLEMENTATION_PATHS.items()
    }
    payload = {
        "schema_version": 1,
        "scientific_status": (
            "successful fresh fixed-sequence calibration; exact factorized policy "
            "and formal implementation frozen before formal manifest export"
        ),
        "formal_gate_status": "ready_for_formal_manifest",
        "formal_outcomes_used": False,
        "code_revision": code_revision,
        "implementation_commit_time_unix": implementation_commit_time,
        "implementation_committed_before_calibration_output": True,
        "calibration": {
            "selection_status": calibration["selection_status"],
            "selected_threshold": threshold,
            "n_sources": calibration["n_sources"],
            "n_decisions": calibration["n_decisions"],
            "formal_outcomes_used": False,
        },
        "formal_test": {
            "allocated_sources": FORMAL_SOURCES,
            "manifest_materialized": False,
            "rollouts_collected": False,
            "outcomes_used": False,
            "bootstrap_resamples": 20000,
            "bootstrap_confidence": 0.975,
            "bootstrap_seed": 20260828,
        },
        "artifacts": artifacts,
        "implementation": implementation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
