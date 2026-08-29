from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping

from beyond_entropy.docvqa_calibration import (
    MODEL_REVISION,
    SUCCESS,
    validate_docvqa_calibration_artifact_bundle,
)
from beyond_entropy.docvqa_candidate_freeze import PROTOCOL_SHA256
from beyond_entropy.docvqa_formal import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CALIBRATION_SOURCES,
    FORMAL_SOURCES,
    POLICY_FREEZE_SCIENTIFIC_STATUS,
    REQUIRED_ARTIFACTS,
    REQUIRED_IMPLEMENTATION,
    validate_policy_freeze,
)
from beyond_entropy.docvqa_manifest_export import validate_exported_docvqa_manifest
from beyond_entropy.docvqa_train_allocation import sha256_file
from beyond_entropy.manifest_audit import audit_manifest
from beyond_entropy.rollout_audit import audit_sibling_rollout_bank


RANKER_SCIENTIFIC_STATUS = (
    "fresh DocVQA-train factorized-v2 ranker sibling bank; outcomes may train "
    "the sole candidate only"
)
CALIBRATION_SCIENTIFIC_STATUS = (
    "fresh DocVQA-train factorized-v2 calibration sibling bank; outcomes may "
    "calibrate the sole frozen candidate only"
)

ARTIFACT_SOURCES: dict[str, tuple[str, str | None]] = {
    "allocation": ("allocation", None),
    "allocation_audit": ("allocation_audit", None),
    "calibrated_model": ("calibration_dir", "model.json"),
    "calibration_audit": ("calibration_dir", "calibration.audit.json"),
    "calibration_features": ("calibration_features", None),
    "calibration_label_free_audit": ("calibration_label_free_audit", None),
    "calibration_manifest": ("calibration_manifest_dir", "manifest.jsonl"),
    "calibration_manifest_audit": (
        "calibration_manifest_dir",
        "manifest.audit.json",
    ),
    "calibration_manifest_provenance": (
        "calibration_manifest_dir",
        "manifest.provenance.json",
    ),
    "calibration_report": ("calibration_dir", "calibration.json"),
    "calibration_rollout_audit": ("calibration_rollout_audit", None),
    "calibration_rollouts": ("calibration_rollouts", None),
    "candidate": ("candidate", None),
    "candidate_audit": ("candidate_audit", None),
    "oof_report": ("oof_report", None),
    "protocol": ("protocol", None),
    "ranker_features": ("ranker_features", None),
    "ranker_label_free_audit": ("ranker_label_free_audit", None),
    "ranker_manifest": ("ranker_manifest_dir", "manifest.jsonl"),
    "ranker_manifest_audit": ("ranker_manifest_dir", "manifest.audit.json"),
    "ranker_manifest_provenance": (
        "ranker_manifest_dir",
        "manifest.provenance.json",
    ),
    "ranker_rollout_audit": ("ranker_rollout_audit", None),
    "ranker_rollouts": ("ranker_rollouts", None),
}

IMPLEMENTATION_PATHS = {
    "action_value": "src/beyond_entropy/action_value.py",
    "allocation_script": "scripts/allocate_docvqa_train_factorized_v2.py",
    "allocation_verifier": "scripts/verify_docvqa_train_factorized_v2_allocation.py",
    "attention_extraction": "scripts/extract_question_region_attention.py",
    "calibration_renderer": "scripts/render_docvqa_train_factorized_v2_calibration.py",
    "calibration_rollout_audit_script": (
        "scripts/audit_docvqa_train_factorized_v2_rollouts.py"
    ),
    "calibration_script": "scripts/calibrate_docvqa_train_factorized_v2.py",
    "candidate_freeze_script": (
        "scripts/freeze_docvqa_train_factorized_v2_candidate.py"
    ),
    "context_reembedding": "scripts/reembed_contextual_questions.py",
    "development_feature_job": "scripts/slurm_docvqa_train_factorized_v2_features.sh",
    "development_fit_job": "scripts/slurm_docvqa_train_factorized_v2_fit.sh",
    "development_manifest_export": (
        "scripts/export_docvqa_train_factorized_v2_manifest.py"
    ),
    "development_manifest_verifier": (
        "scripts/verify_docvqa_train_factorized_v2_manifest.py"
    ),
    "development_rollout_job": "scripts/slurm_docvqa_train_factorized_v2_rollout.sh",
    "docvqa_allocation_contract": "src/beyond_entropy/docvqa_train_allocation.py",
    "docvqa_calibration_contract": "src/beyond_entropy/docvqa_calibration.py",
    "docvqa_candidate_contract": "src/beyond_entropy/docvqa_candidate_freeze.py",
    "docvqa_formal_contract": "src/beyond_entropy/docvqa_formal.py",
    "docvqa_formal_export_contract": "src/beyond_entropy/docvqa_formal_export.py",
    "docvqa_manifest_contract": "src/beyond_entropy/docvqa_manifest_export.py",
    "factorized_evaluation": "src/beyond_entropy/factorized_evaluation.py",
    "formal_evaluation_job": (
        "scripts/slurm_docvqa_train_factorized_v2_formal_evaluate.sh"
    ),
    "formal_evaluator_script": (
        "scripts/evaluate_docvqa_train_factorized_v2_formal.py"
    ),
    "formal_export_job": "scripts/slurm_docvqa_train_factorized_v2_formal_export.sh",
    "formal_export_script": "scripts/export_docvqa_train_factorized_v2_formal.py",
    "formal_export_submission": (
        "scripts/submit_docvqa_train_factorized_v2_formal_export.sh"
    ),
    "formal_feature_job": "scripts/slurm_docvqa_train_factorized_v2_formal_features.sh",
    "formal_gate_verifier": (
        "scripts/verify_docvqa_train_factorized_v2_formal_gate.py"
    ),
    "formal_renderer": "scripts/render_docvqa_train_factorized_v2_formal.py",
    "formal_rollout_job": "scripts/slurm_docvqa_train_factorized_v2_formal_rollout.sh",
    "formal_submission": "scripts/submit_docvqa_train_factorized_v2_formal.sh",
    "label_free_audit": "scripts/audit_label_free_semantic_features.py",
    "manifest_audit": "src/beyond_entropy/manifest_audit.py",
    "manifest_export": "src/beyond_entropy/manifest_export.py",
    "policy_freeze_script": (
        "scripts/freeze_docvqa_train_factorized_v2_formal_policy.py"
    ),
    "qwen_backend": "src/beyond_entropy/qwen_backend.py",
    "qwen_semantic": "src/beyond_entropy/qwen_semantic.py",
    "risk_control": "src/beyond_entropy/risk_control.py",
    "rollout": "src/beyond_entropy/rollout.py",
    "rollout_audit": "src/beyond_entropy/rollout_audit.py",
}


def _require(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"DocVQA policy freeze mismatch for {name}")


def _load_mapping(path: Path, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"DocVQA policy freeze {name} must be a JSON object")
    return payload


def _component(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"DocVQA frozen component does not exist: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _artifact_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, (attribute, suffix) in ARTIFACT_SOURCES.items():
        base = Path(getattr(args, attribute)).resolve()
        paths[name] = base if suffix is None else base / suffix
    _require(set(paths), set(REQUIRED_ARTIFACTS), "artifact inventory")
    return paths


def _require_path_binding(
    metadata: Mapping[str, Any],
    *,
    path_key: str,
    path: Path,
    digest: str,
    name: str,
) -> None:
    _require(
        str(Path(str(metadata.get(path_key, ""))).resolve()),
        str(path.resolve()),
        f"{name} path",
    )
    _require(metadata.get(f"{path_key}_sha256"), digest, f"{name} SHA-256")


def _require_unmaterialized(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("DocVQA formal output must remain unmaterialized")


def _validate_successful_calibration(
    *,
    candidate: Mapping[str, Any],
    calibration: Mapping[str, Any],
    model: Mapping[str, Any],
    audit: Mapping[str, Any],
    calibration_sha256: str,
    model_sha256: str,
) -> float:
    status = validate_docvqa_calibration_artifact_bundle(
        calibration,
        model,
        audit,
        calibration_sha256=calibration_sha256,
        model_sha256=model_sha256,
    )
    _require(status, SUCCESS, "calibration selection")
    _require(calibration.get("n_sources"), CALIBRATION_SOURCES, "calibration sources")
    decisions = calibration.get("n_decisions")
    if not isinstance(decisions, int) or decisions < CALIBRATION_SOURCES:
        raise ValueError("DocVQA successful calibration has invalid decisions")
    selected = calibration.get("selected_threshold")
    if not isinstance(selected, (int, float)) or not math.isfinite(float(selected)):
        raise ValueError("DocVQA successful calibration lacks a finite threshold")
    threshold = float(selected)
    raw_grid = candidate.get("threshold_grid")
    if not isinstance(raw_grid, list) or threshold not in [
        float(value) for value in raw_grid
    ]:
        raise ValueError("DocVQA selected threshold is outside the frozen sequence")
    for name, value in candidate.items():
        expected = threshold if name == "threshold" else value
        _require(model.get(name), expected, f"calibrated model {name}")
    if set(model) != set(candidate) | {"risk_calibration"}:
        raise ValueError("DocVQA calibrated model has an unexpected field inventory")
    return threshold


def _validate_manifest_bundle(
    *,
    role: str,
    manifest_dir: Path,
    stored_audit: Mapping[str, Any],
    allocation: Mapping[str, Any],
    allocation_sha256: str,
    allocation_audit_sha256: str,
    candidate_sha256: str | None,
    provenance_sha256: str,
    code_revision: str,
    ranker_manifest_sha256: str | None,
) -> Mapping[str, Any]:
    recomputed = validate_exported_docvqa_manifest(
        audit_manifest(manifest_dir),
        allocation,
        role,
        allocation_sha256=allocation_sha256,
        allocation_audit_sha256=allocation_audit_sha256,
        candidate_sha256=candidate_sha256,
    )
    for name, expected in recomputed.items():
        _require(stored_audit.get(name), expected, f"{role} manifest audit {name}")
    expected_extra = {
        "code_revision": code_revision,
        "manifest_provenance_sha256": provenance_sha256,
        "ranker_manifest_sha256": ranker_manifest_sha256,
        "formal_targets_materialized": False,
    }
    for name, expected in expected_extra.items():
        _require(stored_audit.get(name), expected, f"{role} manifest audit {name}")
    return recomputed["manifest"]


def _validate_label_free_audit(
    audit: Mapping[str, Any],
    *,
    features: Path,
    features_sha256: str,
    rollouts: Path,
    rollouts_sha256: str,
    name: str,
) -> None:
    expected = {
        "features": str(features.resolve()),
        "features_sha256": features_sha256,
        "rollouts": str(rollouts.resolve()),
        "rollouts_sha256": rollouts_sha256,
        "outcome_fields_present": [],
        "outcomes_included_metadata": False,
    }
    for field, value in expected.items():
        _require(audit.get(field), value, f"{name} label-free audit {field}")
    decisions = audit.get("decisions")
    if not isinstance(decisions, int) or decisions <= 0:
        raise ValueError(f"DocVQA {name} label-free audit has invalid decisions")


def _git_state(repo_dir: Path) -> tuple[str, int]:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("tracked worktree must be clean before DocVQA policy freeze")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    commit_time = int(
        subprocess.run(
            ["git", "show", "-s", "--format=%ct", "HEAD"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return revision, commit_time


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a successful DocVQA-train factorized-v2 calibration and "
            "the complete pre-formal implementation"
        )
    )
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--allocation-audit", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-audit", type=Path, required=True)
    parser.add_argument("--oof-report", type=Path, required=True)
    parser.add_argument("--ranker-manifest-dir", type=Path, required=True)
    parser.add_argument("--ranker-rollouts", type=Path, required=True)
    parser.add_argument("--ranker-rollout-audit", type=Path, required=True)
    parser.add_argument("--ranker-features", type=Path, required=True)
    parser.add_argument("--ranker-label-free-audit", type=Path, required=True)
    parser.add_argument("--calibration-manifest-dir", type=Path, required=True)
    parser.add_argument("--calibration-rollouts", type=Path, required=True)
    parser.add_argument("--calibration-rollout-audit", type=Path, required=True)
    parser.add_argument("--calibration-features", type=Path, required=True)
    parser.add_argument("--calibration-label-free-audit", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--formal-output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    repo_dir = Path(__file__).resolve().parents[1]
    output_path = args.output.resolve()
    formal_output_dir = args.formal_output_dir.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite DocVQA policy freeze: {output_path}")
    _require_unmaterialized(formal_output_dir)
    code_revision, implementation_commit_time = _git_state(repo_dir)

    paths = _artifact_paths(args)
    artifacts = {name: _component(path) for name, path in paths.items()}
    _require(artifacts["protocol"]["sha256"], PROTOCOL_SHA256, "protocol SHA-256")
    calibration_times = [
        paths[name].stat().st_mtime
        for name in ("calibration_report", "calibrated_model", "calibration_audit")
    ]
    if implementation_commit_time > int(min(calibration_times)):
        raise ValueError("formal implementation was not committed before calibration output")

    candidate = _load_mapping(paths["candidate"], "candidate")
    candidate_audit = _load_mapping(paths["candidate_audit"], "candidate audit")
    allocation = _load_mapping(paths["allocation"], "allocation")
    allocation_audit = _load_mapping(paths["allocation_audit"], "allocation audit")
    calibration = _load_mapping(paths["calibration_report"], "calibration report")
    model = _load_mapping(paths["calibrated_model"], "calibrated model")
    calibration_audit = _load_mapping(paths["calibration_audit"], "calibration audit")
    threshold = _validate_successful_calibration(
        candidate=candidate,
        calibration=calibration,
        model=model,
        audit=calibration_audit,
        calibration_sha256=artifacts["calibration_report"]["sha256"],
        model_sha256=artifacts["calibrated_model"]["sha256"],
    )
    run = calibration.get("run")
    if not isinstance(run, Mapping):
        raise ValueError("DocVQA calibration report lacks run provenance")
    _require(run.get("code_revision"), code_revision, "calibration code revision")
    _require(run.get("formal_outcomes_used"), False, "formal outcome exclusion")
    calibration_run_components = {
        "candidate": "candidate",
        "candidate_audit": "candidate_audit",
        "allocation": "allocation",
        "allocation_audit": "allocation_audit",
        "calibration_manifest": "manifest",
        "calibration_manifest_provenance": "manifest_provenance",
        "calibration_rollouts": "rollouts",
        "calibration_rollout_audit": "rollout_audit",
        "calibration_features": "features",
        "protocol": "protocol",
    }
    for artifact_name, run_name in calibration_run_components.items():
        _require_path_binding(
            run,
            path_key=run_name,
            path=paths[artifact_name],
            digest=artifacts[artifact_name]["sha256"],
            name=f"calibration {artifact_name}",
        )

    candidate_freeze = candidate.get("candidate_freeze")
    if not isinstance(candidate_freeze, Mapping):
        raise ValueError("DocVQA candidate lacks freeze provenance")
    _require(candidate_freeze.get("code_revision"), code_revision, "candidate revision")
    ranker_candidate_components = {
        "oof_report": "development_report",
        "ranker_rollouts": "development_rollouts",
        "ranker_features": "development_features",
        "allocation": "allocation",
        "allocation_audit": "allocation_audit",
        "protocol": "protocol",
    }
    for artifact_name, candidate_name in ranker_candidate_components.items():
        _require_path_binding(
            candidate_freeze,
            path_key=candidate_name,
            path=paths[artifact_name],
            digest=artifacts[artifact_name]["sha256"],
            name=f"candidate {artifact_name}",
        )
    _require(
        candidate_audit.get("candidate_sha256"),
        artifacts["candidate"]["sha256"],
        "candidate audit binding",
    )
    _require(candidate_audit.get("code_revision"), code_revision, "candidate audit revision")
    _require(allocation.get("code_revision"), code_revision, "allocation revision")
    _require(allocation_audit.get("passed"), True, "allocation audit status")
    _require(
        allocation_audit.get("allocation_sha256"),
        artifacts["allocation"]["sha256"],
        "allocation audit binding",
    )

    ranker_manifest_audit = _load_mapping(
        paths["ranker_manifest_audit"], "ranker manifest audit"
    )
    calibration_manifest_audit = _load_mapping(
        paths["calibration_manifest_audit"], "calibration manifest audit"
    )
    ranker_manifest = _validate_manifest_bundle(
        role="ranker_training",
        manifest_dir=args.ranker_manifest_dir.resolve(),
        stored_audit=ranker_manifest_audit,
        allocation=allocation,
        allocation_sha256=artifacts["allocation"]["sha256"],
        allocation_audit_sha256=artifacts["allocation_audit"]["sha256"],
        candidate_sha256=None,
        provenance_sha256=artifacts["ranker_manifest_provenance"]["sha256"],
        code_revision=code_revision,
        ranker_manifest_sha256=None,
    )
    calibration_manifest = _validate_manifest_bundle(
        role="risk_calibration",
        manifest_dir=args.calibration_manifest_dir.resolve(),
        stored_audit=calibration_manifest_audit,
        allocation=allocation,
        allocation_sha256=artifacts["allocation"]["sha256"],
        allocation_audit_sha256=artifacts["allocation_audit"]["sha256"],
        candidate_sha256=artifacts["candidate"]["sha256"],
        provenance_sha256=artifacts["calibration_manifest_provenance"]["sha256"],
        code_revision=code_revision,
        ranker_manifest_sha256=artifacts["ranker_manifest"]["sha256"],
    )

    ranker_rollout_audit = _load_mapping(
        paths["ranker_rollout_audit"], "ranker rollout audit"
    )
    recomputed_ranker_rollout_audit = audit_sibling_rollout_bank(
        paths["ranker_manifest"],
        paths["ranker_rollouts"],
        expected_manifest_sha256=artifacts["ranker_manifest"]["sha256"],
        expected_states=int(ranker_manifest["count"]),
        expected_candidate_count=4,
        expected_model_revision=MODEL_REVISION,
        expected_scientific_status=RANKER_SCIENTIFIC_STATUS,
    )
    _require(
        ranker_rollout_audit,
        recomputed_ranker_rollout_audit,
        "ranker rollout audit recomputation",
    )

    calibration_rollout_audit = _load_mapping(
        paths["calibration_rollout_audit"], "calibration rollout audit"
    )
    recomputed_calibration_rollout_audit = audit_sibling_rollout_bank(
        paths["calibration_manifest"],
        paths["calibration_rollouts"],
        expected_manifest_sha256=artifacts["calibration_manifest"]["sha256"],
        expected_states=int(calibration_manifest["count"]),
        expected_candidate_count=4,
        expected_model_revision=MODEL_REVISION,
        expected_scientific_status=CALIBRATION_SCIENTIFIC_STATUS,
    )
    recomputed_calibration_rollout_audit.update(
        {
            "candidate_sha256": artifacts["candidate"]["sha256"],
            "candidate_audit_sha256": artifacts["candidate_audit"]["sha256"],
            "allocation_sha256": artifacts["allocation"]["sha256"],
            "allocation_audit_sha256": artifacts["allocation_audit"]["sha256"],
            "manifest_provenance_sha256": artifacts[
                "calibration_manifest_provenance"
            ]["sha256"],
            "protocol_sha256": artifacts["protocol"]["sha256"],
            "manifest_audit": calibration_manifest,
            "formal_outcomes_used": False,
        }
    )
    _require(
        calibration_rollout_audit,
        recomputed_calibration_rollout_audit,
        "calibration rollout audit recomputation",
    )

    _validate_label_free_audit(
        _load_mapping(paths["ranker_label_free_audit"], "ranker label-free audit"),
        features=paths["ranker_features"],
        features_sha256=artifacts["ranker_features"]["sha256"],
        rollouts=paths["ranker_rollouts"],
        rollouts_sha256=artifacts["ranker_rollouts"]["sha256"],
        name="ranker",
    )
    _validate_label_free_audit(
        _load_mapping(
            paths["calibration_label_free_audit"],
            "calibration label-free audit",
        ),
        features=paths["calibration_features"],
        features_sha256=artifacts["calibration_features"]["sha256"],
        rollouts=paths["calibration_rollouts"],
        rollouts_sha256=artifacts["calibration_rollouts"]["sha256"],
        name="calibration",
    )

    _require(set(IMPLEMENTATION_PATHS), set(REQUIRED_IMPLEMENTATION), "implementation inventory")
    implementation = {
        name: _component(repo_dir / relative_path)
        for name, relative_path in IMPLEMENTATION_PATHS.items()
    }
    payload = {
        "schema_version": 1,
        "scientific_status": POLICY_FREEZE_SCIENTIFIC_STATUS,
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
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_confidence": BOOTSTRAP_CONFIDENCE,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "artifacts": artifacts,
        "implementation": implementation,
    }
    validate_policy_freeze(payload)
    _require_unmaterialized(formal_output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
