#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from beyond_entropy.action_value import factorized_acquisition_calibration_rows
from beyond_entropy.dataset import read_jsonl
from beyond_entropy.risk_control import (
    AcquisitionCalibrationRow,
    RiskConstraint,
    calibrate_source_risk_threshold_fixed_sequence,
)


_export_module = importlib.import_module(
    "scripts.export_screenqa_calibration_manifest"
    if __package__
    else "export_screenqa_calibration_manifest"
)
_manifest_module = importlib.import_module(
    "scripts.verify_screenqa_calibration_manifest"
    if __package__
    else "verify_screenqa_calibration_manifest"
)
_rollout_module = importlib.import_module(
    "scripts.verify_screenqa_calibration_rollouts"
    if __package__
    else "verify_screenqa_calibration_rollouts"
)
verify_candidate = _export_module.verify_candidate
verify_manifest = _manifest_module.verify_manifest
verify_rollouts = _rollout_module.verify_rollouts

EXPECTED_DECISIONS = 9951
EXPECTED_SOURCES = 1016
SUCCESS = "selected_non_degenerate_safe_threshold"
FAILURE = "no_non_degenerate_safe_threshold"
RISK_KEYS = (
    "selection_status",
    "selected_threshold",
    "method",
    "threshold_order",
    "constraints",
    "family_error",
    "per_step_hypothesis_count",
    "adjusted_p_cutoff",
    "min_source_call_rate",
    "min_source_utility",
    "selection_objective",
    "tested_threshold_count",
    "stopping_threshold",
    "untested_thresholds",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _require_empty(path: Path, name: str) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"{name} must remain unmaterialized")


def validate_candidate_model(candidate: Mapping[str, Any]) -> list[float]:
    expected = {
        "model_type": "multidomain_factorized_action_value",
        "training_protocol": "source_grouped_oof_domain_source_balanced_v2",
        "sample_weighting": "equal_domain_then_equal_source_then_equal_row",
        "seed": 20260831,
        "n_folds": 5,
        "lambda_cost": 0.05,
        "domains": ["screenqa"],
    }
    for key, expected_value in expected.items():
        if candidate.get(key) != expected_value:
            raise ValueError(f"ScreenQA candidate model {key} mismatch")
    if candidate.get("feature_mode") not in {
        "context-geometry",
        "spatial-context-geometry",
    }:
        raise ValueError("ScreenQA candidate feature mode is not registered")
    if candidate.get("threshold") is not None:
        raise ValueError("ScreenQA candidate threshold must be unset")
    raw_thresholds = candidate.get("threshold_grid")
    if not isinstance(raw_thresholds, list) or not raw_thresholds:
        raise ValueError("ScreenQA candidate threshold grid is missing")
    thresholds = [float(value) for value in raw_thresholds]
    if any(not math.isfinite(value) for value in thresholds) or any(
        strict <= permissive
        for strict, permissive in zip(thresholds, thresholds[1:])
    ):
        raise ValueError("ScreenQA candidate threshold grid is invalid")
    contract = candidate.get("calibration_contract")
    expected_contract = {
        "method": "fixed_sequence_bounded_mean_kl_ltt_v1",
        "threshold_order": "strict_to_permissive_descending",
        "threshold_rate_weighting": "pooled_development_decisions",
        "target_pooled_development_call_rates": [
            0.005,
            0.01,
            0.015,
            0.02,
            0.03,
            0.05,
        ],
        "constraints": [
            {"kind": "induced_harm", "limit": 0.005},
            {"kind": "net_negative_call_mass", "limit": 0.02},
        ],
        "family_error": 0.05,
        "per_step_p_cutoff": 0.025,
        "min_source_call_rate": 0.01,
        "min_source_utility": 0.001,
        "calibration_sources": EXPECTED_SOURCES,
        "calibration_decisions": EXPECTED_DECISIONS,
        "formal_sources": 1471,
        "formal_decisions": 14672,
    }
    if contract != expected_contract:
        raise ValueError("ScreenQA candidate calibration contract mismatch")
    return thresholds


def calibrate_rows(
    candidate: Mapping[str, Any],
    rows: Sequence[AcquisitionCalibrationRow],
    *,
    expected_sources: int = EXPECTED_SOURCES,
    expected_decisions: int = EXPECTED_DECISIONS,
    run_provenance: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    thresholds = validate_candidate_model(candidate)
    materialized = list(rows)
    if len(materialized) != expected_decisions:
        raise ValueError(
            f"ScreenQA calibration requires {expected_decisions} decision rows"
        )
    if len({row.source_id for row in materialized}) != expected_sources:
        raise ValueError(
            f"ScreenQA calibration requires {expected_sources} source groups"
        )
    calibration = cast(
        dict[str, Any],
        calibrate_source_risk_threshold_fixed_sequence(
            materialized,
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
        ),
    )
    if calibration.get("selection_status") not in {SUCCESS, FAILURE}:
        raise RuntimeError("ScreenQA calibration produced an unknown status")
    provenance = {} if run_provenance is None else dict(run_provenance)
    provenance.update(
        {
            "ranker_training_outcomes_used": True,
            "calibration_outcomes_used": True,
            "formal_outcomes_used": False,
            "reserve_outcomes_used": False,
            "untouched_outcomes_used": False,
        }
    )
    calibration["run"] = provenance
    calibrated_model = dict(candidate)
    calibrated_model["threshold"] = calibration["selected_threshold"]
    calibrated_model["risk_calibration"] = {
        key: calibration[key] for key in RISK_KEYS
    }
    calibrated_model["risk_calibration"]["provenance"] = provenance
    return calibration, calibrated_model


def build_result_audit(
    calibration: Mapping[str, Any],
    *,
    input_hashes: Mapping[str, str],
    code_revision: str,
) -> dict[str, Any]:
    status = calibration.get("selection_status")
    if status not in {SUCCESS, FAILURE}:
        raise ValueError("ScreenQA calibration status is invalid")
    selected_threshold = calibration.get("selected_threshold")
    if status == SUCCESS:
        if not isinstance(selected_threshold, (int, float)) or not math.isfinite(
            float(selected_threshold)
        ):
            raise ValueError("successful ScreenQA calibration lacks a threshold")
    elif selected_threshold is not None:
        raise ValueError("failed ScreenQA calibration unexpectedly selected a threshold")
    return {
        "passed": True,
        "scientific_status": (
            "fresh source-level fixed-sequence ScreenQA risk calibration; formal "
            "remains sealed unless a non-degenerate safe threshold is selected"
        ),
        "selection_status": status,
        "selected_threshold": selected_threshold,
        "formal_allowed": status == SUCCESS,
        "formal_stop_required": status != SUCCESS,
        "code_revision": code_revision,
        "inputs": dict(input_hashes),
        "calibration_sources": calibration.get("n_sources"),
        "calibration_decisions": calibration.get("n_decisions"),
        "tested_threshold_count": calibration.get("tested_threshold_count"),
        "stopping_threshold": calibration.get("stopping_threshold"),
        "ranker_training_outcomes_used": True,
        "calibration_outcomes_used": True,
        "formal_outcomes_opened": False,
        "reserve_outcomes_opened": False,
        "untouched_outcomes_opened": False,
        "official_validation_test_opened": False,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen ScreenQA fixed-sequence risk calibration"
    )
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-manifest-audit-sha256", required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--expected-rollouts-sha256", required=True)
    parser.add_argument("--merge-audit", type=Path, required=True)
    parser.add_argument("--expected-merge-audit-sha256", required=True)
    parser.add_argument("--expected-bank-code-revision", required=True)
    parser.add_argument("--formal-output-dir", type=Path, required=True)
    parser.add_argument("--reserve-output-dir", type=Path, required=True)
    parser.add_argument("--untouched-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite calibration output: {args.output_dir}")
    _require_empty(args.formal_output_dir.resolve(), "formal output")
    _require_empty(args.reserve_output_dir.resolve(), "reserve output")
    _require_empty(args.untouched_output_dir.resolve(), "untouched output")
    code_revision = os.environ.get("BE_CODE_REVISION")
    if not code_revision:
        code_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    candidate_dir = args.candidate_dir.resolve()
    manifest_dir = args.manifest_dir.resolve()
    candidate_info = verify_candidate(candidate_dir)
    manifest_info = verify_manifest(
        manifest_dir,
        candidate_dir=candidate_dir,
        expected_candidate_bundle_sha256=candidate_info["bundle_sha256"],
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_audit_sha256=args.expected_manifest_audit_sha256,
    )
    input_audit = args.output_dir / "calibration-rollouts.audit.json"
    rollout_info = verify_rollouts(
        rollouts=args.rollouts.resolve(),
        expected_rollouts_sha256=args.expected_rollouts_sha256,
        merge_audit=args.merge_audit.resolve(),
        expected_merge_audit_sha256=args.expected_merge_audit_sha256,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_bank_code_revision=args.expected_bank_code_revision,
        output=input_audit,
    )
    candidate_path = candidate_dir / "model.json"
    candidate = _load_json(candidate_path)
    records = read_jsonl(args.rollouts.resolve())
    rows = factorized_acquisition_calibration_rows(candidate, records)
    input_hashes = {
        "candidate_model_sha256": candidate_info["model_sha256"],
        "candidate_audit_sha256": candidate_info["audit_sha256"],
        "candidate_bundle_sha256": candidate_info["bundle_sha256"],
        "candidate_report_sha256": candidate_info["report_sha256"],
        "manifest_sha256": manifest_info["manifest_sha256"],
        "manifest_audit_sha256": manifest_info["audit_sha256"],
        "rollouts_sha256": rollout_info["rollouts_sha256"],
        "merge_audit_sha256": rollout_info["merge_audit_sha256"],
        "rollout_input_audit_sha256": sha256_file(input_audit),
    }
    run = {
        "code_revision": code_revision,
        "bank_code_revision": args.expected_bank_code_revision,
        **input_hashes,
        "candidate_model": str(candidate_path),
        "manifest": str((manifest_dir / "manifest.jsonl").resolve()),
        "rollouts": str(args.rollouts.resolve()),
    }
    calibration, model = calibrate_rows(
        candidate,
        rows,
        run_provenance=run,
    )
    audit = build_result_audit(
        calibration,
        input_hashes=input_hashes,
        code_revision=code_revision,
    )
    _write_json(args.output_dir / "calibration.json", calibration)
    _write_json(args.output_dir / "model.json", model)
    _write_json(args.output_dir / "calibration.audit.json", audit)
    with (args.output_dir / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        for path in sorted(args.output_dir.iterdir()):
            if path.is_file() and path.name != "SHA256SUMS":
                handle.write(f"{sha256_file(path)}  {path.name}\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
