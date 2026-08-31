#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping


_export_module = importlib.import_module(
    "scripts.export_screenqa_calibration_manifest"
    if __package__
    else "export_screenqa_calibration_manifest"
)
_calibration_module = importlib.import_module(
    "scripts.calibrate_screenqa_fixed_sequence"
    if __package__
    else "calibrate_screenqa_fixed_sequence"
)
verify_candidate = _export_module.verify_candidate
validate_candidate_model = _calibration_module.validate_candidate_model
SUCCESS = str(_calibration_module.SUCCESS)
FAILURE = str(_calibration_module.FAILURE)
RISK_KEYS = tuple(_calibration_module.RISK_KEYS)

HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
GIT_REVISION = re.compile(r"[0-9a-f]{40,64}")
EXPECTED_CONSTRAINTS = [
    {"kind": "induced_harm", "limit": 0.005},
    {"kind": "net_negative_call_mass", "limit": 0.02},
]


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


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"ScreenQA calibration result has invalid {name}")
    return value


def _finite(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"ScreenQA calibration result has invalid {name}")
    return float(value)


def verify_sha256sums(directory: Path) -> None:
    sums = directory / "SHA256SUMS"
    if not sums.is_file():
        raise FileNotFoundError(f"calibration checksum bundle is missing: {sums}")
    for line in sums.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = directory / relative.strip()
        if sha256_file(path) != expected:
            raise ValueError(f"ScreenQA calibration checksum mismatch: {path}")


def verify_result(output_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    verify_sha256sums(output_dir)
    candidate_info = verify_candidate(candidate_dir)
    candidate = _load_json(candidate_dir / "model.json")
    thresholds = validate_candidate_model(candidate)
    calibration = _load_json(output_dir / "calibration.json")
    model = _load_json(output_dir / "model.json")
    audit = _load_json(output_dir / "calibration.audit.json")

    expected_scalars = {
        "scientific_status": (
            "source-level fixed-sequence risk calibration; nested thresholds "
            "are frozen before calibration outcomes"
        ),
        "method": "fixed_sequence_bounded_mean_kl_ltt_v1",
        "threshold_order": "strict_to_permissive_descending",
        "lambda_cost": 0.05,
        "max_tool_cost": 1.0,
        "family_error": 0.05,
        "per_step_hypothesis_count": 2,
        "adjusted_p_cutoff": 0.025,
        "n_sources": 1016,
        "n_decisions": 9951,
        "constraints": EXPECTED_CONSTRAINTS,
        "min_source_call_rate": 0.01,
        "min_source_utility": 0.001,
        "selection_objective": "most_permissive_pre_failure_with_non_degeneracy",
    }
    for key, expected in expected_scalars.items():
        if calibration.get(key) != expected:
            raise ValueError(f"ScreenQA calibration scalar {key} mismatch")

    raw_candidates = calibration.get("candidates")
    raw_untested = calibration.get("untested_thresholds")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("ScreenQA calibration result has no tested thresholds")
    if not isinstance(raw_untested, list):
        raise ValueError("ScreenQA calibration untested thresholds are malformed")
    candidates = [_mapping(item, "tested threshold") for item in raw_candidates]
    tested_thresholds = [
        _finite(item.get("threshold"), "tested threshold") for item in candidates
    ]
    untested_thresholds = [
        _finite(item, "untested threshold") for item in raw_untested
    ]
    if tested_thresholds + untested_thresholds != thresholds:
        raise ValueError("ScreenQA calibration did not follow the frozen threshold order")
    if calibration.get("tested_threshold_count") != len(candidates):
        raise ValueError("ScreenQA calibration tested-threshold count mismatch")

    eligible: list[Mapping[str, Any]] = []
    first_failure: Mapping[str, Any] | None = None
    for index, item in enumerate(candidates):
        risks = _mapping(item.get("risks"), "threshold risks")
        if set(risks) != {constraint["kind"] for constraint in EXPECTED_CONSTRAINTS}:
            raise ValueError("ScreenQA calibration risk family mismatch")
        risk_passes = []
        for constraint in EXPECTED_CONSTRAINTS:
            risk = _mapping(risks[str(constraint["kind"])], "risk result")
            if risk.get("limit") != constraint["limit"]:
                raise ValueError("ScreenQA calibration risk limit mismatch")
            risk_passes.append(risk.get("passed") is True)
        risk_accepted = all(risk_passes)
        if item.get("risk_accepted") is not risk_accepted:
            raise ValueError("ScreenQA calibration joint-risk decision mismatch")
        call_rate = _finite(item.get("source_call_rate"), "source call rate")
        utility = _finite(item.get("source_utility"), "source utility")
        if not 0.0 <= call_rate <= 1.0:
            raise ValueError("ScreenQA calibration source call rate is invalid")
        if not risk_accepted:
            if index != len(candidates) - 1:
                raise ValueError("ScreenQA fixed sequence continued after first failure")
            first_failure = item
        elif call_rate >= 0.01 and utility >= 0.001:
            eligible.append(item)

    expected_stopping = (
        None if first_failure is None else float(first_failure["threshold"])
    )
    if calibration.get("stopping_threshold") != expected_stopping:
        raise ValueError("ScreenQA calibration stopping threshold mismatch")
    if first_failure is None and untested_thresholds:
        raise ValueError("ScreenQA calibration left thresholds untested without failure")
    expected_selected = eligible[-1] if eligible else None
    expected_status = SUCCESS if expected_selected is not None else FAILURE
    expected_threshold = (
        None if expected_selected is None else float(expected_selected["threshold"])
    )
    if (
        calibration.get("selection_status") != expected_status
        or calibration.get("selected_threshold") != expected_threshold
        or calibration.get("selected") != expected_selected
    ):
        raise ValueError("ScreenQA calibration selected policy mismatch")

    run = _mapping(calibration.get("run"), "run provenance")
    expected_outcome_access = {
        "ranker_training_outcomes_used": True,
        "calibration_outcomes_used": True,
        "formal_outcomes_used": False,
        "reserve_outcomes_used": False,
        "untouched_outcomes_used": False,
    }
    for key, expected in expected_outcome_access.items():
        if run.get(key) != expected:
            raise ValueError(f"ScreenQA calibration run {key} mismatch")
    if run.get("candidate_model_sha256") != candidate_info["model_sha256"]:
        raise ValueError("ScreenQA calibration run candidate hash mismatch")

    expected_risk = {key: calibration[key] for key in RISK_KEYS}
    expected_risk["provenance"] = dict(run)
    expected_model = dict(candidate)
    expected_model["threshold"] = expected_threshold
    expected_model["risk_calibration"] = expected_risk
    if model != expected_model:
        raise ValueError("ScreenQA calibrated model changed beyond threshold/risk metadata")

    if audit.get("passed") is not True:
        raise ValueError("ScreenQA calibration result audit did not pass")
    expected_audit = {
        "selection_status": expected_status,
        "selected_threshold": expected_threshold,
        "formal_allowed": expected_status == SUCCESS,
        "formal_stop_required": expected_status != SUCCESS,
        "calibration_sources": 1016,
        "calibration_decisions": 9951,
        "ranker_training_outcomes_used": True,
        "calibration_outcomes_used": True,
        "formal_outcomes_opened": False,
        "reserve_outcomes_opened": False,
        "untouched_outcomes_opened": False,
        "official_validation_test_opened": False,
    }
    for key, expected in expected_audit.items():
        if audit.get(key) != expected:
            raise ValueError(f"ScreenQA calibration audit {key} mismatch")
    code_revision = audit.get("code_revision")
    if not isinstance(code_revision, str) or GIT_REVISION.fullmatch(code_revision) is None:
        raise ValueError("ScreenQA calibration audit code revision is malformed")
    inputs = _mapping(audit.get("inputs"), "input hashes")
    required_inputs = {
        "candidate_model_sha256",
        "candidate_audit_sha256",
        "candidate_bundle_sha256",
        "candidate_report_sha256",
        "manifest_sha256",
        "manifest_audit_sha256",
        "rollouts_sha256",
        "merge_audit_sha256",
        "rollout_input_audit_sha256",
    }
    if set(inputs) != required_inputs:
        raise ValueError("ScreenQA calibration input-hash family mismatch")
    for key, value in inputs.items():
        if not isinstance(value, str) or HEX_SHA256.fullmatch(value) is None:
            raise ValueError(f"ScreenQA calibration input hash {key} is malformed")
    if inputs["candidate_model_sha256"] != candidate_info["model_sha256"]:
        raise ValueError("ScreenQA calibration audit candidate hash mismatch")
    return {
        "passed": True,
        "selection_status": expected_status,
        "selected_threshold": expected_threshold,
        "formal_allowed": expected_status == SUCCESS,
        "formal_stop_required": expected_status != SUCCESS,
        "calibration_sha256": sha256_file(output_dir / "calibration.json"),
        "model_sha256": sha256_file(output_dir / "model.json"),
        "audit_sha256": sha256_file(output_dir / "calibration.audit.json"),
        "bundle_sha256": sha256_file(output_dir / "SHA256SUMS"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute and verify the ScreenQA fixed-sequence result gate"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    args = parser.parse_args()
    result = verify_result(args.output_dir.resolve(), args.candidate_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
