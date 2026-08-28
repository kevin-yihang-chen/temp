from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

from beyond_entropy.action_value import factorized_acquisition_calibration_rows
from beyond_entropy.dataset import read_jsonl
from beyond_entropy.qwen_semantic import (
    load_semantic_feature_dataset,
    validate_semantic_feature_dataset,
)
from beyond_entropy.risk_control import (
    RiskConstraint,
    calibrate_source_risk_threshold_fixed_sequence,
)


CANDIDATE_SHA256 = (
    "9a6c9d032ebdbc271b7d3c829fbb3d6ff167cac01b54ce75adc8da86e3063342"
)
ALLOCATION_SHA256 = (
    "bc0ecb4b6f49a5b0e92b90b4c30620f72246722370d59c8078753d5846f5e9b6"
)
ALLOCATION_AUDIT_SHA256 = (
    "f01f853a7de7774466be55c012b7e174f57f4ac120ed58a0bf3984e71252b5c3"
)
MANIFEST_SHA256 = (
    "0db79580d7bb96794901703a6ec0bfc0ae14e31159ddde5664762aa0351b323a"
)
MANIFEST_PROVENANCE_SHA256 = (
    "3cf60f8474c10bc81b83b5cf47ef22224b010154b0933c2ffb00bec7225e0c45"
)
PROTOCOL_SHA256 = (
    "babf01d4090263d1cfcb28c42f86f7b13ae9de4bb6bab0ca10d6e4707f02e2ca"
)
EXPECTED_SOURCES = 3000
EXPECTED_DECISIONS = 4747
EXPECTED_MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
EXPECTED_SCIENTIFIC_STATUS = (
    "fresh factorized TextVQA fixed-sequence calibration sibling bank; outcomes "
    "may calibrate the sole frozen candidate only"
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
        raise ValueError(f"fixed-sequence calibration contract mismatch for {name}")


def _validate_candidate(candidate: Mapping[str, Any]) -> list[float]:
    expected = {
        "model_type": "multidomain_factorized_action_value",
        "training_protocol": "source_grouped_oof_domain_source_balanced_v2",
        "sample_weighting": "equal_domain_then_equal_source_then_equal_row",
        "feature_mode": "hybrid-context-semantic",
        "seed": 20260828,
        "n_folds": 5,
        "lambda_cost": 0.05,
        "selected_alpha": 1.0,
        "domains": ["textvqa"],
        "state_feature_count": 27,
        "action_feature_count": 46,
    }
    for name, value in expected.items():
        _require(candidate.get(name), value, f"candidate {name}")
    _require(candidate.get("threshold"), None, "uncalibrated threshold")
    thresholds = [float(value) for value in candidate.get("threshold_grid", [])]
    if len(thresholds) != 11 or any(
        left <= right for left, right in zip(thresholds, thresholds[1:])
    ):
        raise ValueError("candidate threshold grid is not strictly descending")
    contract = candidate.get("calibration_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("candidate is missing its calibration contract")
    expected_contract = {
        "method": "fixed_sequence_bounded_mean_kl_ltt_v1",
        "threshold_order": "strict_to_permissive_descending",
        "constraints": [
            {"kind": "induced_harm", "limit": 0.005},
            {"kind": "net_negative_call_mass", "limit": 0.02},
        ],
        "family_error": 0.05,
        "per_step_p_cutoff": 0.025,
        "min_source_call_rate": 0.01,
        "min_source_utility": 0.001,
        "calibration_sources": EXPECTED_SOURCES,
        "formal_sources": 5953,
    }
    for name, value in expected_contract.items():
        _require(contract.get(name), value, f"candidate calibration {name}")
    freeze = candidate.get("candidate_freeze")
    if not isinstance(freeze, Mapping):
        raise ValueError("candidate is missing freeze provenance")
    _require(freeze.get("protocol_sha256"), PROTOCOL_SHA256, "protocol hash")
    _require(
        freeze.get("calibration_outcomes_used"),
        False,
        "candidate calibration outcome exclusion",
    )
    _require(
        freeze.get("formal_outcomes_used"),
        False,
        "candidate formal outcome exclusion",
    )
    return thresholds


def _validate_allocation(
    allocation: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> None:
    _require(allocation.get("candidate_sha256"), CANDIDATE_SHA256, "allocation candidate")
    _require(allocation.get("protocol_sha256"), PROTOCOL_SHA256, "allocation protocol")
    contract = allocation.get("selection_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("allocation is missing its selection contract")
    expected_contract = {
        "selection_target_fields_accessed": False,
        "calibration_manifest_exported": True,
        "calibration_targets_materialized_after_selection": True,
        "calibration_outcomes_collected": False,
        "formal_manifest_exported": False,
        "formal_rollouts_collected": False,
    }
    for name, value in expected_contract.items():
        _require(contract.get(name), value, f"allocation {name}")
    body = allocation.get("allocation")
    if not isinstance(body, Mapping):
        raise ValueError("allocation is missing role assignments")
    roles = body.get("roles")
    if not isinstance(roles, Mapping):
        raise ValueError("allocation roles are invalid")
    calibration = roles.get("risk_calibration")
    formal = roles.get("formal_test")
    if not isinstance(calibration, Mapping) or not isinstance(formal, Mapping):
        raise ValueError("allocation is missing calibration or formal roles")
    _require(calibration.get("offset"), 13000, "calibration offset")
    _require(calibration.get("count"), EXPECTED_SOURCES, "calibration sources")
    _require(formal.get("offset"), 16000, "formal offset")
    _require(formal.get("count"), 5953, "formal sources")
    _require(audit.get("passed"), True, "allocation audit")
    _require(
        audit.get("calibration_outcomes_collected"),
        False,
        "allocation calibration outcome exclusion",
    )
    _require(
        audit.get("formal_outcomes_collected"),
        False,
        "allocation formal outcome exclusion",
    )


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite calibration output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale calibration temporary exists: {temporary}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen factorized fixed-sequence calibration"
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--allocation-audit", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-provenance", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--expected-rollouts-sha256", required=True)
    parser.add_argument("--rollout-audit", type=Path, required=True)
    parser.add_argument("--expected-rollout-audit-sha256", required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--expected-features-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    code_revision = os.environ.get("BE_CODE_REVISION")
    if not code_revision:
        code_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    frozen_inputs = (
        (args.candidate, CANDIDATE_SHA256, "candidate"),
        (args.allocation, ALLOCATION_SHA256, "allocation"),
        (args.allocation_audit, ALLOCATION_AUDIT_SHA256, "allocation audit"),
        (args.manifest, MANIFEST_SHA256, "manifest"),
        (
            args.manifest_provenance,
            MANIFEST_PROVENANCE_SHA256,
            "manifest provenance",
        ),
        (args.rollouts, args.expected_rollouts_sha256, "rollouts"),
        (
            args.rollout_audit,
            args.expected_rollout_audit_sha256,
            "rollout audit",
        ),
        (args.features, args.expected_features_sha256, "features"),
        (args.protocol, PROTOCOL_SHA256, "protocol"),
    )
    input_hashes = {}
    for path, expected, name in frozen_inputs:
        if not path.is_file():
            raise FileNotFoundError(f"calibration input does not exist: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"{name} SHA-256 mismatch")
        input_hashes[name] = actual

    candidate = _load_mapping(args.candidate, "candidate")
    thresholds = _validate_candidate(candidate)
    allocation = _load_mapping(args.allocation, "allocation")
    allocation_audit = _load_mapping(args.allocation_audit, "allocation audit")
    _validate_allocation(allocation, allocation_audit)
    rollout_audit = _load_mapping(args.rollout_audit, "rollout audit")
    _require(rollout_audit.get("passed"), True, "rollout audit status")
    _require(rollout_audit.get("manifest_sha256"), MANIFEST_SHA256, "rollout manifest")
    _require(rollout_audit.get("code_revision"), code_revision, "rollout code revision")
    _require(
        rollout_audit.get("model_revision"),
        EXPECTED_MODEL_REVISION,
        "rollout model revision",
    )
    _require(
        rollout_audit.get("scientific_status"),
        EXPECTED_SCIENTIFIC_STATUS,
        "rollout scientific status",
    )
    _require(rollout_audit.get("states"), EXPECTED_DECISIONS, "rollout states")
    _require(
        rollout_audit.get("records"),
        EXPECTED_DECISIONS * 5,
        "rollout records",
    )
    _require(rollout_audit.get("unique_sources"), EXPECTED_SOURCES, "rollout sources")
    _require(rollout_audit.get("unique_images"), EXPECTED_SOURCES, "rollout images")
    _require(rollout_audit.get("candidate_count"), 4, "rollout candidates")
    _require(
        rollout_audit.get("answer_records"), EXPECTED_DECISIONS, "answer records"
    )
    _require(
        rollout_audit.get("zoom_records"),
        EXPECTED_DECISIONS * 4,
        "zoom records",
    )

    records = read_jsonl(args.rollouts)
    if len(records) != EXPECTED_DECISIONS * 5:
        raise ValueError("calibration rollouts do not contain five siblings per decision")
    features = load_semantic_feature_dataset(args.features)
    validate_semantic_feature_dataset(features, records)
    if bool(features["metadata"].get("outcomes_included", True)):
        raise ValueError("calibration requires label-free semantic features")
    semantic_decisions = {
        (str(decision["state_id"]), str(decision["replicate_id"])): decision
        for decision in features["decisions"]
    }
    rows = factorized_acquisition_calibration_rows(
        candidate,
        records,
        semantic_decisions=semantic_decisions,
    )
    if len(rows) != EXPECTED_DECISIONS:
        raise RuntimeError("calibration rows do not cover every decision")
    if len({row.source_id for row in rows}) != EXPECTED_SOURCES:
        raise RuntimeError("calibration rows do not cover exactly 3,000 sources")

    calibration = calibrate_source_risk_threshold_fixed_sequence(
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
    )
    run = {
        "code_revision": code_revision,
        "candidate": str(args.candidate.resolve()),
        "candidate_sha256": CANDIDATE_SHA256,
        "allocation": str(args.allocation.resolve()),
        "allocation_sha256": ALLOCATION_SHA256,
        "allocation_audit_sha256": ALLOCATION_AUDIT_SHA256,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": MANIFEST_SHA256,
        "rollouts": str(args.rollouts.resolve()),
        "rollouts_sha256": input_hashes["rollouts"],
        "rollout_audit_sha256": input_hashes["rollout audit"],
        "features": str(args.features.resolve()),
        "features_sha256": input_hashes["features"],
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": PROTOCOL_SHA256,
        "formal_outcomes_used": False,
    }
    calibration["run"] = run
    calibrated_model = dict(candidate)
    calibrated_model["threshold"] = calibration["selected_threshold"]
    calibrated_model["risk_calibration"] = {
        key: calibration[key]
        for key in (
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
    }
    calibrated_model["risk_calibration"]["provenance"] = run

    _write_atomic(args.output_dir / "calibration.json", calibration)
    _write_atomic(args.output_dir / "model.json", calibrated_model)
    print(
        json.dumps(
            {
                "selection_status": calibration["selection_status"],
                "selected_threshold": calibration["selected_threshold"],
                "tested_threshold_count": calibration["tested_threshold_count"],
                "stopping_threshold": calibration["stopping_threshold"],
                "n_sources": calibration["n_sources"],
                "n_decisions": calibration["n_decisions"],
                "formal_outcomes_used": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
