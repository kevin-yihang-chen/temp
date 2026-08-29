from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .docvqa_candidate_freeze import (
    PROTOCOL_SHA256,
    validate_candidate_freeze_gate,
)
from .risk_control import (
    AcquisitionCalibrationRow,
    RiskConstraint,
    calibrate_source_risk_threshold_fixed_sequence,
)


EXPECTED_CONSTRAINTS = [
    {"kind": "induced_harm", "limit": 0.005},
    {"kind": "net_negative_call_mass", "limit": 0.02},
]
MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
EXPECTED_SCIENTIFIC_STATUS = (
    "fresh DocVQA-train factorized-v2 calibration sibling bank; outcomes may "
    "calibrate the sole frozen candidate only"
)


def _require(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"DocVQA calibration contract mismatch for {name}")


def validate_frozen_candidate(candidate: Mapping[str, Any]) -> list[float]:
    expected = {
        "model_type": "multidomain_factorized_action_value",
        "training_protocol": "source_grouped_oof_domain_source_balanced_v2",
        "sample_weighting": "equal_domain_then_equal_source_then_equal_row",
        "feature_mode": "hybrid-context-semantic",
        "seed": 20260829,
        "n_folds": 5,
        "lambda_cost": 0.05,
        "selected_alpha": 1.0,
        "domains": ["docvqa"],
        "state_feature_count": 27,
        "action_feature_count": 46,
    }
    for name, value in expected.items():
        _require(candidate.get(name), value, f"candidate {name}")
    _require(candidate.get("threshold"), None, "uncalibrated threshold")
    raw_thresholds = candidate.get("threshold_grid")
    if not isinstance(raw_thresholds, list) or any(
        not isinstance(value, (int, float)) for value in raw_thresholds
    ):
        raise ValueError("DocVQA candidate threshold grid is invalid")
    thresholds = [float(value) for value in raw_thresholds]
    if (
        not 2 <= len(thresholds) <= 11
        or any(not math.isfinite(value) for value in thresholds)
        or any(left <= right for left, right in zip(thresholds, thresholds[1:]))
    ):
        raise ValueError("DocVQA candidate threshold grid is invalid")
    contract = candidate.get("calibration_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("DocVQA candidate is missing its calibration contract")
    expected_contract = {
        "method": "fixed_sequence_bounded_mean_kl_ltt_v1",
        "threshold_order": "strict_to_permissive_descending",
        "threshold_rate_weighting": "equal_source_then_equal_question",
        "constraints": EXPECTED_CONSTRAINTS,
        "family_error": 0.05,
        "per_step_p_cutoff": 0.025,
        "min_source_call_rate": 0.01,
        "min_source_utility": 0.001,
        "calibration_sources": 2500,
        "formal_sources": 3500,
    }
    for name, value in expected_contract.items():
        _require(contract.get(name), value, f"candidate calibration {name}")
    summaries = contract.get("threshold_summaries")
    if not isinstance(summaries, list) or len(summaries) != len(thresholds):
        raise ValueError("DocVQA threshold summaries do not match the grid")
    summary_thresholds: list[float] = []
    for summary in summaries:
        if not isinstance(summary, Mapping):
            raise ValueError("DocVQA threshold summary must be a mapping")
        value = summary.get("threshold")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("DocVQA threshold summary has an invalid threshold")
        summary_thresholds.append(float(value))
    if summary_thresholds != thresholds:
        raise ValueError("DocVQA threshold summaries changed order or values")
    freeze = candidate.get("candidate_freeze")
    if not isinstance(freeze, Mapping):
        raise ValueError("DocVQA candidate is missing freeze provenance")
    _require(
        freeze.get("ranker_training_outcomes_used"),
        True,
        "ranker outcome disclosure",
    )
    _require(
        freeze.get("calibration_outcomes_used"),
        False,
        "calibration outcome exclusion",
    )
    _require(
        freeze.get("formal_outcomes_used"),
        False,
        "formal outcome exclusion",
    )
    return thresholds


def validate_calibration_preoutcome_gate(
    candidate: Mapping[str, Any],
    candidate_audit: Mapping[str, Any],
    allocation: Mapping[str, Any],
    allocation_audit: Mapping[str, Any],
    *,
    candidate_sha256: str,
    allocation_sha256: str,
    code_revision: str,
) -> None:
    """Validate frozen candidate and allocation metadata before outcome access."""

    thresholds = validate_frozen_candidate(candidate)
    freeze = candidate.get("candidate_freeze")
    assert isinstance(freeze, Mapping)
    _require(freeze.get("protocol_sha256"), PROTOCOL_SHA256, "candidate protocol")
    _require(
        freeze.get("allocation_sha256"),
        allocation_sha256,
        "candidate allocation SHA-256 binding",
    )
    _require(freeze.get("code_revision"), code_revision, "candidate code revision")
    _require(candidate_audit.get("passed"), True, "candidate audit status")
    _require(
        candidate_audit.get("scientific_status"),
        "sole DocVQA factorized-v2 candidate frozen before calibration export",
        "candidate audit scientific status",
    )
    _require(
        candidate_audit.get("candidate_sha256"),
        candidate_sha256,
        "candidate audit SHA-256 binding",
    )
    _require(
        candidate_audit.get("protocol_sha256"),
        PROTOCOL_SHA256,
        "candidate audit protocol",
    )
    _require(
        candidate_audit.get("development_sources"),
        3500,
        "candidate development sources",
    )
    _require(
        candidate_audit.get("threshold_count"),
        len(thresholds),
        "candidate threshold count",
    )
    _require(
        candidate_audit.get("ranker_training_outcomes_used"),
        True,
        "candidate ranker outcome disclosure",
    )
    _require(
        candidate_audit.get("calibration_outcomes_used"),
        False,
        "candidate calibration outcome exclusion",
    )
    _require(
        candidate_audit.get("formal_outcomes_used"),
        False,
        "candidate formal outcome exclusion",
    )
    _require(
        candidate_audit.get("code_revision"),
        code_revision,
        "candidate code revision",
    )
    validate_candidate_freeze_gate(
        allocation,
        allocation_audit,
        allocation_sha256=allocation_sha256,
    )


def validate_calibration_manifest(
    manifest_audit: Mapping[str, Any],
    *,
    candidate_sha256: str,
    allocation_sha256: str,
    manifest_sha256: str,
) -> int:
    """Validate the outcome-independent calibration manifest selection."""

    expected_manifest = {
        "task": "docvqa",
        "dataset_id": "lmms-lab/DocVQA",
        "dataset_name": "DocVQA",
        "dataset_revision": "539088ef8a8ada01ac8e2e6d4e372586748a265e",
        "split": "train",
        "scorer": "docvqa",
        "unique_sources": 2500,
        "unique_images": 2500,
        "manifest_sha256": manifest_sha256,
    }
    for name, value in expected_manifest.items():
        _require(manifest_audit.get(name), value, f"manifest {name}")
    manifest_count = manifest_audit.get("count")
    if not isinstance(manifest_count, int) or manifest_count < 2500:
        raise ValueError("DocVQA calibration manifest count is invalid")
    _require(
        manifest_audit.get("unique_states"),
        manifest_count,
        "manifest unique states",
    )
    selection = manifest_audit.get("selection_metadata")
    if not isinstance(selection, Mapping):
        raise ValueError("DocVQA calibration manifest lacks selection metadata")
    expected_selection = {
        "allocation_sha256": allocation_sha256,
        "candidate_sha256": candidate_sha256,
        "protocol_sha256": PROTOCOL_SHA256,
        "namespace": "beyond-entropy-docvqa-train-factorized-v2",
        "role": "risk_calibration",
        "selected_source_group_count": 2500,
        "selection_uses_targets": False,
    }
    for name, value in expected_selection.items():
        _require(selection.get(name), value, f"manifest selection {name}")
    return manifest_count


def validate_calibration_rollout_audit(
    rollout_audit: Mapping[str, Any],
    *,
    manifest_count: int,
    candidate_sha256: str,
    candidate_audit_sha256: str,
    allocation_sha256: str,
    allocation_audit_sha256: str,
    manifest_sha256: str,
    manifest_provenance_sha256: str,
    rollouts_sha256: str,
    code_revision: str,
) -> None:
    """Validate a complete five-sibling calibration rollout audit."""

    expected_rollout = {
        "passed": True,
        "manifest_sha256": manifest_sha256,
        "manifest_provenance_sha256": manifest_provenance_sha256,
        "rollouts_sha256": rollouts_sha256,
        "model_revision": MODEL_REVISION,
        "code_revision": code_revision,
        "scientific_status": EXPECTED_SCIENTIFIC_STATUS,
        "candidate_sha256": candidate_sha256,
        "candidate_audit_sha256": candidate_audit_sha256,
        "allocation_sha256": allocation_sha256,
        "allocation_audit_sha256": allocation_audit_sha256,
        "protocol_sha256": PROTOCOL_SHA256,
        "states": manifest_count,
        "records": manifest_count * 5,
        "unique_sources": 2500,
        "unique_images": 2500,
        "candidate_count": 4,
        "answer_records": manifest_count,
        "zoom_records": manifest_count * 4,
    }
    for name, value in expected_rollout.items():
        _require(rollout_audit.get(name), value, f"rollout audit {name}")


def validate_calibration_feature_metadata(
    features: Mapping[str, Any],
    *,
    rollouts_sha256: str,
    code_revision: str,
    expected_decisions: int,
) -> None:
    """Bind all three label-free feature stages to the audited rollout bank."""

    decisions = features.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != expected_decisions:
        raise ValueError("DocVQA calibration feature decision count mismatch")
    metadata = features.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("DocVQA calibration features lack metadata")
    expected_root = {
        "source_rollouts_sha256": rollouts_sha256,
        "model_revision": MODEL_REVISION,
        "code_revision": code_revision,
        "outcomes_included": False,
    }
    for name, expected in expected_root.items():
        _require(metadata.get(name), expected, f"feature metadata {name}")
    for stage in ("question_reembedding", "question_region_attention"):
        stage_metadata = metadata.get(stage)
        if not isinstance(stage_metadata, Mapping):
            raise ValueError(f"DocVQA calibration features lack {stage} metadata")
        expected_stage = {
            "source_rollouts_sha256": rollouts_sha256,
            "model_revision": MODEL_REVISION,
            "code_revision": code_revision,
            "completed_decisions": expected_decisions,
            "total_decisions": expected_decisions,
        }
        for name, expected in expected_stage.items():
            _require(
                stage_metadata.get(name),
                expected,
                f"feature {stage} {name}",
            )


def validate_calibration_bundle(
    candidate: Mapping[str, Any],
    candidate_audit: Mapping[str, Any],
    allocation: Mapping[str, Any],
    allocation_audit: Mapping[str, Any],
    manifest_audit: Mapping[str, Any],
    rollout_audit: Mapping[str, Any],
    *,
    candidate_sha256: str,
    candidate_audit_sha256: str,
    allocation_sha256: str,
    allocation_audit_sha256: str,
    manifest_sha256: str,
    manifest_provenance_sha256: str,
    rollouts_sha256: str,
    code_revision: str,
) -> int:
    """Validate every frozen gate before reading calibration outcomes."""

    validate_calibration_preoutcome_gate(
        candidate,
        candidate_audit,
        allocation,
        allocation_audit,
        candidate_sha256=candidate_sha256,
        allocation_sha256=allocation_sha256,
        code_revision=code_revision,
    )
    manifest_count = validate_calibration_manifest(
        manifest_audit,
        candidate_sha256=candidate_sha256,
        allocation_sha256=allocation_sha256,
        manifest_sha256=manifest_sha256,
    )
    validate_calibration_rollout_audit(
        rollout_audit,
        manifest_count=manifest_count,
        candidate_sha256=candidate_sha256,
        candidate_audit_sha256=candidate_audit_sha256,
        allocation_sha256=allocation_sha256,
        allocation_audit_sha256=allocation_audit_sha256,
        manifest_sha256=manifest_sha256,
        manifest_provenance_sha256=manifest_provenance_sha256,
        rollouts_sha256=rollouts_sha256,
        code_revision=code_revision,
    )
    return manifest_count


def calibrate_frozen_candidate_rows(
    candidate: Mapping[str, Any],
    rows: Sequence[AcquisitionCalibrationRow],
    *,
    expected_sources: int = 2500,
    run_provenance: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the preregistered DocVQA fixed sequence to labeled source rows."""

    thresholds = validate_frozen_candidate(candidate)
    if expected_sources <= 0:
        raise ValueError("expected calibration source count must be positive")
    materialized_rows = list(rows)
    if not materialized_rows:
        raise ValueError("DocVQA calibration rows must not be empty")
    source_count = len({row.source_id for row in materialized_rows})
    if source_count != expected_sources:
        raise ValueError(
            f"DocVQA calibration requires {expected_sources} source groups"
        )
    calibration = calibrate_source_risk_threshold_fixed_sequence(
        materialized_rows,
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
    provenance = {} if run_provenance is None else dict(run_provenance)
    provenance["ranker_training_outcomes_used"] = True
    provenance["calibration_outcomes_used"] = True
    provenance["formal_outcomes_used"] = False
    calibration["run"] = provenance

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
    calibrated_model["risk_calibration"]["provenance"] = provenance
    return calibration, calibrated_model
