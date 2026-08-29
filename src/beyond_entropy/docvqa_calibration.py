from __future__ import annotations

import math
import re
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
SUCCESS = "selected_non_degenerate_safe_threshold"
FAILURE = "no_non_degenerate_safe_threshold"
_HEX_DIGEST = re.compile(r"[0-9a-f]{64}")
_RISK_KEYS = (
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


def _require(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"DocVQA calibration contract mismatch for {name}")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"DocVQA calibration result has invalid {name}")
    return value


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"DocVQA calibration result has non-finite {name}")
    return number


def _sha256(value: Any, name: str) -> str:
    digest = str(value)
    if _HEX_DIGEST.fullmatch(digest) is None:
        raise ValueError(f"DocVQA calibration result has invalid {name}")
    return digest


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


def validate_docvqa_calibration_result(
    calibration: Mapping[str, Any],
    model: Mapping[str, Any],
    *,
    expected_decisions: int | None = None,
) -> str:
    """Recompute the DocVQA fixed-sequence decision and model embedding."""

    uncalibrated = dict(model)
    uncalibrated["threshold"] = None
    thresholds = validate_frozen_candidate(uncalibrated)
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
        "n_sources": 2500,
        "constraints": EXPECTED_CONSTRAINTS,
        "min_source_call_rate": 0.01,
        "min_source_utility": 0.001,
        "selection_objective": "most_permissive_pre_failure_with_non_degeneracy",
    }
    for name, expected in expected_scalars.items():
        _require(calibration.get(name), expected, f"result {name}")
    n_decisions = calibration.get("n_decisions")
    if not isinstance(n_decisions, int) or n_decisions < 2500:
        raise ValueError("DocVQA calibration result has invalid decision count")
    if expected_decisions is not None:
        _require(n_decisions, expected_decisions, "result decision count")

    run = _mapping(calibration.get("run"), "run provenance")
    _require(run.get("protocol_sha256"), PROTOCOL_SHA256, "run protocol")
    for name in (
        "candidate_sha256",
        "candidate_audit_sha256",
        "allocation_sha256",
        "allocation_audit_sha256",
        "manifest_sha256",
        "manifest_provenance_sha256",
        "rollouts_sha256",
        "rollout_audit_sha256",
        "features_sha256",
        "protocol_sha256",
    ):
        _sha256(run.get(name), f"run {name}")
    _require(
        uncalibrated["candidate_freeze"].get("allocation_sha256"),
        run.get("allocation_sha256"),
        "candidate/run allocation binding",
    )
    _require(run.get("ranker_training_outcomes_used"), True, "ranker outcomes")
    _require(run.get("calibration_outcomes_used"), True, "calibration outcomes")
    _require(run.get("formal_outcomes_used"), False, "formal outcome exclusion")

    raw_candidates = calibration.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("DocVQA calibration result has no tested candidates")
    candidates = [_mapping(value, "candidate") for value in raw_candidates]
    _require(
        calibration.get("tested_threshold_count"),
        len(candidates),
        "tested threshold count",
    )
    raw_untested = calibration.get("untested_thresholds")
    if not isinstance(raw_untested, list):
        raise ValueError("DocVQA calibration result has invalid untested thresholds")
    tested_thresholds = [
        _finite(candidate.get("threshold"), "candidate threshold")
        for candidate in candidates
    ]
    untested = [_finite(value, "untested threshold") for value in raw_untested]
    _require(tested_thresholds + untested, thresholds, "frozen threshold sequence")

    decisions: list[tuple[bool, bool]] = []
    for candidate in candidates:
        call_rate = _finite(candidate.get("source_call_rate"), "source call rate")
        utility = _finite(candidate.get("source_utility"), "source utility")
        if not 0.0 <= call_rate <= 1.0:
            raise ValueError("DocVQA calibration result has invalid call rate")
        risks = _mapping(candidate.get("risks"), "candidate risks")
        _require(set(risks), {item["kind"] for item in EXPECTED_CONSTRAINTS}, "risks")
        risk_passes = []
        for constraint in EXPECTED_CONSTRAINTS:
            name = str(constraint["kind"])
            risk = _mapping(risks[name], f"{name} risk")
            _require(risk.get("limit"), constraint["limit"], f"{name} limit")
            upper_bound = _finite(risk.get("upper_bound"), f"{name} upper bound")
            risk_mean = _finite(
                risk.get("source_balanced_mean"),
                f"{name} source-balanced mean",
            )
            p_value = _finite(risk.get("p_value"), f"{name} p-value")
            if upper_bound <= 0.0 or not 0.0 <= risk_mean <= upper_bound:
                raise ValueError(f"DocVQA calibration result has invalid {name} mean")
            if not 0.0 <= p_value <= 1.0:
                raise ValueError(f"DocVQA calibration result has invalid {name} p-value")
            passed = p_value <= 0.025
            _require(risk.get("passed"), passed, f"{name} pass decision")
            risk_passes.append(passed)
        accepted = all(risk_passes)
        _require(candidate.get("risk_accepted"), accepted, "joint risk decision")
        decisions.append((accepted, accepted and call_rate >= 0.01 and utility >= 0.001))

    first_failure = next(
        (index for index, (accepted, _) in enumerate(decisions) if not accepted),
        None,
    )
    if first_failure is None:
        _require(calibration.get("stopping_threshold"), None, "stopping threshold")
        _require(untested, [], "untested thresholds")
    else:
        _require(first_failure, len(candidates) - 1, "fixed-sequence stopping index")
        _require(
            calibration.get("stopping_threshold"),
            tested_thresholds[-1],
            "stopping threshold",
        )
        _require(untested, thresholds[len(candidates) :], "untested thresholds")

    eligible = [
        candidate
        for candidate, (_, nondegenerate) in zip(candidates, decisions)
        if nondegenerate
    ]
    selected = eligible[-1] if eligible else None
    status = SUCCESS if selected is not None else FAILURE
    selected_threshold = (
        float(selected["threshold"]) if selected is not None else None
    )
    _require(calibration.get("selection_status"), status, "selection status")
    _require(calibration.get("selected"), selected, "selected candidate")
    _require(
        calibration.get("selected_threshold"),
        selected_threshold,
        "selected threshold",
    )
    answer_now = _mapping(calibration.get("answer_now"), "answer-now baseline")
    expected_answer_now = {
        "threshold": None,
        "answer_now_only": True,
        "source_call_rate": 0.0,
        "source_utility": 0.0,
    }
    for name, expected in expected_answer_now.items():
        _require(answer_now.get(name), expected, f"answer-now {name}")
    _require(model.get("threshold"), selected_threshold, "model threshold")
    risk_calibration = _mapping(model.get("risk_calibration"), "model risk calibration")
    expected_risk = {name: calibration[name] for name in _RISK_KEYS}
    expected_risk["provenance"] = calibration["run"]
    _require(dict(risk_calibration), expected_risk, "embedded model calibration")
    return status


def validate_docvqa_calibration_artifact_bundle(
    calibration: Mapping[str, Any],
    model: Mapping[str, Any],
    audit: Mapping[str, Any],
    *,
    calibration_sha256: str,
    model_sha256: str,
) -> str:
    """Validate calibration/model/audit correspondence for rendering."""

    status = validate_docvqa_calibration_result(
        calibration,
        model,
        expected_decisions=int(audit.get("n_decisions", -1)),
    )
    run = _mapping(calibration.get("run"), "run provenance")
    expected_audit = {
        "passed": True,
        "scientific_status": (
            "DocVQA-train fixed sequence executed once; formal role remains sealed"
        ),
        "selection_status": status,
        "selected_threshold": calibration.get("selected_threshold"),
        "tested_threshold_count": calibration.get("tested_threshold_count"),
        "stopping_threshold": calibration.get("stopping_threshold"),
        "n_sources": 2500,
        "n_decisions": calibration.get("n_decisions"),
        "calibration_sha256": calibration_sha256,
        "model_sha256": model_sha256,
        "code_revision": run.get("code_revision"),
        "ranker_training_outcomes_used": True,
        "calibration_outcomes_used": True,
        "formal_outcomes_used": False,
    }
    for name, expected in expected_audit.items():
        _require(audit.get(name), expected, f"artifact audit {name}")
    inputs = _mapping(audit.get("inputs"), "artifact audit inputs")
    for name in (
        "candidate",
        "candidate_audit",
        "allocation",
        "allocation_audit",
        "manifest",
        "manifest_provenance",
        "rollouts",
        "rollout_audit",
        "features",
        "protocol",
    ):
        _require(
            inputs.get(f"{name}_sha256"),
            run.get(f"{name}_sha256"),
            f"artifact audit {name} hash",
        )
    return status
