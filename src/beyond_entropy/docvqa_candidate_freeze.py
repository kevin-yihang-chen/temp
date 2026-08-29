from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping

from .rescue_gate import DecisionKey
from .risk_control import threshold_grid_from_source_balanced_call_rates


PROTOCOL_SHA256 = (
    "f2fc21218085d0b2bce1c92f3a4c30e1dac78b5e813d28a03258bf28fdb06124"
)
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
_HEX_DIGEST = re.compile(r"[0-9a-f]{64}")
_GIT_REVISION = re.compile(r"[0-9a-f]{40,64}")
_PROVENANCE_PATH_KEYS = (
    "raw_model",
    "development_report",
    "development_rollouts",
    "development_features",
    "allocation",
    "allocation_audit",
    "protocol",
)
_ROLE_COUNTS = {
    "ranker_training": 3500,
    "risk_calibration": 2500,
    "formal_test": 3500,
}


def _require(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"DocVQA candidate violates frozen {name}")


def _serialized_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"


def serialized_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_serialized_json(payload).encode()).hexdigest()


def validate_raw_candidate_inputs(
    model: Mapping[str, Any],
    report: Mapping[str, Any],
) -> None:
    common_expected = {
        "model_type": "multidomain_factorized_action_value",
        "training_protocol": "source_grouped_oof_domain_source_balanced_v2",
        "sample_weighting": "equal_domain_then_equal_source_then_equal_row",
        "feature_mode": "hybrid-context-semantic",
        "seed": 20260829,
        "n_folds": 5,
        "lambda_cost": 0.05,
        "selected_alpha": 1.0,
        "domains": ["docvqa"],
    }
    for name, expected in common_expected.items():
        _require(model.get(name), expected, f"model {name}")
        _require(report.get(name), expected, f"report {name}")
    _require(model.get("state_feature_count"), 27, "model state feature count")
    _require(model.get("action_feature_count"), 46, "model action feature count")
    raw_threshold = model.get("threshold")
    if not isinstance(raw_threshold, (int, float)) or not math.isfinite(
        float(raw_threshold)
    ):
        raise ValueError("raw DocVQA factorized model has no finite OOF threshold")
    refit = report.get("refit")
    if not isinstance(refit, Mapping):
        raise ValueError("DocVQA report is missing refit metadata")
    _require(refit.get("state_feature_count"), 27, "report state feature count")
    _require(refit.get("action_feature_count"), 46, "report action feature count")
    run = report.get("run")
    if not isinstance(run, Mapping):
        raise ValueError("DocVQA report is missing run provenance")
    _require(run.get("formal_outcomes_used"), False, "formal outcome exclusion")
    development_inputs = run.get("development_inputs")
    semantic_features = run.get("semantic_features")
    if not isinstance(development_inputs, Mapping) or set(development_inputs) != {
        "docvqa"
    }:
        raise ValueError("DocVQA candidate requires exactly one development domain")
    if not isinstance(semantic_features, Mapping) or set(semantic_features) != {
        "docvqa"
    }:
        raise ValueError("DocVQA candidate requires exactly one semantic feature bank")


def validate_candidate_freeze_gate(
    allocation: Mapping[str, Any],
    allocation_audit: Mapping[str, Any],
    *,
    allocation_sha256: str,
) -> None:
    """Require the registered identity allocation before calibration materialization."""

    _require(allocation.get("protocol_sha256"), PROTOCOL_SHA256, "allocation protocol")
    contract = allocation.get("selection_contract")
    expected_contract = {
        "selection_target_fields_accessed": False,
        "selection_allowed_fields": ["docId", "image"],
        "ranker_manifest_exported": False,
        "calibration_manifest_exported": False,
        "formal_manifest_exported": False,
        "ranker_outcomes_collected": False,
        "calibration_outcomes_collected": False,
        "formal_outcomes_collected": False,
    }
    _require(contract, expected_contract, "allocation outcome-sealing contract")
    _require(allocation_audit.get("passed"), True, "allocation audit status")
    _require(
        allocation_audit.get("allocation_sha256"),
        allocation_sha256,
        "allocation audit SHA-256 binding",
    )
    _require(
        allocation_audit.get("protocol_sha256"),
        PROTOCOL_SHA256,
        "allocation audit protocol",
    )
    for name in (
        "ranker_outcomes_collected",
        "calibration_outcomes_collected",
        "formal_outcomes_collected",
    ):
        _require(allocation_audit.get(name), False, f"allocation audit {name}")

    body = allocation.get("allocation")
    if not isinstance(body, Mapping):
        raise ValueError("DocVQA allocation body is missing")
    roles = body.get("roles")
    if not isinstance(roles, Mapping) or set(roles) != set(_ROLE_COUNTS):
        raise ValueError("DocVQA allocation role set changed")
    for name, expected_count in _ROLE_COUNTS.items():
        role = roles[name]
        if not isinstance(role, Mapping):
            raise ValueError(f"DocVQA allocation role {name!r} is invalid")
        _require(role.get("count"), expected_count, f"allocation {name} count")


def _validate_provenance(provenance: Mapping[str, Any]) -> None:
    for path_key in _PROVENANCE_PATH_KEYS:
        path = str(provenance.get(path_key, "")).strip()
        digest = str(provenance.get(f"{path_key}_sha256", "")).strip()
        if not path:
            raise ValueError(f"candidate provenance is missing {path_key}")
        if _HEX_DIGEST.fullmatch(digest) is None:
            raise ValueError(f"candidate provenance has invalid {path_key} SHA-256")
    _require(
        provenance.get("protocol_sha256"),
        PROTOCOL_SHA256,
        "protocol SHA-256",
    )


def _source_balanced_call_rate(
    scores: Mapping[DecisionKey, float],
    source_by_key: Mapping[DecisionKey, str],
    threshold: float,
) -> float:
    calls_by_source: dict[str, list[float]] = {}
    for key, score in scores.items():
        source = source_by_key[key]
        calls_by_source.setdefault(source, []).append(float(score >= threshold))
    return sum(sum(values) / len(values) for values in calls_by_source.values()) / len(
        calls_by_source
    )


def build_frozen_candidate(
    model: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    scores_by_key: Mapping[DecisionKey, float],
    source_by_key: Mapping[DecisionKey, str],
    provenance: Mapping[str, Any],
    code_revision: str,
    expected_sources: int = 3500,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Remove the development gate and freeze one source-balanced LTT family."""

    validate_raw_candidate_inputs(model, report)
    _validate_provenance(provenance)
    if expected_sources <= 0:
        raise ValueError("expected source count must be positive")
    revision = str(code_revision).strip()
    if _GIT_REVISION.fullmatch(revision) is None:
        raise ValueError("candidate freeze code revision must be a Git object ID")
    if not scores_by_key or set(scores_by_key) != set(source_by_key):
        raise ValueError("candidate scores and sources must be non-empty and aligned")
    scores = {key: float(value) for key, value in scores_by_key.items()}
    sources = {key: str(value).strip() for key, value in source_by_key.items()}
    if any(not math.isfinite(value) for value in scores.values()):
        raise ValueError("candidate scores must be finite")
    if any(not source for source in sources.values()):
        raise ValueError("candidate source IDs must be non-empty")
    unique_sources = set(sources.values())
    if len(unique_sources) != expected_sources:
        raise ValueError(
            f"candidate requires {expected_sources} ranker-training sources"
        )

    ordered_keys = sorted(scores)
    thresholds = threshold_grid_from_source_balanced_call_rates(
        [scores[key] for key in ordered_keys],
        [sources[key] for key in ordered_keys],
        TARGET_CALL_RATES,
    )
    if any(left <= right for left, right in zip(thresholds, thresholds[1:])):
        raise RuntimeError("candidate threshold sequence is not strictly descending")
    threshold_summaries = [
        {
            "threshold": threshold,
            "source_balanced_development_call_rate": _source_balanced_call_rate(
                scores,
                sources,
                threshold,
            ),
            "pooled_development_call_rate": sum(
                score >= threshold for score in scores.values()
            )
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
        "threshold_rate_weighting": "equal_source_then_equal_question",
        "target_source_balanced_development_call_rates": list(TARGET_CALL_RATES),
        "threshold_summaries": threshold_summaries,
        "constraints": [
            {"kind": "induced_harm", "limit": 0.005},
            {"kind": "net_negative_call_mass", "limit": 0.02},
        ],
        "family_error": 0.05,
        "per_step_p_cutoff": 0.025,
        "min_source_call_rate": 0.01,
        "min_source_utility": 0.001,
        "calibration_sources": 2500,
        "formal_sources": 3500,
    }
    candidate["candidate_freeze"] = {
        **dict(provenance),
        "code_revision": revision,
        "ranker_training_outcomes_used": True,
        "calibration_outcomes_used": False,
        "formal_outcomes_used": False,
    }
    candidate_sha256 = serialized_sha256(candidate)
    audit = {
        "passed": True,
        "scientific_status": (
            "sole DocVQA factorized-v2 candidate frozen before calibration export"
        ),
        "candidate_sha256": candidate_sha256,
        "code_revision": revision,
        "protocol_sha256": PROTOCOL_SHA256,
        "development_decisions": len(scores),
        "development_sources": len(unique_sources),
        "threshold_count": len(thresholds),
        "thresholds": thresholds,
        "threshold_summaries": threshold_summaries,
        "ranker_training_outcomes_used": True,
        "calibration_outcomes_used": False,
        "formal_outcomes_used": False,
    }
    return candidate, audit
