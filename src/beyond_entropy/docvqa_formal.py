from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Mapping

from .docvqa_calibration import SUCCESS
from .docvqa_candidate_freeze import PROTOCOL_SHA256
from .docvqa_train_allocation import sha256_file


FORMAL_SOURCES = 3500
CALIBRATION_SOURCES = 2500
BOOTSTRAP_RESAMPLES = 20000
BOOTSTRAP_CONFIDENCE = 0.975
BOOTSTRAP_SEED = 20260829
MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
FORMAL_SCIENTIFIC_STATUS = (
    "one-shot DocVQA-train factorized-v2 formal sibling bank; frozen policy and "
    "implementation; no target-derived tuning"
)
POLICY_FREEZE_SCIENTIFIC_STATUS = (
    "successful fresh DocVQA-train fixed-sequence calibration; exact factorized "
    "policy and complete formal implementation frozen before formal manifest export"
)
REQUIRED_ARTIFACTS = frozenset(
    {
        "allocation",
        "allocation_audit",
        "calibrated_model",
        "calibration_audit",
        "calibration_features",
        "calibration_label_free_audit",
        "calibration_manifest",
        "calibration_manifest_audit",
        "calibration_manifest_provenance",
        "calibration_report",
        "calibration_rollout_audit",
        "calibration_rollouts",
        "candidate",
        "candidate_audit",
        "oof_report",
        "protocol",
        "ranker_features",
        "ranker_label_free_audit",
        "ranker_manifest",
        "ranker_manifest_audit",
        "ranker_manifest_provenance",
        "ranker_rollout_audit",
        "ranker_rollouts",
    }
)
REQUIRED_IMPLEMENTATION = frozenset(
    {
        "action_value",
        "allocation_script",
        "allocation_verifier",
        "attention_extraction",
        "calibration_renderer",
        "calibration_rollout_audit_script",
        "calibration_script",
        "candidate_freeze_script",
        "context_reembedding",
        "development_feature_job",
        "development_fit_job",
        "development_manifest_export",
        "development_manifest_verifier",
        "development_rollout_job",
        "docvqa_allocation_contract",
        "docvqa_calibration_contract",
        "docvqa_candidate_contract",
        "docvqa_formal_contract",
        "docvqa_formal_export_contract",
        "docvqa_manifest_contract",
        "factorized_evaluation",
        "formal_evaluation_job",
        "formal_evaluator_script",
        "formal_export_job",
        "formal_export_script",
        "formal_feature_job",
        "formal_gate_verifier",
        "formal_renderer",
        "formal_rollout_job",
        "formal_export_submission",
        "formal_submission",
        "label_free_audit",
        "manifest_audit",
        "manifest_export",
        "policy_freeze_script",
        "qwen_backend",
        "qwen_semantic",
        "risk_control",
        "rollout",
        "rollout_audit",
    }
)
_GIT_REVISION = re.compile(r"[0-9a-f]{40}")


def _require(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"DocVQA formal contract mismatch for {name}")


def check_hash(path: Path, expected: str, name: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"DocVQA formal {name} does not exist: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"DocVQA formal {name} SHA-256 mismatch")
    return actual


def validate_policy_freeze(
    freeze: Mapping[str, Any],
    *,
    verify_components: bool = True,
) -> None:
    """Validate a successful pre-formal policy and implementation freeze."""

    _require(freeze.get("schema_version"), 1, "freeze schema")
    _require(
        freeze.get("scientific_status"),
        POLICY_FREEZE_SCIENTIFIC_STATUS,
        "scientific status",
    )
    _require(
        freeze.get("formal_gate_status"),
        "ready_for_formal_manifest",
        "formal gate status",
    )
    _require(freeze.get("formal_outcomes_used"), False, "formal outcome exclusion")
    code_revision = str(freeze.get("code_revision", ""))
    if _GIT_REVISION.fullmatch(code_revision) is None:
        raise ValueError("DocVQA policy freeze has an invalid code revision")
    commit_time = freeze.get("implementation_commit_time_unix")
    if not isinstance(commit_time, int) or commit_time <= 0:
        raise ValueError("DocVQA policy freeze has an invalid implementation time")
    _require(
        freeze.get("implementation_committed_before_calibration_output"),
        True,
        "implementation timing",
    )
    calibration = freeze.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("DocVQA policy freeze lacks calibration metadata")
    expected_calibration = {
        "selection_status": SUCCESS,
        "n_sources": CALIBRATION_SOURCES,
        "formal_outcomes_used": False,
    }
    for name, expected in expected_calibration.items():
        _require(calibration.get(name), expected, f"calibration {name}")
    n_decisions = calibration.get("n_decisions")
    if not isinstance(n_decisions, int) or n_decisions < CALIBRATION_SOURCES:
        raise ValueError("DocVQA policy freeze has invalid calibration decisions")
    threshold = calibration.get("selected_threshold")
    if not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)):
        raise ValueError("DocVQA policy freeze lacks a finite selected threshold")

    formal = freeze.get("formal_test")
    if not isinstance(formal, Mapping):
        raise ValueError("DocVQA policy freeze lacks formal metadata")
    expected_formal = {
        "allocated_sources": FORMAL_SOURCES,
        "manifest_materialized": False,
        "rollouts_collected": False,
        "outcomes_used": False,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_confidence": BOOTSTRAP_CONFIDENCE,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    for name, expected in expected_formal.items():
        _require(formal.get(name), expected, f"formal {name}")

    required_sections = {
        "artifacts": REQUIRED_ARTIFACTS,
        "implementation": REQUIRED_IMPLEMENTATION,
    }
    for section_name, required_names in required_sections.items():
        section = freeze.get(section_name)
        if not isinstance(section, Mapping) or not section:
            raise ValueError(f"DocVQA policy freeze lacks {section_name}")
        actual_names = set(section)
        missing = required_names.difference(actual_names)
        extra = actual_names.difference(required_names)
        if missing or extra:
            details = []
            if missing:
                details.append("missing " + ", ".join(sorted(missing)))
            if extra:
                details.append("unexpected " + ", ".join(sorted(extra)))
            raise ValueError(
                f"DocVQA policy freeze has an invalid {section_name} inventory: "
                + "; ".join(details)
            )
        for name, raw_component in section.items():
            if not isinstance(raw_component, Mapping):
                raise ValueError(f"DocVQA frozen component {section_name}.{name} invalid")
            path = Path(str(raw_component.get("path", ""))).resolve()
            digest = str(raw_component.get("sha256", ""))
            if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError(f"DocVQA frozen hash {section_name}.{name} invalid")
            if verify_components:
                check_hash(path, digest, f"{section_name}.{name}")


def validate_materialized_formal_gate(
    *,
    policy_freeze_path: Path,
    expected_policy_freeze_sha256: str,
    model_path: Path,
    expected_model_sha256: str,
    manifest_path: Path,
    expected_manifest_sha256: str,
    manifest_provenance_path: Path,
    expected_manifest_provenance_sha256: str,
    audit_path: Path,
    expected_audit_sha256: str,
) -> dict[str, Any]:
    """Bind one materialized formal manifest to the exact frozen policy."""

    import json

    check_hash(policy_freeze_path, expected_policy_freeze_sha256, "policy freeze")
    freeze = json.loads(policy_freeze_path.read_text(encoding="utf-8"))
    if not isinstance(freeze, dict):
        raise ValueError("DocVQA policy freeze must be a JSON object")
    validate_policy_freeze(freeze)
    check_hash(model_path, expected_model_sha256, "calibrated model")
    frozen_model = freeze["artifacts"].get("calibrated_model")
    if not isinstance(frozen_model, Mapping):
        raise ValueError("DocVQA policy freeze lacks its calibrated model")
    _require(
        str(Path(str(frozen_model.get("path", ""))).resolve()),
        str(model_path.resolve()),
        "formal model path",
    )
    _require(frozen_model.get("sha256"), expected_model_sha256, "formal model hash")
    check_hash(manifest_path, expected_manifest_sha256, "manifest")
    check_hash(
        manifest_provenance_path,
        expected_manifest_provenance_sha256,
        "manifest provenance",
    )
    check_hash(audit_path, expected_audit_sha256, "manifest audit")

    provenance = json.loads(manifest_provenance_path.read_text(encoding="utf-8"))
    if not isinstance(provenance, Mapping):
        raise ValueError("DocVQA formal manifest provenance must be a mapping")
    _require(
        provenance.get("code_revision"),
        freeze.get("code_revision"),
        "formal provenance code revision",
    )
    _require(
        provenance.get("formal_outcomes_collected"),
        False,
        "formal provenance outcome exclusion",
    )
    selection = provenance.get("selection_metadata")
    if not isinstance(selection, Mapping):
        raise ValueError("DocVQA formal manifest lacks selection metadata")
    allocation = freeze["artifacts"].get("allocation")
    if not isinstance(allocation, Mapping):
        raise ValueError("DocVQA policy freeze lacks allocation")
    expected_selection = {
        "policy_freeze_sha256": expected_policy_freeze_sha256,
        "allocation_sha256": allocation.get("sha256"),
        "protocol_sha256": PROTOCOL_SHA256,
        "role": "formal_test",
        "selected_source_group_count": FORMAL_SOURCES,
        "selection_uses_targets": False,
    }
    for name, expected in expected_selection.items():
        _require(selection.get(name), expected, f"formal selection {name}")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(audit, Mapping):
        raise ValueError("DocVQA formal manifest audit must be a mapping")
    formal_audit = audit.get("formal")
    if not isinstance(formal_audit, Mapping):
        raise ValueError("DocVQA formal audit lacks manifest details")
    expected_audit = {
        "passed": True,
        "policy_freeze_sha256": expected_policy_freeze_sha256,
        "allocation_sha256": allocation.get("sha256"),
        "formal_outcomes_collected": False,
    }
    for name, expected in expected_audit.items():
        _require(audit.get(name), expected, f"formal audit {name}")
    _require(
        formal_audit.get("manifest_sha256"),
        expected_manifest_sha256,
        "formal audit manifest hash",
    )
    _require(
        formal_audit.get("unique_sources"),
        FORMAL_SOURCES,
        "formal audit sources",
    )
    _require(
        formal_audit.get("unique_images"),
        FORMAL_SOURCES,
        "formal audit images",
    )
    count = formal_audit.get("count")
    if not isinstance(count, int) or count < FORMAL_SOURCES:
        raise ValueError("DocVQA formal audit has an invalid question count")
    _require(formal_audit.get("unique_states"), count, "formal audit states")
    overlap = audit.get("overlap")
    if not isinstance(overlap, Mapping) or any(int(value) != 0 for value in overlap.values()):
        raise ValueError("DocVQA formal audit reports identity overlap")
    return freeze
