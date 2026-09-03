from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping, Sequence


AUDIT_SCHEMA = "n3_tool_checkpoint_novelty_audit_v1"
REGISTRY_SCHEMA = "n3_tool_checkpoint_novelty_registry_v1"
_FULL_REVISION = re.compile(r"[0-9a-f]{40}")
_PERMISSIVE_LICENSES = frozenset({"apache-2.0", "mit", "bsd-3-clause"})


def _required_text(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _required_bool(record: Mapping[str, Any], key: str) -> bool:
    value = record.get(key)
    if type(value) is not bool:
        raise ValueError(f"{key} must be a boolean")
    return value


def _required_nonnegative_int(record: Mapping[str, Any], key: str) -> int:
    value = record.get(key)
    if type(value) is not int or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class CheckpointCandidate:
    model_id: str
    revision: str
    weights_license: str
    private: bool
    gated: bool
    disabled: bool
    architecture: str
    model_type: str
    parameter_count: int
    used_storage_bytes: int
    model_card_bytes: int
    current_code_checkpoint_mapping_documented: bool
    exact_prompt_parser_contract_documented: bool
    exact_parser_valid_execution_trace_available: bool
    runtime_family_compatible: bool
    local_cache_present: bool

    @classmethod
    def from_record(
        cls, record: Mapping[str, Any], *, local_cache_present: bool
    ) -> "CheckpointCandidate":
        revision = _required_text(record, "revision")
        if _FULL_REVISION.fullmatch(revision) is None:
            raise ValueError("revision must be a full 40-character lowercase SHA")
        candidate = cls(
            model_id=_required_text(record, "model_id"),
            revision=revision,
            weights_license=_required_text(record, "weights_license").lower(),
            private=_required_bool(record, "private"),
            gated=_required_bool(record, "gated"),
            disabled=_required_bool(record, "disabled"),
            architecture=_required_text(record, "architecture"),
            model_type=_required_text(record, "model_type"),
            parameter_count=_required_nonnegative_int(record, "parameter_count"),
            used_storage_bytes=_required_nonnegative_int(
                record, "used_storage_bytes"
            ),
            model_card_bytes=_required_nonnegative_int(record, "model_card_bytes"),
            current_code_checkpoint_mapping_documented=_required_bool(
                record, "current_code_checkpoint_mapping_documented"
            ),
            exact_prompt_parser_contract_documented=_required_bool(
                record, "exact_prompt_parser_contract_documented"
            ),
            exact_parser_valid_execution_trace_available=_required_bool(
                record, "exact_parser_valid_execution_trace_available"
            ),
            runtime_family_compatible=_required_bool(
                record, "runtime_family_compatible"
            ),
            local_cache_present=local_cache_present,
        )
        if candidate.parameter_count == 0 or candidate.used_storage_bytes == 0:
            raise ValueError("checkpoint parameter count and storage must be positive")
        return candidate

    @property
    def is_public_and_usable(self) -> bool:
        return not self.private and not self.gated and not self.disabled

    @property
    def has_permissive_weights_license(self) -> bool:
        return self.weights_license in _PERMISSIVE_LICENSES

    @property
    def has_exact_artifact_support_evidence(self) -> bool:
        return (
            self.current_code_checkpoint_mapping_documented
            and self.exact_prompt_parser_contract_documented
            and self.exact_parser_valid_execution_trace_available
        )


def _parse_collision_registry(
    literature: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    parsed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in literature:
        work = _required_text(entry, "work")
        if work in seen:
            raise ValueError(f"duplicate literature work: {work}")
        seen.add(work)
        source = _required_text(entry, "source")
        overlaps_raw = entry.get("overlaps")
        if not isinstance(overlaps_raw, list) or not overlaps_raw:
            raise ValueError(f"literature work {work} must list overlaps")
        overlaps = tuple(str(item).strip() for item in overlaps_raw)
        if any(not item for item in overlaps):
            raise ValueError(f"literature work {work} has an empty overlap")
        parsed.append({"work": work, "source": source, "overlaps": overlaps})
    if not parsed:
        raise ValueError("literature collision registry must be non-empty")
    return tuple(parsed)


def audit_checkpoint_and_novelty(
    registry: Mapping[str, Any], *, local_cache_model_ids: Sequence[str] = ()
) -> dict[str, Any]:
    if registry.get("schema") != REGISTRY_SCHEMA:
        raise ValueError("unexpected N3 registry schema")
    raw_candidates = registry.get("checkpoint_candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("checkpoint_candidates must be a non-empty list")
    cache_ids = set(local_cache_model_ids)
    candidates = tuple(
        CheckpointCandidate.from_record(
            record,
            local_cache_present=_required_text(record, "model_id") in cache_ids,
        )
        for record in raw_candidates
        if isinstance(record, Mapping)
    )
    if len(candidates) != len(raw_candidates):
        raise ValueError("every checkpoint candidate must be an object")
    if len({candidate.model_id for candidate in candidates}) != len(candidates):
        raise ValueError("checkpoint model IDs must be unique")

    public_candidates = tuple(
        candidate
        for candidate in candidates
        if candidate.is_public_and_usable
        and candidate.has_permissive_weights_license
        and candidate.runtime_family_compatible
    )
    selected = (
        min(public_candidates, key=lambda item: item.used_storage_bytes)
        if public_candidates
        else None
    )
    literature_raw = registry.get("literature_collision")
    if not isinstance(literature_raw, list):
        raise ValueError("literature_collision must be a list")
    literature = _parse_collision_registry(literature_raw)
    covered_claims = sorted(
        {overlap for entry in literature for overlap in entry["overlaps"]}
    )
    candidate_claims_raw = registry.get("candidate_method_claims")
    if not isinstance(candidate_claims_raw, list) or not candidate_claims_raw:
        raise ValueError("candidate_method_claims must be a non-empty list")
    candidate_claims = tuple(str(item).strip() for item in candidate_claims_raw)
    if any(not claim for claim in candidate_claims):
        raise ValueError("candidate method claims must be non-empty")
    uncovered_claims = sorted(set(candidate_claims) - set(covered_claims))

    baseline_checks = {
        "public_ungated_checkpoint_exists": selected is not None,
        "full_immutable_revision_exists": bool(
            selected and _FULL_REVISION.fullmatch(selected.revision)
        ),
        "weights_license_is_permissive": bool(
            selected and selected.has_permissive_weights_license
        ),
        "runtime_model_family_is_compatible": bool(
            selected and selected.runtime_family_compatible
        ),
        "current_code_maps_exact_checkpoint": bool(
            selected and selected.current_code_checkpoint_mapping_documented
        ),
        "exact_prompt_parser_contract_is_documented": bool(
            selected and selected.exact_prompt_parser_contract_documented
        ),
        "exact_checkpoint_has_parser_valid_execution_trace": bool(
            selected and selected.exact_parser_valid_execution_trace_available
        ),
    }
    baseline_gate_passed = all(baseline_checks.values())
    novelty_checks = {
        "signed_tool_value_is_not_covered": "signed_tool_value" in uncovered_claims,
        "tool_responsibility_routing_is_not_covered": (
            "tool_responsibility_routing" in uncovered_claims
        ),
        "fixed_prefix_observation_contrast_is_not_covered": (
            "fixed_prefix_observation_contrast" in uncovered_claims
        ),
        "action_level_counterfactual_credit_is_not_covered": (
            "action_level_counterfactual_credit" in uncovered_claims
        ),
        "with_without_tool_benefit_is_not_covered": (
            "with_without_tool_benefit_supervision" in uncovered_claims
        ),
        "candidate_has_uncovered_core_claim": bool(uncovered_claims),
    }
    novelty_gate_passed = all(novelty_checks.values())
    joint_gate_passed = baseline_gate_passed and novelty_gate_passed
    decision = (
        "n3_checkpoint_and_novelty_joint_gate_passed"
        if joint_gate_passed
        else "n3_public_initializer_exists_but_joint_gate_failed_before_download"
    )
    return {
        "schema": AUDIT_SCHEMA,
        "audited_at": _required_text(registry, "audited_at"),
        "checkpoint_candidates": [asdict(candidate) for candidate in candidates],
        "selected_candidate_if_scientifically_authorized": (
            None if selected is None else selected.model_id
        ),
        "selected_candidate_revision": None if selected is None else selected.revision,
        "selected_candidate_download_bytes": (
            0 if selected is None else selected.used_storage_bytes
        ),
        "literature_collision": list(literature),
        "candidate_method_claims": list(candidate_claims),
        "covered_core_claims": covered_claims,
        "uncovered_core_claims": uncovered_claims,
        "baseline_checks": baseline_checks,
        "baseline_gate_passed": baseline_gate_passed,
        "novelty_checks": novelty_checks,
        "novelty_gate_passed": novelty_gate_passed,
        "joint_gate_passed": joint_gate_passed,
        "decision": decision,
        "downloaded_checkpoint_bytes": 0,
        "authorized_new_gpu_jobs": int(joint_gate_passed),
        "authorized_new_checkpoints": 0,
    }
