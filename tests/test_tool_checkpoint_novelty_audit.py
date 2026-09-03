from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from beyond_entropy.tool_checkpoint_novelty_audit import (
    AUDIT_SCHEMA,
    REGISTRY_SCHEMA,
    CheckpointCandidate,
    audit_checkpoint_and_novelty,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs" / "n3_tool_checkpoint_novelty_audit_v1.json"


def _registry() -> dict[str, Any]:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def test_checkpoint_candidate_requires_full_revision() -> None:
    record = _registry()["checkpoint_candidates"][0]
    record["revision"] = "0ca11e8"
    with pytest.raises(ValueError, match="full 40-character"):
        CheckpointCandidate.from_record(record, local_cache_present=False)


def test_public_candidate_selection_prefers_smallest_compatible_artifact() -> None:
    report = audit_checkpoint_and_novelty(_registry())
    assert report["schema"] == AUDIT_SCHEMA
    assert report["selected_candidate_if_scientifically_authorized"] == (
        "VTOOL/VTool-Qwen2.5-3B"
    )
    assert report["selected_candidate_download_bytes"] == 8_143_089_840
    assert report["baseline_checks"]["public_ungated_checkpoint_exists"]
    assert report["baseline_checks"]["full_immutable_revision_exists"]
    assert report["baseline_checks"]["weights_license_is_permissive"]
    assert report["baseline_checks"]["runtime_model_family_is_compatible"]


def test_joint_gate_rejects_undocumented_support_and_covered_method_claims() -> None:
    report = audit_checkpoint_and_novelty(_registry())
    assert not report["baseline_gate_passed"]
    assert not report["novelty_gate_passed"]
    assert not report["joint_gate_passed"]
    assert report["uncovered_core_claims"] == []
    assert report["decision"] == (
        "n3_public_initializer_exists_but_joint_gate_failed_before_download"
    )
    assert report["downloaded_checkpoint_bytes"] == 0
    assert report["authorized_new_gpu_jobs"] == 0
    assert report["authorized_new_checkpoints"] == 0


def test_local_cache_presence_is_reported_but_does_not_repair_scientific_gate() -> None:
    report = audit_checkpoint_and_novelty(
        _registry(), local_cache_model_ids=["VTOOL/VTool-Qwen2.5-3B"]
    )
    selected = next(
        item
        for item in report["checkpoint_candidates"]
        if item["model_id"] == "VTOOL/VTool-Qwen2.5-3B"
    )
    assert selected["local_cache_present"]
    assert not report["joint_gate_passed"]


def test_registry_rejects_duplicate_literature_work() -> None:
    registry = _registry()
    registry["literature_collision"].append(
        dict(registry["literature_collision"][0])
    )
    with pytest.raises(ValueError, match="duplicate literature work"):
        audit_checkpoint_and_novelty(registry)


def test_registry_schema_is_explicit() -> None:
    registry = _registry()
    assert registry["schema"] == REGISTRY_SCHEMA
    registry["schema"] = "wrong"
    with pytest.raises(ValueError, match="unexpected N3 registry schema"):
        audit_checkpoint_and_novelty(registry)
