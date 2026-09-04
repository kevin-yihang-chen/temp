from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_formal_development_budget_is_bound_and_self_consistent() -> None:
    config = json.loads(
        (ROOT / "configs/predictability_formal_execution_v1.json").read_text()
    )
    protocol = ROOT / config["protocol"]["path"]
    assert (
        hashlib.sha256(protocol.read_bytes()).hexdigest()
        == config["protocol"]["sha256"]
    )
    assert config["scope"] == "development_train_and_validation_only"
    assert config["test_artifacts_authorized"] is False
    assert config["learned_backbone_checkpoints"] == 0
    assert config["final_frozen_matrix_bundles_after_development"] == 1

    roles = config["roles_in_submission_order"]
    assert [(item["benchmark"], item["role"]) for item in roles] == [
        ("chartqa", "train"),
        ("chartqa", "validation"),
        ("docvqa", "train"),
        ("docvqa", "validation"),
        ("hrbench", "train"),
        ("hrbench", "validation"),
    ]
    assert sum(item["states"] for item in roles) == config["aggregate"]["states"]
    interval = config["checkpoint_interval_states"]
    assert all(
        item["planned_checkpoint_save_events"]
        == 2 * math.ceil(item["states"] / interval)
        for item in roles
    )
    assert (
        sum(item["planned_checkpoint_save_events"] for item in roles)
        == config["planned_checkpoint_save_events"]
    )
    assert config["persistent_resume_files"] == 2 * len(roles)

    observed = config["observed_64_state_throughput"]
    assert all(
        math.isclose(
            item["states_per_h800_hour"],
            64 / item["elapsed_seconds"] * 3600,
        )
        for item in observed.values()
    )
    expected_raw = sum(item["raw_estimated_h800_hours"] for item in roles)
    assert math.isclose(expected_raw, config["aggregate"]["raw_estimated_h800_hours"])
    assert math.isclose(
        expected_raw * config["conservative_runtime_multiplier"],
        config["aggregate"]["conservative_reserved_h800_hours"],
    )


def test_formal_development_worker_is_train_validation_only_and_notifies() -> None:
    worker = (ROOT / "scripts/slurm_predictability_formal_development.sh").read_text()
    submitter = (
        ROOT / "scripts/submit_predictability_formal_development.sh"
    ).read_text()
    assert "train|validation" in worker
    assert "train|validation" in submitter
    assert "--mail-type=ALL" in worker
    assert "--mail-type=ALL" in submitter
    assert "yihangc@connect.hku.hk" in worker
    assert "yihangc@connect.hku.hk" in submitter
    assert "data/predictability-audit-v1/${BE_PRED_BENCHMARK}/${BE_PRED_ROLE}" in worker
    assert "test_artifacts_authorized" in submitter
    assert '--dataset-role "${BE_PRED_ROLE}"' in worker
