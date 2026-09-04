from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from beyond_entropy.predictability_matrix_artifacts import (
    TEST_ACCESS_SCHEMA,
    sha256_file,
)
from beyond_entropy.predictability_test_transaction import (
    start_test_transaction,
    validate_existing_test_access_ledger,
)


ROOT = Path(__file__).resolve().parents[1]


def _fake_plan(tmp_path: Path) -> dict:
    return {
        "frozen": {
            "model": {"path": "/frozen.pkl", "sha256": "1" * 64},
            "report": {"path": "/freeze.json", "sha256": "2" * 64},
        },
        "protocol": {"path": "/protocol.json", "sha256": "3" * 64},
        "execution_config": {"path": "/execution.json", "sha256": "4" * 64},
        "allocation_report": {"path": "/allocation.json", "sha256": "6" * 64},
        "code_revision": "5" * 40,
        "run_root": str(tmp_path),
        "access_ledger": str(tmp_path / "test-access.json"),
    }


def test_test_access_record_is_exclusive_and_hash_bound(tmp_path, monkeypatch) -> None:
    plan = _fake_plan(tmp_path)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        "beyond_entropy.predictability_test_transaction.load_test_transaction_plan",
        lambda *args, **kwargs: (plan_path, plan),
    )
    started = start_test_transaction(
        plan_path, expected_sha256="a" * 64, repo_root=tmp_path
    )
    ledger = Path(started["ledger"])
    record = json.loads(ledger.read_text(encoding="utf-8"))
    assert record["schema"] == TEST_ACCESS_SCHEMA
    assert record["test_artifacts_accessed_before_this_record"] is False
    assert record["automatic_retry_allowed"] is False
    assert started["ledger_sha256"] == sha256_file(ledger)
    validate_existing_test_access_ledger(
        {"path": str(ledger), "sha256": started["ledger_sha256"]},
        plan_sha256="a" * 64,
        model_sha256="1" * 64,
        report_sha256="2" * 64,
        protocol_sha256="3" * 64,
        allocation_report_sha256="6" * 64,
        code_revision="5" * 40,
    )
    with pytest.raises(FileExistsError):
        start_test_transaction(plan_path, expected_sha256="a" * 64, repo_root=tmp_path)
    with pytest.raises(ValueError, match="plan_sha256"):
        validate_existing_test_access_ledger(
            {"path": str(ledger), "sha256": started["ledger_sha256"]},
            plan_sha256="b" * 64,
            model_sha256="1" * 64,
            report_sha256="2" * 64,
            protocol_sha256="3" * 64,
            allocation_report_sha256="6" * 64,
            code_revision="5" * 40,
        )


def test_formal_test_budget_and_irreversible_worker_order_are_frozen() -> None:
    config_path = ROOT / "configs/predictability_formal_test_execution_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    protocol = ROOT / config["protocol"]["path"]
    assert (
        hashlib.sha256(protocol.read_bytes()).hexdigest()
        == config["protocol"]["sha256"]
    )
    assert config["automatic_retry_after_access_ledger"] is False
    assert config["allocation_report"]["required_test_historically_opened"] is False
    assert config["persistent_resume_files"] == 6
    roles = config["roles_in_execution_order"]
    interval = config["checkpoint_interval_states"]
    assert sum(item["states"] for item in roles) == config["aggregate"]["states"]
    assert all(
        item["planned_checkpoint_save_events"]
        == 2 * math.ceil(item["states"] / interval)
        for item in roles
    )
    assert sum(item["planned_checkpoint_save_events"] for item in roles) == 28
    assert math.isclose(
        sum(item["raw_estimated_h800_hours"] for item in roles),
        config["aggregate"]["raw_estimated_h800_hours"],
    )
    assert math.isclose(
        config["aggregate"]["raw_estimated_h800_hours"]
        * config["aggregate"]["conservative_runtime_multiplier"],
        config["aggregate"]["conservative_reserved_h800_hours"],
    )

    worker = (ROOT / "scripts/slurm_predictability_formal_test_once.sh").read_text()
    submitter = (ROOT / "scripts/submit_predictability_formal_test_once.sh").read_text()
    boundary = worker.index('"${python_bin}" "${starter}"')
    first_manifest_hash = worker.index(
        'check_hash "${manifest}" "${expected_manifest_sha256}"'
    )
    first_manifest_count = worker.index("actual_states=$(awk")
    assert boundary < first_manifest_hash
    assert boundary < first_manifest_count
    assert "--resume" not in worker
    assert "--mail-type=ALL" in worker
    assert "--mail-type=ALL" in submitter
    assert "yihangc@connect.hku.hk" in worker
    assert "yihangc@connect.hku.hk" in submitter
    assert ".manifest_path" not in submitter
