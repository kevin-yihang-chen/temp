from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .predictability_audit import AUDIT_BENCHMARKS
from .predictability_matrix import (
    frozen_predictability_matrix_report,
    load_frozen_predictability_matrix,
)
from .predictability_matrix_artifacts import (
    TEST_ACCESS_SCHEMA,
    TEST_TRANSACTION_PLAN_SCHEMA,
    atomic_json_write_exclusive,
    current_clean_revision,
    load_hashed_json,
    sha256_file,
    validate_protocol_artifact,
)
from .predictability_verdict import FINAL_AUDIT_FILENAME


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} keys differ from the frozen schema")


def _hashed_artifact(value: Any, *, name: str) -> tuple[Path, str]:
    spec = _mapping(value, name=name)
    _exact_keys(spec, {"path", "sha256"}, name=name)
    path = Path(str(spec["path"])).resolve()
    expected = str(spec["sha256"])
    if len(expected) != 64 or sha256_file(path) != expected:
        raise ValueError(f"{name} SHA-256 mismatch")
    return path, expected


def load_test_transaction_plan(
    path: str | Path,
    *,
    expected_sha256: str,
    repo_root: str | Path,
    require_unstarted: bool = True,
) -> tuple[Path, dict[str, Any]]:
    """Validate pre-test state without opening any held-out test artifact."""

    source, plan = load_hashed_json(
        path,
        expected_sha256=expected_sha256,
        schema=TEST_TRANSACTION_PLAN_SCHEMA,
    )
    _exact_keys(
        plan,
        {
            "schema",
            "code_revision",
            "protocol",
            "execution_config",
            "allocation_report",
            "frozen",
            "run_root",
            "access_ledger",
            "test_input_spec",
            "report_output",
            "final_audit_output",
            "benchmarks",
        },
        name="test transaction plan",
    )
    revision = current_clean_revision(repo_root)
    if plan.get("code_revision") != revision:
        raise ValueError("test transaction plan differs from clean HEAD")
    protocol_path, protocol_sha256, _ = validate_protocol_artifact(plan["protocol"])
    execution_path, execution_sha256 = _hashed_artifact(
        plan["execution_config"], name="test execution config"
    )
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    if (
        not isinstance(execution, Mapping)
        or execution.get("schema") != "predictability_formal_test_execution_config_v1"
        or _mapping(execution.get("protocol"), name="execution protocol").get("sha256")
        != protocol_sha256
    ):
        raise ValueError("test execution config does not bind the protocol")
    allocation = _mapping(plan["allocation_report"], name="allocation_report")
    _exact_keys(allocation, {"path", "sha256"}, name="allocation_report")
    configured_allocation = _mapping(
        execution.get("allocation_report"), name="configured allocation_report"
    )
    if (
        allocation.get("sha256") != configured_allocation.get("sha256")
        or not isinstance(allocation.get("path"), str)
        or not Path(str(allocation.get("path"))).is_absolute()
        or len(str(allocation.get("sha256"))) != 64
    ):
        raise ValueError("test transaction allocation report binding differs")
    frozen = _mapping(plan["frozen"], name="frozen")
    _exact_keys(frozen, {"model", "report"}, name="frozen")
    model_path, model_sha256 = _hashed_artifact(frozen["model"], name="frozen.model")
    report_path, report_sha256 = _hashed_artifact(
        frozen["report"], name="frozen.report"
    )
    _, freeze_report = load_hashed_json(
        report_path,
        expected_sha256=report_sha256,
        schema="predictability_matrix_freeze_report_v2",
    )
    model = load_frozen_predictability_matrix(model_path, expected_sha256=model_sha256)
    inventory = frozen_predictability_matrix_report(model)
    if (
        freeze_report.get("model_sha256") != model_sha256
        or freeze_report.get("test_data_present") is not False
        or freeze_report.get("formal_claim_eligible") is not True
        or model.provenance.get("protocol_sha256") != protocol_sha256
        or model.provenance.get("code_revision") != revision
    ):
        raise ValueError("frozen model is not eligible for the formal test")
    for field, value in inventory.items():
        if freeze_report.get(field) != value:
            raise ValueError(f"frozen report/model mismatch for {field}")
    paths = {
        name: Path(str(plan[name])).resolve()
        for name in (
            "run_root",
            "access_ledger",
            "test_input_spec",
            "report_output",
            "final_audit_output",
        )
    }
    if len(set(paths.values())) != len(paths):
        raise ValueError("test transaction output paths must be distinct")
    if paths["final_audit_output"].name != FINAL_AUDIT_FILENAME:
        raise ValueError("test transaction final audit filename is invalid")
    if require_unstarted and any(
        paths[name].exists() for name in paths if name != "run_root"
    ):
        raise FileExistsError("test transaction output or access ledger already exists")
    raw_benchmarks = _mapping(plan["benchmarks"], name="benchmarks")
    if set(raw_benchmarks) != set(AUDIT_BENCHMARKS):
        raise ValueError("test transaction requires exactly three benchmarks")
    configured_roles = {
        str(item["benchmark"]): item
        for item in execution.get("roles_in_execution_order", ())
        if isinstance(item, Mapping)
    }
    if set(configured_roles) != set(AUDIT_BENCHMARKS):
        raise ValueError("test execution config benchmark roles are incomplete")
    for benchmark in AUDIT_BENCHMARKS:
        role = _mapping(raw_benchmarks[benchmark], name=benchmark)
        _exact_keys(
            role,
            {"manifest_path", "expected_manifest_sha256", "expected_states"},
            name=benchmark,
        )
        manifest = role.get("manifest_path")
        digest = role.get("expected_manifest_sha256")
        states = role.get("expected_states")
        if not isinstance(manifest, str) or not manifest:
            raise ValueError(f"{benchmark} test manifest path is invalid")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"{benchmark} expected test manifest hash is invalid")
        if not isinstance(states, int) or states <= 0:
            raise ValueError(f"{benchmark} expected test state count is invalid")
        configured = configured_roles[benchmark]
        if (
            configured.get("expected_manifest_sha256") != digest
            or configured.get("states") != states
        ):
            raise ValueError(f"{benchmark} test plan differs from execution config")
    # Keep these names live so static review can see every verified binding.
    if not protocol_path.is_file() or not execution_sha256:
        raise AssertionError("verified test transaction binding disappeared")
    return source, plan


def start_test_transaction(
    path: str | Path,
    *,
    expected_sha256: str,
    repo_root: str | Path,
) -> dict[str, Any]:
    plan_path, plan = load_test_transaction_plan(
        path, expected_sha256=expected_sha256, repo_root=repo_root
    )
    frozen = _mapping(plan["frozen"], name="frozen")
    protocol = _mapping(plan["protocol"], name="protocol")
    execution = _mapping(plan["execution_config"], name="execution_config")
    allocation = _mapping(plan["allocation_report"], name="allocation_report")
    model = _mapping(frozen["model"], name="frozen.model")
    report = _mapping(frozen["report"], name="frozen.report")
    record = {
        "schema": TEST_ACCESS_SCHEMA,
        "status": "started",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_transaction_plan": str(plan_path),
        "test_transaction_plan_sha256": expected_sha256,
        "frozen_model": str(Path(str(model["path"])).resolve()),
        "frozen_model_sha256": str(model["sha256"]),
        "frozen_report": str(Path(str(report["path"])).resolve()),
        "frozen_report_sha256": str(report["sha256"]),
        "protocol": str(Path(str(protocol["path"])).resolve()),
        "protocol_sha256": str(protocol["sha256"]),
        "execution_config": str(Path(str(execution["path"])).resolve()),
        "execution_config_sha256": str(execution["sha256"]),
        "allocation_report": str(allocation["path"]),
        "allocation_report_sha256": str(allocation["sha256"]),
        "code_revision": str(plan["code_revision"]),
        "run_root": str(Path(str(plan["run_root"])).resolve()),
        "test_artifacts_accessed_before_this_record": False,
        "automatic_retry_allowed": False,
    }
    ledger = Path(str(plan["access_ledger"])).resolve()
    atomic_json_write_exclusive(ledger, record)
    return {**record, "ledger": str(ledger), "ledger_sha256": sha256_file(ledger)}


def validate_existing_test_access_ledger(
    value: Any,
    *,
    plan_sha256: str,
    model_sha256: str,
    report_sha256: str,
    protocol_sha256: str,
    allocation_report_sha256: str,
    code_revision: str,
) -> tuple[Path, str, dict[str, Any]]:
    path, expected = _hashed_artifact(value, name="access_ledger")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("test access ledger is not valid JSON") from exc
    if not isinstance(record, dict) or record.get("schema") != TEST_ACCESS_SCHEMA:
        raise ValueError("unexpected test access ledger")
    required = {
        "status": "started",
        "test_transaction_plan_sha256": plan_sha256,
        "frozen_model_sha256": model_sha256,
        "frozen_report_sha256": report_sha256,
        "protocol_sha256": protocol_sha256,
        "allocation_report_sha256": allocation_report_sha256,
        "code_revision": code_revision,
        "test_artifacts_accessed_before_this_record": False,
        "automatic_retry_allowed": False,
    }
    for name, required_value in required.items():
        if record.get(name) != required_value:
            raise ValueError(f"test access ledger mismatch for {name}")
    return path, expected, record
