from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from beyond_entropy.predictability_audit import AUDIT_BENCHMARKS
from beyond_entropy.predictability_matrix_artifacts import (
    TEST_TRANSACTION_PLAN_SCHEMA,
    atomic_json_write_exclusive,
    current_clean_revision,
    load_hashed_json,
    sha256_file,
    validate_protocol_artifact,
)


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build the hash-bound pre-access formal test transaction plan"
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--execution-config", required=True)
    parser.add_argument("--frozen-model", required=True)
    parser.add_argument("--frozen-report", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    root = Path(args.repo_root).resolve()
    revision = current_clean_revision(root)
    protocol_path, protocol_sha256, _ = validate_protocol_artifact(
        {
            "path": str(Path(args.protocol).resolve()),
            "sha256": args.expected_protocol_sha256,
        }
    )
    execution_path = Path(args.execution_config).resolve()
    execution_sha256 = sha256_file(execution_path)
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    if (
        not isinstance(execution, Mapping)
        or execution.get("schema") != "predictability_formal_test_execution_config_v1"
        or _mapping(execution.get("protocol"), name="execution protocol").get("sha256")
        != protocol_sha256
    ):
        raise ValueError("formal test execution config does not bind the protocol")
    model_path = Path(args.frozen_model).resolve()
    report_path = Path(args.frozen_report).resolve()
    model_sha256 = sha256_file(model_path)
    report_sha256 = sha256_file(report_path)
    _, freeze_report = load_hashed_json(
        report_path,
        expected_sha256=report_sha256,
        schema="predictability_matrix_freeze_report_v2",
    )
    if (
        freeze_report.get("model_sha256") != model_sha256
        or freeze_report.get("formal_claim_eligible") is not True
        or freeze_report.get("test_data_present") is not False
        or _mapping(freeze_report.get("provenance"), name="freeze provenance").get(
            "code_revision"
        )
        != revision
        or _mapping(freeze_report.get("provenance"), name="freeze provenance").get(
            "protocol_sha256"
        )
        != protocol_sha256
    ):
        raise ValueError("frozen artifacts are not eligible for the formal test")
    run_root = Path(args.run_root).resolve()
    roles = execution.get("roles_in_execution_order")
    if not isinstance(roles, list) or len(roles) != len(AUDIT_BENCHMARKS):
        raise ValueError("formal test execution roles are incomplete")
    by_benchmark = {
        str(_mapping(item, name="test role")["benchmark"]): _mapping(
            item, name="test role"
        )
        for item in roles
    }
    if set(by_benchmark) != set(AUDIT_BENCHMARKS):
        raise ValueError("formal test execution roles differ from benchmarks")
    configured_allocation = _mapping(
        execution.get("allocation_report"), name="allocation_report"
    )
    allocation_path = root / str(configured_allocation["path"])
    allocation_sha256 = str(configured_allocation["sha256"])
    if len(allocation_sha256) != 64:
        raise ValueError("configured allocation report hash is invalid")
    benchmarks = {
        benchmark: {
            # These paths and expected hashes are copied from the frozen config;
            # this builder intentionally does not stat, hash, or load them.
            "manifest_path": str(
                root
                / "data"
                / "predictability-audit-v1"
                / benchmark
                / "test"
                / "manifest.jsonl"
            ),
            "expected_manifest_sha256": str(
                by_benchmark[benchmark]["expected_manifest_sha256"]
            ),
            "expected_states": int(by_benchmark[benchmark]["states"]),
        }
        for benchmark in AUDIT_BENCHMARKS
    }
    plan = {
        "schema": TEST_TRANSACTION_PLAN_SCHEMA,
        "code_revision": revision,
        "protocol": {"path": str(protocol_path), "sha256": protocol_sha256},
        "execution_config": {
            "path": str(execution_path),
            "sha256": execution_sha256,
        },
        # This plan copies the preregistered digest but intentionally does not
        # stat, hash, or read the allocation report before the access ledger.
        "allocation_report": {
            "path": str(allocation_path),
            "sha256": allocation_sha256,
        },
        "frozen": {
            "model": {"path": str(model_path), "sha256": model_sha256},
            "report": {"path": str(report_path), "sha256": report_sha256},
        },
        "run_root": str(run_root),
        "access_ledger": str(run_root / "test-access.json"),
        "test_input_spec": str(run_root / "test-inputs.json"),
        "report_output": str(run_root / "predictability-matrix-test-report.json"),
        "final_audit_output": str(run_root / "PREDICTABILITY_AUDIT.md"),
        "benchmarks": benchmarks,
    }
    atomic_json_write_exclusive(args.output, plan)
    print(
        json.dumps(
            {
                "schema": TEST_TRANSACTION_PLAN_SCHEMA,
                "output": str(Path(args.output).resolve()),
                "output_sha256": sha256_file(args.output),
                "test_artifacts_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
