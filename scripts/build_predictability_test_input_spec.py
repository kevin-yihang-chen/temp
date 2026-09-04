from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from beyond_entropy.predictability_audit import AUDIT_BENCHMARKS
from beyond_entropy.predictability_matrix_artifacts import (
    TEST_INPUT_SCHEMA,
    atomic_json_write_exclusive,
    sha256_file,
)
from beyond_entropy.predictability_test_transaction import (
    load_test_transaction_plan,
    validate_existing_test_access_ledger,
)


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build the post-generation hash-bound one-shot test input spec"
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)

    _, plan = load_test_transaction_plan(
        args.plan,
        expected_sha256=args.expected_plan_sha256,
        repo_root=args.repo_root,
        require_unstarted=False,
    )
    frozen = _mapping(plan["frozen"], name="frozen")
    model = _mapping(frozen["model"], name="frozen.model")
    report = _mapping(frozen["report"], name="frozen.report")
    protocol = _mapping(plan["protocol"], name="protocol")
    allocation = _mapping(plan["allocation_report"], name="allocation_report")
    ledger_path = Path(str(plan["access_ledger"])).resolve()
    ledger_sha256 = sha256_file(ledger_path)
    validate_existing_test_access_ledger(
        {"path": str(ledger_path), "sha256": ledger_sha256},
        plan_sha256=args.expected_plan_sha256,
        model_sha256=str(model["sha256"]),
        report_sha256=str(report["sha256"]),
        protocol_sha256=str(protocol["sha256"]),
        allocation_report_sha256=str(allocation["sha256"]),
        code_revision=str(plan["code_revision"]),
    )
    run_root = Path(str(plan["run_root"])).resolve()
    benchmarks: dict[str, Any] = {}
    for benchmark in AUDIT_BENCHMARKS:
        completion = run_root / benchmark / "test" / "complete.json"
        completion_sha256 = sha256_file(completion)
        value = json.loads(completion.read_text(encoding="utf-8"))
        if (
            not isinstance(value, Mapping)
            or value.get("schema") != "predictability_formal_test_role_v1"
            or value.get("passed") is not True
            or value.get("benchmark") != benchmark
            or value.get("role") != "test"
            or value.get("code_revision") != plan["code_revision"]
            or value.get("protocol_sha256") != protocol["sha256"]
            or value.get("test_transaction_plan_sha256") != args.expected_plan_sha256
            or value.get("access_ledger_sha256") != ledger_sha256
        ):
            raise ValueError(f"invalid sealed test role: {benchmark}")
        artifacts = _mapping(value.get("artifacts"), name=f"{benchmark}.artifacts")
        role: dict[str, Any] = {}
        for name in ("manifest", "rollouts", "rollout_provenance", "features"):
            path = Path(str(artifacts[f"{name}_path"])).resolve()
            expected = str(artifacts[name])
            if sha256_file(path) != expected:
                raise ValueError(f"sealed test artifact changed: {benchmark}.{name}")
            role[name] = {"path": str(path), "sha256": expected}
        benchmarks[benchmark] = {
            "test": role,
            "completion_sha256": completion_sha256,
        }
    # Completion hashes are checked above but excluded from the loader's exact
    # benchmark schema, so retain them in the CLI output instead of the spec.
    completion_hashes = {
        benchmark: benchmarks[benchmark].pop("completion_sha256")
        for benchmark in AUDIT_BENCHMARKS
    }
    spec = {
        "schema": TEST_INPUT_SCHEMA,
        "code_revision": plan["code_revision"],
        "protocol": dict(protocol),
        "allocation_report": dict(allocation),
        "frozen": {"model": dict(model), "report": dict(report)},
        "benchmarks": benchmarks,
        "access_ledger": {"path": str(ledger_path), "sha256": ledger_sha256},
        "test_transaction_plan_sha256": args.expected_plan_sha256,
        "output": plan["report_output"],
    }
    output = Path(str(plan["test_input_spec"])).resolve()
    atomic_json_write_exclusive(output, spec)
    print(
        json.dumps(
            {
                "schema": TEST_INPUT_SCHEMA,
                "output": str(output),
                "output_sha256": sha256_file(output),
                "sealed_test_role_completion_sha256": completion_hashes,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
