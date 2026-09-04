from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from beyond_entropy.predictability_matrix import (
    evaluate_frozen_predictability_matrix,
    frozen_predictability_matrix_report,
    load_frozen_predictability_matrix,
)
from beyond_entropy.predictability_matrix_artifacts import (
    atomic_json_write_exclusive,
    load_hashed_json,
    load_test_datasets_after_access_ledger,
    load_test_input_spec_header,
    sha256_file,
)
from beyond_entropy.predictability_test_transaction import (
    validate_existing_test_access_ledger,
)


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _frozen_artifact(value: Any, *, name: str) -> tuple[Path, str]:
    spec = _mapping(value, name=name)
    if set(spec) != {"path", "sha256"}:
        raise ValueError(f"{name} must contain exactly path and sha256")
    path = Path(str(spec["path"])).resolve()
    expected = str(spec["sha256"])
    if len(expected) != 64 or sha256_file(path) != expected:
        raise ValueError(f"{name} SHA-256 mismatch")
    return path, expected


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Apply one persisted predictability freeze to held-out test once"
    )
    parser.add_argument("--input-spec", required=True)
    parser.add_argument("--input-spec-sha256", required=True)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)

    spec_path, spec, protocol, revision = load_test_input_spec_header(
        args.input_spec,
        expected_sha256=args.input_spec_sha256,
        repo_root=args.repo_root,
    )
    output = Path(spec["output"]).resolve()
    if output.exists():
        raise FileExistsError("one-shot formal test output already exists")
    frozen_spec = _mapping(spec["frozen"], name="frozen")
    raw_model = _mapping(frozen_spec["model"], name="frozen.model")
    raw_report = _mapping(frozen_spec["report"], name="frozen.report")
    raw_allocation = _mapping(spec["allocation_report"], name="allocation_report")
    protocol_sha256 = str(_mapping(spec["protocol"], name="protocol")["sha256"])
    access_ledger, access_sha256, _ = validate_existing_test_access_ledger(
        spec["access_ledger"],
        plan_sha256=str(spec["test_transaction_plan_sha256"]),
        model_sha256=str(raw_model["sha256"]),
        report_sha256=str(raw_report["sha256"]),
        protocol_sha256=protocol_sha256,
        allocation_report_sha256=str(raw_allocation["sha256"]),
        code_revision=revision,
    )
    allocation_path, allocation_sha256 = _frozen_artifact(
        spec["allocation_report"], name="allocation_report"
    )
    allocation = json.loads(allocation_path.read_text(encoding="utf-8"))
    allocation_benchmarks = _mapping(
        allocation.get("benchmarks"), name="allocation benchmarks"
    )
    raw_benchmarks = _mapping(spec["benchmarks"], name="benchmarks")
    for benchmark in ("chartqa", "docvqa", "hrbench"):
        allocated = _mapping(
            _mapping(allocation_benchmarks[benchmark], name=benchmark)["test"],
            name=f"{benchmark}.allocated test",
        )
        test_spec = _mapping(
            _mapping(raw_benchmarks[benchmark], name=benchmark)["test"],
            name=f"{benchmark}.test",
        )
        manifest_spec = _mapping(
            test_spec["manifest"], name=f"{benchmark}.test.manifest"
        )
        if allocated.get("historically_opened") is not False or allocated.get(
            "manifest_sha256"
        ) != manifest_spec.get("sha256"):
            raise ValueError(f"{benchmark} test allocation is not untouched and bound")
    model_path, model_sha256 = _frozen_artifact(
        frozen_spec["model"], name="frozen.model"
    )
    report_path, report_sha256 = _frozen_artifact(
        frozen_spec["report"], name="frozen.report"
    )
    _, freeze_report = load_hashed_json(
        report_path,
        expected_sha256=report_sha256,
        schema="predictability_matrix_freeze_report_v2",
    )
    if (
        freeze_report.get("model_sha256") != model_sha256
        or freeze_report.get("test_data_present") is not False
        or freeze_report.get("formal_claim_eligible") is not True
    ):
        raise ValueError("frozen report is not a formal test-free model inventory")
    frozen = load_frozen_predictability_matrix(model_path, expected_sha256=model_sha256)
    inventory = frozen_predictability_matrix_report(frozen)
    for field in inventory:
        if freeze_report.get(field) != inventory[field]:
            raise ValueError(f"frozen report/model mismatch for {field}")
    if (
        frozen.provenance.get("protocol_sha256") != protocol_sha256
        or frozen.provenance.get("code_revision") != revision
    ):
        raise ValueError("frozen model provenance differs from test protocol or code")

    datasets, test_hashes = load_test_datasets_after_access_ledger(
        spec, code_revision=revision, protocol=protocol
    )
    metrics = protocol["metrics"]
    uncertainty = metrics["uncertainty"]
    report = evaluate_frozen_predictability_matrix(
        frozen,
        datasets,
        bootstrap_resamples=int(uncertainty["resamples"]),
        bootstrap_confidence=float(uncertainty["confidence_level"]),
        bootstrap_seed=int(uncertainty["seed"]),
        call_rates=tuple(float(item) for item in metrics["curve_call_rates"]),
    )
    report["one_shot_test_access"] = {
        "ledger": str(access_ledger),
        "ledger_sha256": access_sha256,
        "input_spec": str(spec_path),
        "input_spec_sha256": args.input_spec_sha256,
        "frozen_model_sha256": model_sha256,
        "frozen_report_sha256": report_sha256,
        "protocol_sha256": protocol_sha256,
        "allocation_report": str(allocation_path),
        "allocation_report_sha256": allocation_sha256,
        "code_revision": revision,
        "test_artifacts": test_hashes,
    }
    atomic_json_write_exclusive(output, report)
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "formal_claim_eligible": report["formal_claim_eligible"],
                "frozen_before_test": report["frozen_before_test"],
                "matrix_complete": report["matrix"]["complete"],
                "output": str(output),
                "output_sha256": sha256_file(output),
                "test_access_ledger_sha256": access_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
