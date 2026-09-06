"""Freeze the exact development-pilot validation subsets and frozen-VOI calls.

This step is development-only.  It never opens a test manifest and does not
fit, recalibrate, or otherwise change the previously frozen VOI policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from beyond_entropy.predictability_matrix_artifacts import (
    atomic_json_write_exclusive,
    sha256_file,
)
from beyond_entropy.utility_dataset import audit_utility_splits, load_utility_development
from beyond_entropy.utility_training import source_hash_subset


BENCHMARKS = ("chartqa", "docvqa", "hrbench")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def freeze_validation(
    *,
    development_bundle: str | Path,
    frozen_voi_report: str | Path,
    output_root: str | Path,
    maximum_sources: int,
    seed: int,
) -> dict[str, Any]:
    if maximum_sources <= 0:
        raise ValueError("maximum_sources must be positive")
    bundle_path = Path(development_bundle).resolve()
    freeze_path = Path(frozen_voi_report).resolve()
    destination = Path(output_root).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite validation freeze: {destination}")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        bundle.get("schema") != "utility_sft_development_bundle_v1"
        or bundle.get("test_data_present") is not False
        or bundle.get("formal_test_eligible") is not False
        or _mapping(bundle.get("split_audit"), "bundle split audit").get("passed") is not True
    ):
        raise ValueError("invalid development-only Utility-SFT bundle")
    if (
        freeze.get("schema") != "predictability_matrix_freeze_report_v2"
        or freeze.get("test_data_present") is not False
        or not isinstance(freeze.get("model_sha256"), str)
        or len(freeze["model_sha256"]) != 64
    ):
        raise ValueError("invalid frozen VOI development report")

    stage = destination.with_name(destination.name + ".staging")
    if stage.exists():
        raise FileExistsError(f"staging output already exists: {stage}")
    stage.mkdir(parents=True)
    inventory: dict[str, Any] = {}
    try:
        for benchmark in BENCHMARKS:
            source = _mapping(bundle["inventory"], "bundle inventory")[
                f"{benchmark}.validation"
            ]
            source_path = Path(str(source["path"])).resolve()
            if sha256_file(source_path) != source["sha256"]:
                raise ValueError(f"{benchmark} full validation dataset changed")
            full_payload = json.loads(source_path.read_text(encoding="utf-8"))
            full_samples = load_utility_development(source_path, role="validation")
            if {sample.benchmark for sample in full_samples} != {benchmark}:
                raise ValueError(f"{benchmark} validation identity mismatch")
            if any(len(sample.replicate_ids) != 1 for sample in full_samples):
                raise ValueError("frozen VOI alignment requires the single-seed MVP")

            frozen_benchmark = _mapping(freeze["benchmarks"][benchmark], benchmark)
            selection = _mapping(
                frozen_benchmark["frozen_policy_selection"],
                f"{benchmark} frozen policy selection",
            )
            raw_calls = selection["selected_deployable_validation_calls"]
            if (
                not isinstance(raw_calls, list)
                or len(raw_calls) != len(full_samples)
                or any(type(value) is not bool for value in raw_calls)
            ):
                raise ValueError(f"{benchmark} frozen VOI call coverage mismatch")
            decisions = sorted(
                (sample.inputs.state.state_id, sample.replicate_ids[0])
                for sample in full_samples
            )
            expected = [
                (sample.inputs.state.state_id, sample.replicate_ids[0])
                for sample in full_samples
            ]
            if decisions != expected:
                raise ValueError("utility samples are not in frozen decision order")
            calls_by_state = {
                state_id: call
                for (state_id, _), call in zip(decisions, raw_calls, strict=True)
            }

            selected = source_hash_subset(
                full_samples,
                maximum_sources=maximum_sources,
                seed=seed,
                namespace=f"utility-development-pilot-v1:{benchmark}:validation",
            )
            selected_ids = {sample.inputs.state.state_id for sample in selected}
            selected_rows = [
                row
                for row in full_payload["samples"]
                if row["inputs"]["state_id"] in selected_ids
            ]
            if [row["inputs"]["state_id"] for row in selected_rows] != [
                sample.inputs.state.state_id for sample in selected
            ]:
                raise ValueError("materialized validation order differs from pilot order")
            dataset_payload = {
                "schema": "utility_sft_dataset_v1",
                "role": "validation",
                "benchmark": benchmark,
                "formal_test_eligible": False,
                "aggregation": full_payload["aggregation"],
                "samples": selected_rows,
                "split_audit": audit_utility_splits(selected),
                "provenance": {
                    "parent_dataset": str(source_path),
                    "parent_dataset_sha256": source["sha256"],
                    "selection": "whole-source SHA256; outcome independent",
                    "namespace": f"utility-development-pilot-v1:{benchmark}:validation",
                    "seed": seed,
                    "maximum_sources": maximum_sources,
                    "test_accessed": False,
                },
            }
            benchmark_dir = stage / benchmark
            dataset_path = benchmark_dir / "validation.json"
            atomic_json_write_exclusive(dataset_path, dataset_payload)
            dataset_sha256 = sha256_file(dataset_path)
            call_payload = {
                "schema": "frozen_voi_decisions_v1",
                "role": "validation",
                "benchmark": benchmark,
                "dataset_sha256": dataset_sha256,
                "frozen_model_sha256": freeze["model_sha256"],
                "policy": {
                    "selected_cell_keys": selection["selected_deployable_cell_keys"],
                    "aggregation": "strict_seed_majority_with_even_tie_as_no_call",
                    "action_when_called": "full_exhaustive_UG_with_four_call_cost",
                    "selection_role": "prior_validation_freeze",
                    "recalibrated_for_utility_sft": False,
                },
                "calls": {
                    sample.inputs.state.state_id: calls_by_state[
                        sample.inputs.state.state_id
                    ]
                    for sample in selected
                },
            }
            calls_path = benchmark_dir / "frozen-voi-decisions.json"
            atomic_json_write_exclusive(calls_path, call_payload)
            inventory[benchmark] = {
                "dataset": str(destination / benchmark / "validation.json"),
                "dataset_sha256": dataset_sha256,
                "frozen_voi_decisions": str(
                    destination / benchmark / "frozen-voi-decisions.json"
                ),
                "frozen_voi_decisions_sha256": sha256_file(calls_path),
                "states": len(selected),
                "sources": len(
                    {sample.inputs.state.source_id for sample in selected}
                ),
                "frozen_voi_calls": sum(call_payload["calls"].values()),
            }
        report = {
            "schema": "utility_sft_validation_freeze_v1",
            "formal_claim_eligible": False,
            "test_data_present": False,
            "development_bundle": str(bundle_path),
            "development_bundle_sha256": sha256_file(bundle_path),
            "frozen_voi_report": str(freeze_path),
            "frozen_voi_report_sha256": sha256_file(freeze_path),
            "frozen_voi_model_sha256": freeze["model_sha256"],
            "selection": {
                "seed": seed,
                "maximum_sources_per_benchmark": maximum_sources,
                "uses_model_outcomes": False,
            },
            "inventory": inventory,
        }
        atomic_json_write_exclusive(stage / "VALIDATION_FREEZE.json", report)
        stage.replace(destination)
    except BaseException:
        # Keep staging for forensic inspection; callers must explicitly remove it.
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-bundle", required=True)
    parser.add_argument("--frozen-voi-report", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--maximum-sources", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    result = freeze_validation(
        development_bundle=args.development_bundle,
        frozen_voi_report=args.frozen_voi_report,
        output_root=args.output_root,
        maximum_sources=args.maximum_sources,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output_root).resolve()),
                "sha256": sha256_file(Path(args.output_root) / "VALIDATION_FREEZE.json"),
                "inventory": result["inventory"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
