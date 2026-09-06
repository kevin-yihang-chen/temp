import json
from pathlib import Path

import pytest

from beyond_entropy.predictability_matrix_artifacts import sha256_file
from beyond_entropy.utility_dataset import audit_utility_splits
from scripts.freeze_utility_sft_validation import freeze_validation
from tests.test_utility_sft import make_samples


def _dataset(path: Path, benchmark: str):
    samples = make_samples(benchmark=benchmark, role="validation")
    payload = {
        "schema": "utility_sft_dataset_v1",
        "role": "validation",
        "benchmark": benchmark,
        "formal_test_eligible": False,
        "aggregation": "single",
        "samples": [sample.to_dict() for sample in samples],
        "split_audit": audit_utility_splits(samples),
        "provenance": {},
    }
    path.write_text(json.dumps(payload))
    return samples


def test_freezes_matched_validation_and_prior_voi_calls(tmp_path: Path):
    inventory = {}
    for benchmark in ("chartqa", "docvqa", "hrbench"):
        path = tmp_path / f"{benchmark}.json"
        _dataset(path, benchmark)
        inventory[f"{benchmark}.validation"] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "states": 1,
            "sources": 1,
        }
    bundle = tmp_path / "bundle.json"
    bundle.write_text(
        json.dumps(
            {
                "schema": "utility_sft_development_bundle_v1",
                "test_data_present": False,
                "formal_test_eligible": False,
                "split_audit": {"passed": True},
                "inventory": inventory,
            }
        )
    )
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "schema": "predictability_matrix_freeze_report_v2",
                "test_data_present": False,
                "model_sha256": "f" * 64,
                "benchmarks": {
                    benchmark: {
                        "frozen_policy_selection": {
                            "selected_deployable_cell_keys": [
                                {"level": "l0_uncertainty", "target": "direct_gain", "seed": 17}
                            ],
                            "selected_deployable_validation_calls": [benchmark == "docvqa"],
                        }
                    }
                    for benchmark in ("chartqa", "docvqa", "hrbench")
                },
            }
        )
    )
    output = tmp_path / "output"
    report = freeze_validation(
        development_bundle=bundle,
        frozen_voi_report=freeze,
        output_root=output,
        maximum_sources=1,
        seed=17,
    )
    assert report["test_data_present"] is False
    assert report["inventory"]["docvqa"]["frozen_voi_calls"] == 1
    assert Path(report["inventory"]["docvqa"]["dataset"]).is_file()
    assert Path(
        report["inventory"]["docvqa"]["frozen_voi_decisions"]
    ).is_file()
    decisions = json.loads(
        (output / "docvqa" / "frozen-voi-decisions.json").read_text()
    )
    dataset = output / "docvqa" / "validation.json"
    assert decisions["dataset_sha256"] == sha256_file(dataset)
    assert decisions["calls"] == {"s1": True}


def test_refuses_overwrite_and_call_coverage_mismatch(tmp_path: Path):
    inventory = {}
    for benchmark in ("chartqa", "docvqa", "hrbench"):
        path = tmp_path / f"{benchmark}.json"
        _dataset(path, benchmark)
        inventory[f"{benchmark}.validation"] = {
            "path": str(path), "sha256": sha256_file(path)
        }
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({
        "schema": "utility_sft_development_bundle_v1", "test_data_present": False,
        "formal_test_eligible": False, "split_audit": {"passed": True},
        "inventory": inventory,
    }))
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({
        "schema": "predictability_matrix_freeze_report_v2", "test_data_present": False,
        "model_sha256": "f" * 64,
        "benchmarks": {benchmark: {"frozen_policy_selection": {
            "selected_deployable_cell_keys": [],
            "selected_deployable_validation_calls": [],
        }} for benchmark in ("chartqa", "docvqa", "hrbench")},
    }))
    with pytest.raises(ValueError, match="coverage"):
        freeze_validation(
            development_bundle=bundle, frozen_voi_report=freeze,
            output_root=tmp_path / "bad", maximum_sources=1, seed=17,
        )
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        freeze_validation(
            development_bundle=bundle, frozen_voi_report=freeze,
            output_root=existing, maximum_sources=1, seed=17,
        )
