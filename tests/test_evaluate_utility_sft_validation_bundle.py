import json
from pathlib import Path

import pytest

from beyond_entropy.predictability_matrix_artifacts import sha256_file
from scripts.evaluate_utility_sft_validation_bundle import evaluate_bundle
from tests.test_utility_sft import make_samples


def _fixtures(tmp_path: Path):
    validation_inventory = {}
    prediction_inventory = {}
    for benchmark in ("chartqa", "docvqa", "hrbench"):
        directory = tmp_path / benchmark
        directory.mkdir()
        sample = make_samples(benchmark=benchmark, role="validation")[0]
        dataset = directory / "validation.json"
        dataset.write_text(json.dumps({
            "schema": "utility_sft_dataset_v1", "role": "validation",
            "benchmark": benchmark, "aggregation": "single",
            "samples": [sample.to_dict()],
        }))
        dataset_sha = sha256_file(dataset)
        frozen = directory / "frozen.json"
        frozen.write_text(json.dumps({
            "schema": "frozen_voi_decisions_v1", "role": "validation",
            "benchmark": benchmark, "dataset_sha256": dataset_sha,
            "frozen_model_sha256": "f"*64, "calls": {"s1": False},
        }))
        entry = {
            "predicted_gain": {"s1": [0, .5, -.5]},
            "selector_measurements": {"s1": {
                "original_image_tokens": 16, "candidate_crop_executions": 0,
            }},
            "checkpoint_sha256": "c"*64,
        }
        predictions = directory / "predictions.json"
        predictions.write_text(json.dumps({
            "schema": "utility_sft_predictions_v1", "role": "validation",
            "benchmark": benchmark, "dataset_sha256": dataset_sha,
            "arms": {name: entry for name in (
                "format_sft", "best_action_sft", "utility_sft"
            )},
        }))
        validation_inventory[benchmark] = {
            "dataset": str(dataset), "dataset_sha256": dataset_sha,
            "frozen_voi_decisions": str(frozen), "states": 1, "sources": 1,
        }
        prediction_inventory[benchmark] = {
            "predictions": str(predictions), "dataset_sha256": dataset_sha,
        }
    validation = tmp_path / "VALIDATION_FREEZE.json"
    validation.write_text(json.dumps({
        "schema": "utility_sft_validation_freeze_v1", "test_data_present": False,
        "formal_claim_eligible": False, "inventory": validation_inventory,
    }))
    prediction = tmp_path / "PREDICTION_FREEZE.json"
    prediction.write_text(json.dumps({
        "schema": "utility_sft_prediction_freeze_v1", "test_data_present": False,
        "validation_freeze": str(validation.resolve()),
        "validation_freeze_sha256": sha256_file(validation),
        "inventory": prediction_inventory,
    }))
    return validation, prediction


def test_evaluates_bound_three_domain_bundle(tmp_path: Path):
    validation, predictions = _fixtures(tmp_path)
    output = tmp_path / "evaluation"
    report = evaluate_bundle(
        validation_freeze=validation, prediction_freeze=predictions,
        output_root=output, resamples=10, bootstrap_seed=17,
    )
    assert set(report["inventory"]) == {"chartqa", "docvqa", "hrbench"}
    assert all(Path(item["evaluation"]).is_file() for item in report["inventory"].values())
    with pytest.raises(FileExistsError):
        evaluate_bundle(
            validation_freeze=validation, prediction_freeze=predictions,
            output_root=output, resamples=10,
        )
