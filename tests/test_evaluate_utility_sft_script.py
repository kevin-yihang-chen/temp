from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from beyond_entropy.predictability_matrix_artifacts import sha256_file
from test_utility_sft import make_samples


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_utility_sft.py"
SPEC = importlib.util.spec_from_file_location("evaluate_utility_sft", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixtures(tmp_path):
    sample = make_samples(role="validation")[0]
    dataset = tmp_path / "validation.json"
    dataset.write_text(json.dumps({
        "schema": "utility_sft_dataset_v1", "role": "validation", "benchmark": "chartqa",
        "aggregation": "single", "samples": [sample.to_dict()],
    }))
    data_hash = sha256_file(dataset)
    entry = {"predicted_gain": {"s1": [0, .5, -.5]},
             "selector_measurements": {"s1": {"original_image_tokens": 32,
                                                  "candidate_crop_executions": 0}},
             "checkpoint_sha256": "a"*64}
    predictions = tmp_path / "predictions.json"
    predictions.write_text(json.dumps({
        "schema": "utility_sft_predictions_v1", "role": "validation",
        "benchmark": "chartqa", "dataset_sha256": data_hash,
        "arms": {arm: entry for arm in MODULE.ARMS},
    }))
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({
        "schema": "frozen_voi_decisions_v1", "role": "validation", "benchmark": "chartqa",
        "dataset_sha256": data_hash, "frozen_model_sha256": "b"*64,
        "calls": {"s1": True},
    }))
    return dataset, predictions, frozen


def test_deterministic_evaluator_requires_and_reports_all_policies(tmp_path):
    dataset, predictions, frozen = fixtures(tmp_path)
    output = tmp_path / "report.json"
    report = MODULE.evaluate(str(dataset), str(predictions), str(frozen), str(output), resamples=20)
    assert report["lambdas"] == list(MODULE.LAMBDAS)
    primary = report["frontier"]["0.05"]
    assert set(primary["policies"]) == {
        "answer_only", "random_crop", "ug", "frozen_voi",
        "format_sft", "best_action_sft", "utility_sft", "oracle",
    }
    assert primary["policies"]["ug"]["source_balanced"]["avg_tool_calls"] == 2
    assert primary["policies"]["utility_sft"]["source_balanced"]["avg_tool_calls"] == 1
    assert primary["policies"]["utility_sft"]["selector_overhead"]["candidate_crop_executions"] == 0
    assert set(primary["paired"]) == {"utility_minus_frozen_voi", "utility_minus_best_action"}
    with pytest.raises(FileExistsError):
        MODULE.evaluate(str(dataset), str(predictions), str(frozen), str(output), resamples=20)


def test_evaluator_rejects_candidate_crop_leakage(tmp_path):
    dataset, predictions, frozen = fixtures(tmp_path)
    payload = json.loads(predictions.read_text())
    payload["arms"]["utility_sft"]["selector_measurements"]["s1"]["candidate_crop_executions"] = 1
    predictions.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="executed a candidate crop"):
        MODULE.evaluate(str(dataset), str(predictions), str(frozen), str(tmp_path / "bad.json"), resamples=20)
