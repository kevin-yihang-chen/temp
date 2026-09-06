import json
from pathlib import Path

import pytest

from beyond_entropy.predictability_matrix_artifacts import sha256_file
from scripts.freeze_utility_sft_predictions import freeze_predictions
from tests.test_freeze_utility_sft_validation import _dataset


def _setup(tmp_path: Path):
    inventory = {}
    for benchmark in ("chartqa", "docvqa", "hrbench"):
        dataset = tmp_path / benchmark / "validation.json"
        dataset.parent.mkdir(parents=True)
        _dataset(dataset, benchmark)
        inventory[benchmark] = {
            "dataset": str(dataset), "dataset_sha256": sha256_file(dataset)
        }
    validation = tmp_path / "VALIDATION_FREEZE.json"
    validation.write_text(json.dumps({
        "schema": "utility_sft_validation_freeze_v1",
        "test_data_present": False, "formal_claim_eligible": False,
        "development_bundle_sha256": "d"*64, "inventory": inventory,
    }))
    reports = []
    for method in ("format", "best_action", "utility"):
        run = tmp_path / f"run-{method}"
        run.mkdir()
        selector = run / "selector.pt"
        selector.write_bytes(method.encode())
        predictions = [{
            "benchmark": benchmark, "state_id": "s1", "source_id": "source1",
            "gains": [0, .5, -.5], "predicted_gain": [0, .25, -.25],
            "action_logits": [0, 1, -1], "best_action": 1, "support_action": 0,
            "measurement": {"original_image_tokens": 16, "prompt_tokens": 8,
                            "vision_encoder_calls": 1, "candidate_crop_executions": 0},
        } for benchmark in ("chartqa", "docvqa", "hrbench")]
        report = run / "report.json"
        report.write_text(json.dumps({
            "schema": "utility_sft_development_pilot_v1", "method": method,
            "formal_claim_eligible": False, "test_accessed": False,
            "provenance": {"development_bundle_sha256": "d"*64, "config": {
                "method": method, "scope": "three_domain_development_pilot",
                "test_authorized": False, "validation_sources_per_benchmark": 1,
                "temperature": .25, "same": 1,
            }},
            "validation": {"predictions": predictions},
            "selector_sha256": sha256_file(selector),
        }))
        reports.append(report)
    return validation, reports


def test_freezes_three_matched_arms_without_rescaling(tmp_path: Path):
    validation, reports = _setup(tmp_path)
    output = tmp_path / "predictions"
    report = freeze_predictions(
        validation_freeze=validation, reports=reports, output_root=output
    )
    assert report["score_calibration"] == "none"
    payload = json.loads((output / "chartqa" / "predictions.json").read_text())
    assert set(payload["arms"]) == {
        "format_sft", "best_action_sft", "utility_sft"
    }
    assert payload["arms"]["utility_sft"]["predicted_gain"]["s1"] == [0, .25, -.25]
    assert Path(report["inventory"]["chartqa"]["predictions"]).is_file()


def test_accepts_three_matched_coverage_correction_reports(tmp_path: Path):
    validation, reports = _setup(tmp_path)
    for path in reports:
        payload = json.loads(path.read_text())
        payload["schema"] = "utility_sft_development_correction_v1"
        payload["provenance"]["config"]["scope"] = (
            "three_domain_development_correction"
        )
        path.write_text(json.dumps(payload))
    report = freeze_predictions(
        validation_freeze=validation,
        reports=reports,
        output_root=tmp_path / "correction-predictions",
    )
    assert report["three_arm_config_audit"] == "matched_except_method"


def test_rejects_post_action_execution_and_unmatched_configs(tmp_path: Path):
    validation, reports = _setup(tmp_path)
    payload = json.loads(reports[0].read_text())
    payload["validation"]["predictions"][0]["measurement"]["candidate_crop_executions"] = 1
    reports[0].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=r"O\(1\)"):
        freeze_predictions(
            validation_freeze=validation, reports=reports,
            output_root=tmp_path / "bad",
        )

    validation, reports = _setup(tmp_path / "second")
    payload = json.loads(reports[0].read_text())
    payload["provenance"]["config"]["same"] = 2
    reports[0].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="not matched"):
        freeze_predictions(
            validation_freeze=validation, reports=reports,
            output_root=tmp_path / "unmatched",
        )
