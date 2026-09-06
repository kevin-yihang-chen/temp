"""Bind three completed development arms to the frozen validation subsets.

The exported score is the model's ANSWER-anchored head output.  No outcome-
based rescaling, threshold fitting, or test access is permitted here.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from beyond_entropy.predictability_matrix_artifacts import (
    atomic_json_write_exclusive,
    sha256_file,
)
from beyond_entropy.utility_dataset import load_utility_development


BENCHMARKS = ("chartqa", "docvqa", "hrbench")
ARM_BY_METHOD = {
    "format": "format_sft",
    "best_action": "best_action_sft",
    "utility": "utility_sft",
}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _load_report(path: str | Path, *, development_bundle_sha256: str) -> dict[str, Any]:
    report_path = Path(path).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    method = report.get("method")
    valid_report_contracts = {
        "utility_sft_development_pilot_v1": "three_domain_development_pilot",
        "utility_sft_development_correction_v1": "three_domain_development_correction",
    }
    if (
        report.get("schema") not in valid_report_contracts
        or method not in ARM_BY_METHOD
        or report.get("test_accessed") is not False
        or report.get("formal_claim_eligible") is not False
    ):
        raise ValueError(f"invalid development report: {report_path}")
    provenance = _mapping(report.get("provenance"), "report provenance")
    config = _mapping(provenance.get("config"), "report config")
    if (
        provenance.get("development_bundle_sha256") != development_bundle_sha256
        or config.get("method") != method
        or config.get("scope") != valid_report_contracts[report["schema"]]
        or config.get("test_authorized") is not False
        or int(config.get("validation_sources_per_benchmark", 0)) <= 0
    ):
        raise ValueError("development report was not trained under the frozen contract")
    selector = report_path.parent / "selector.pt"
    if (
        not selector.is_file()
        or report.get("selector_sha256") != sha256_file(selector)
        or not isinstance(report.get("selector_sha256"), str)
    ):
        raise ValueError("selector checkpoint/report hash mismatch")
    report["_path"] = str(report_path)
    report["_sha256"] = sha256_file(report_path)
    return report


def _arm_for_benchmark(
    report: Mapping[str, Any], benchmark: str, expected: Mapping[str, int]
) -> dict[str, Any]:
    rows = [
        row
        for row in _mapping(report["validation"], "validation")["predictions"]
        if row.get("benchmark") == benchmark
    ]
    by_state = {str(row.get("state_id")): row for row in rows}
    if len(by_state) != len(rows) or set(by_state) != set(expected):
        raise ValueError(f"{report['method']} {benchmark} prediction coverage mismatch")
    temperature = float(report["provenance"]["config"]["temperature"])
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("invalid training temperature")
    gains: dict[str, list[float]] = {}
    measurements: dict[str, Mapping[str, Any]] = {}
    for state_id in sorted(by_state):
        row = by_state[state_id]
        predicted = row.get("predicted_gain")
        logits = row.get("action_logits")
        if (
            not isinstance(predicted, list)
            or len(predicted) != expected[state_id]
            or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in predicted)
            or abs(float(predicted[0])) > 1e-6
            or not isinstance(logits, list)
            or len(logits) != len(predicted)
            or any(
                not isinstance(logit, (int, float))
                or not math.isfinite(logit)
                or abs(float(logit) * temperature - float(gain)) > 2e-5
                for logit, gain in zip(logits, predicted, strict=True)
            )
        ):
            raise ValueError(f"{report['method']} has invalid anchored scores for {state_id}")
        measurement = _mapping(row.get("measurement"), "selector measurement")
        if (
            measurement.get("vision_encoder_calls") != 1
            or measurement.get("candidate_crop_executions") != 0
            or not isinstance(measurement.get("original_image_tokens"), int)
            or measurement["original_image_tokens"] <= 0
        ):
            raise ValueError("selector measurement violates O(1) original-image inference")
        gains[state_id] = [float(value) for value in predicted]
        measurements[state_id] = dict(measurement)
    return {
        "predicted_gain": gains,
        "selector_measurements": measurements,
        "checkpoint_sha256": report["selector_sha256"],
    }


def freeze_predictions(
    *,
    validation_freeze: str | Path,
    reports: list[str | Path],
    output_root: str | Path,
) -> dict[str, Any]:
    freeze_path = Path(validation_freeze).resolve()
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("schema") != "utility_sft_validation_freeze_v1"
        or freeze.get("test_data_present") is not False
        or freeze.get("formal_claim_eligible") is not False
    ):
        raise ValueError("invalid validation freeze")
    loaded = [
        _load_report(
            path,
            development_bundle_sha256=freeze["development_bundle_sha256"],
        )
        for path in reports
    ]
    by_method = {report["method"]: report for report in loaded}
    if len(loaded) != 3 or set(by_method) != set(ARM_BY_METHOD):
        raise ValueError("exactly Format, Best-Action, and Utility reports are required")
    matched_configs = []
    for report in loaded:
        config = dict(report["provenance"]["config"])
        config.pop("method")
        matched_configs.append(config)
    if not matched_configs[0] == matched_configs[1] == matched_configs[2]:
        raise ValueError("three development arms are not matched except objective")

    destination = Path(output_root).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite prediction freeze: {destination}")
    stage = destination.with_name(destination.name + ".staging")
    if stage.exists():
        raise FileExistsError(f"prediction staging output exists: {stage}")
    stage.mkdir(parents=True)
    inventory: dict[str, Any] = {}
    for benchmark in BENCHMARKS:
        frozen_entry = freeze["inventory"][benchmark]
        dataset_path = Path(frozen_entry["dataset"]).resolve()
        if sha256_file(dataset_path) != frozen_entry["dataset_sha256"]:
            raise ValueError(f"{benchmark} frozen validation dataset changed")
        samples = load_utility_development(dataset_path, role="validation")
        expected = {
            sample.inputs.state.state_id: len(sample.gains) for sample in samples
        }
        payload = {
            "schema": "utility_sft_predictions_v1",
            "role": "validation",
            "benchmark": benchmark,
            "dataset_sha256": frozen_entry["dataset_sha256"],
            "score_contract": {
                "definition": "ANSWER-anchored action_logit_times_training_temperature",
                "outcome_based_rescaling": False,
                "threshold_fitting": False,
                "cost_applied_only_by_evaluation_policy": True,
            },
            "arms": {
                ARM_BY_METHOD[method]: _arm_for_benchmark(
                    by_method[method], benchmark, expected
                )
                for method in ARM_BY_METHOD
            },
        }
        output = stage / benchmark / "predictions.json"
        atomic_json_write_exclusive(output, payload)
        inventory[benchmark] = {
            "predictions": str(destination / benchmark / "predictions.json"),
            "predictions_sha256": sha256_file(output),
            "dataset_sha256": frozen_entry["dataset_sha256"],
            "states": len(samples),
        }
    report = {
        "schema": "utility_sft_prediction_freeze_v1",
        "formal_claim_eligible": False,
        "test_data_present": False,
        "validation_freeze": str(freeze_path),
        "validation_freeze_sha256": sha256_file(freeze_path),
        "reports": {
            ARM_BY_METHOD[item["method"]]: {
                "path": item["_path"],
                "sha256": item["_sha256"],
                "checkpoint_sha256": item["selector_sha256"],
            }
            for item in loaded
        },
        "three_arm_config_audit": "matched_except_method",
        "score_calibration": "none",
        "inventory": inventory,
    }
    atomic_json_write_exclusive(stage / "PREDICTION_FREEZE.json", report)
    stage.replace(destination)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-freeze", required=True)
    parser.add_argument("--format-report", required=True)
    parser.add_argument("--best-action-report", required=True)
    parser.add_argument("--utility-report", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    report = freeze_predictions(
        validation_freeze=args.validation_freeze,
        reports=[args.format_report, args.best_action_report, args.utility_report],
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output_root).resolve()),
                "sha256": sha256_file(Path(args.output_root) / "PREDICTION_FREEZE.json"),
                "inventory": report["inventory"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
