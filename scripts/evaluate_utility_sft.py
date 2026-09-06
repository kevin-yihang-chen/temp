"""Deterministically evaluate all eight required policies on validation data.

Formal test evaluation will call the same core from a separately frozen,
ledger-first worker. This CLI intentionally rejects test roles.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from beyond_entropy.predictability_matrix_artifacts import atomic_json_write_exclusive, sha256_file
from beyond_entropy.utility_dataset import load_utility_development
from beyond_entropy.utility_evaluation import choice_metrics, paired_choice_interval, policy_choices


LAMBDAS = (0.0, 0.01, 0.025, 0.05, 0.1, 0.2)
ARMS = ("format_sft", "best_action_sft", "utility_sft")


def _load_predictions(path: str, *, dataset_sha256: str, benchmark: str) -> tuple[dict, dict]:
    payload = json.loads(Path(path).read_text())
    if (payload.get("schema") != "utility_sft_predictions_v1"
            or payload.get("role") != "validation"
            or payload.get("benchmark") != benchmark
            or payload.get("dataset_sha256") != dataset_sha256
            or set(payload.get("arms", {})) != set(ARMS)):
        raise ValueError("prediction identity/schema/three-arm coverage mismatch")
    predictions, overhead = {}, {}
    for arm in ARMS:
        entry = payload["arms"][arm]
        if set(entry) != {"predicted_gain", "selector_measurements", "checkpoint_sha256"}:
            raise ValueError("prediction arm has unexpected or missing fields")
        predictions[arm] = entry["predicted_gain"]
        if not isinstance(entry["checkpoint_sha256"], str) or len(entry["checkpoint_sha256"]) != 64:
            raise ValueError("checkpoint hash required")
        overhead[arm] = entry["selector_measurements"]
    return predictions, overhead


def _load_frozen_voi(path: str, *, dataset_sha256: str, benchmark: str) -> dict[str, bool]:
    payload = json.loads(Path(path).read_text())
    if (payload.get("schema") != "frozen_voi_decisions_v1"
            or payload.get("role") != "validation"
            or payload.get("benchmark") != benchmark
            or payload.get("dataset_sha256") != dataset_sha256
            or not isinstance(payload.get("frozen_model_sha256"), str)):
        raise ValueError("frozen VOI identity/schema mismatch")
    calls = payload.get("calls")
    if not isinstance(calls, dict) or any(type(v) is not bool for v in calls.values()):
        raise ValueError("frozen VOI decisions must be explicit booleans")
    return calls


def evaluate(dataset: str, predictions: str, frozen_voi: str, output: str,
             *, resamples: int = 20000, bootstrap_seed: int = 17) -> dict:
    if resamples <= 0:
        raise ValueError("positive bootstrap count required")
    data_hash = sha256_file(dataset)
    samples = load_utility_development(dataset, role="validation")
    benchmark = samples[0].benchmark
    gains, overhead = _load_predictions(
        predictions, dataset_sha256=data_hash, benchmark=benchmark
    )
    calls = _load_frozen_voi(frozen_voi, dataset_sha256=data_hash, benchmark=benchmark)
    state_ids = {s.inputs.state.state_id for s in samples}
    if any(set(gains[arm]) != state_ids for arm in ARMS):
        raise ValueError("learned prediction state coverage mismatch")
    for arm in ARMS:
        for state, values in gains[arm].items():
            expected = len(next(s for s in samples if s.inputs.state.state_id == state).gains)
            if (not isinstance(values, list) or len(values) != expected
                    or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in values)
                    or abs(values[0]) > 1e-6):
                raise ValueError("learned predictions must be finite, complete and ANSWER-anchored")
    if set(calls) != state_ids:
        raise ValueError("frozen VOI state coverage mismatch")
    frontier = {}
    for lambda_cost in LAMBDAS:
        choices = policy_choices(
            samples, lambda_cost=lambda_cost, learned_gains=gains,
            frozen_voi_calls=calls, random_seed=17,
        )
        by_policy = {
            name: choice_metrics(samples, selected, lambda_cost=lambda_cost)
            for name, selected in choices.items()
        }
        for arm in ARMS:
            measurements = overhead[arm]
            if set(measurements) != state_ids:
                raise ValueError("selector measurement coverage mismatch")
            image_tokens = [measurements[state]["original_image_tokens"] for state in state_ids]
            if any(measurements[state].get("candidate_crop_executions") != 0 for state in state_ids):
                raise ValueError("learned selector executed a candidate crop")
            by_policy[arm]["selector_overhead"] = {
                "original_image_tokens_question_weighted": sum(image_tokens)/len(image_tokens),
                "candidate_crop_executions": 0,
            }
        paired = {
            "utility_minus_frozen_voi": paired_choice_interval(
                samples, choices["utility_sft"], choices["frozen_voi"],
                lambda_cost=lambda_cost, resamples=resamples, seed=bootstrap_seed,
            ),
            "utility_minus_best_action": paired_choice_interval(
                samples, choices["utility_sft"], choices["best_action_sft"],
                lambda_cost=lambda_cost, resamples=resamples, seed=bootstrap_seed+1,
            ),
        }
        frontier[str(lambda_cost)] = {"policies": by_policy, "paired": paired}
    report = {
        "schema": "utility_sft_evaluation_v1", "formal_claim_eligible": False,
        "role": "validation", "benchmark": benchmark,
        "dataset": {"path": str(Path(dataset).resolve()), "sha256": data_hash},
        "predictions": {"path": str(Path(predictions).resolve()), "sha256": sha256_file(predictions)},
        "frozen_voi": {"path": str(Path(frozen_voi).resolve()), "sha256": sha256_file(frozen_voi)},
        "lambdas": list(LAMBDAS), "primary_lambda": 0.05,
        "bootstrap": {"resamples": resamples, "unit": "source_id", "confidence_level": 0.95,
                      "seeds": [bootstrap_seed, bootstrap_seed+1]},
        "frontier": frontier,
    }
    atomic_json_write_exclusive(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--frozen-voi", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resamples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    args = parser.parse_args()
    report = evaluate(args.validation_data, args.predictions, args.frozen_voi, args.output,
                      resamples=args.resamples, bootstrap_seed=args.bootstrap_seed)
    print(json.dumps({"output": str(Path(args.output).resolve()),
                      "sha256": sha256_file(args.output), "benchmark": report["benchmark"]}))


if __name__ == "__main__":
    main()
