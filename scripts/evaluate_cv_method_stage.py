#!/usr/bin/env python3
"""Evaluate matched-cost post-training arms and make the frozen stage decision."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from beyond_entropy.predictability_matrix_artifacts import atomic_json_write_exclusive, sha256_file
from beyond_entropy.sequential_metrics import (
    paired_source_bootstrap_accuracy_delta,
    paired_source_bootstrap_utility_delta,
    policy_metrics,
    sequential_diagnostic,
)
from beyond_entropy.sequential_schema import SequentialRolloutRecord


BENCHMARKS = ("chartqa", "docvqa")
RATES = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
LAMBDAS = (0.0, 0.025, 0.05, 0.1, 0.2)
BASELINE_ORDER = ("entropy", "confidence", "margin")


def read_records(path: str | Path) -> list[SequentialRolloutRecord]:
    result = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                result.append(SequentialRolloutRecord.from_dict(json.loads(line)))
    if not result:
        raise ValueError("empty rollout file")
    return result


def top_count_mask(scores: list[float], count: int) -> list[bool]:
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    chosen = set(order[:count])
    return [index in chosen for index in range(len(scores))]


def calibration(scores: list[float], records: list[SequentialRolloutRecord]) -> dict:
    order = sorted(range(len(scores)), key=lambda index: (scores[index], index))
    bins = []
    for bin_index in range(5):
        lower = round(bin_index * len(order) / 5)
        upper = round((bin_index + 1) * len(order) / 5)
        indices = order[lower:upper]
        if indices:
            bins.append({
                "bin": bin_index, "states": len(indices),
                "mean_score": mean(scores[index] for index in indices),
                "mean_gain": mean(records[index].delta_success for index in indices),
                "beneficial_rate": mean(records[index].delta_success > 0 for index in indices),
            })
    return {"quantile_bins": bins}


def load_report(path: str | Path, expected_method: str) -> dict:
    report = json.loads(Path(path).read_text())
    if (report.get("schema") != "cv_method_post_training_report_v1"
            or report.get("method") != expected_method
            or report.get("test_accessed") is not False):
        raise ValueError(f"invalid {expected_method} report")
    return report


def accuracy(records: list[SequentialRolloutRecord], mask: list[bool]) -> float:
    return mean(record.continue_correct if use else record.stop_correct
                for record, use in zip(records, mask))


def phase_b_decision(benchmarks: dict) -> tuple[str, str]:
    versus_baseline = {
        benchmark: benchmarks[benchmark]["primary_comparisons"]
        ["counterfactual_minus_strongest_uncertainty"]["accuracy"]["observed_delta"]
        for benchmark in BENCHMARKS
    }
    versus_outcome = {
        benchmark: benchmarks[benchmark]["primary_comparisons"]
        ["counterfactual_minus_outcome_only"]["accuracy"]["observed_delta"]
        for benchmark in BENCHMARKS
    }
    if all(value <= 0 for value in versus_baseline.values()):
        reason = "counterfactual did not improve over the strongest matched baseline on either benchmark"
    elif all(value <= 0 for value in versus_outcome.values()):
        reason = "counterfactual did not improve over outcome-only on either benchmark"
    else:
        phase_b_go = (
            any(value > .01 for value in versus_baseline.values())
            and all(value > -.01 for value in versus_baseline.values())
        )
        if phase_b_go:
            return "PHASE_B_GO", "the pre-registered Phase B to C transition rule was met"
        reason = "the pre-registered >+1pp and >-1pp cross-domain transition rule was not met"
    return "PHASE_B_NO_GO", reason


def evaluate_benchmark(
    benchmark: str,
    records: list[SequentialRolloutRecord],
    predictions: dict[str, dict[tuple[str, str], dict]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    ids = [record.decision_id for record in records]
    score_sets = {}
    for method in ("outcome_only", "counterfactual_utility"):
        if set(predictions[method]) != set(ids):
            raise ValueError(f"{benchmark} {method} prediction coverage mismatch")
        score_sets[method] = [
            float(predictions[method][decision]["continue_score"]) for decision in ids
        ]
    score_sets.update({
        "entropy": [record.stop_entropy for record in records],
        "confidence": [-record.stop_max_probability for record in records],
        "margin": [-record.stop_top1_top2_margin for record in records],
    })
    frontiers = []
    masks_by_rate = {}
    for rate in RATES:
        count = round(rate * len(records))
        masks = {name: top_count_mask(scores, count) for name, scores in score_sets.items()}
        masks["answer_only"] = [False] * len(records)
        masks_by_rate[rate] = masks
        accuracy_at_rate = {name: accuracy(records, mask) for name, mask in masks.items()}
        strongest = max(BASELINE_ORDER, key=lambda name: (accuracy_at_rate[name], -BASELINE_ORDER.index(name)))
        entry = {
            "target_call_rate": rate, "calls": count,
            "realized_call_rate": count / len(records),
            "strongest_uncertainty_baseline": strongest,
            "policies_by_lambda": {},
        }
        for lambda_cost in LAMBDAS:
            entry["policies_by_lambda"][str(lambda_cost)] = {
                name: policy_metrics(records, mask, lambda_cost=lambda_cost, policy_name=name)
                for name, mask in masks.items()
            }
        frontiers.append(entry)
    primary_masks = masks_by_rate[0.25]
    primary_accuracy = {name: accuracy(records, mask) for name, mask in primary_masks.items()}
    strongest = max(BASELINE_ORDER, key=lambda name: (primary_accuracy[name], -BASELINE_ORDER.index(name)))
    cf = primary_masks["counterfactual_utility"]
    outcome = primary_masks["outcome_only"]
    baseline = primary_masks[strongest]
    comparisons = {}
    for label, right in (("strongest_uncertainty", baseline), ("outcome_only", outcome)):
        comparisons[f"counterfactual_minus_{label}"] = {
            "accuracy": paired_source_bootstrap_accuracy_delta(
                records, cf, right, samples=bootstrap_samples, seed=bootstrap_seed
            ),
            "utility_lambda_0.05": paired_source_bootstrap_utility_delta(
                records, cf, right, lambda_cost=.05,
                samples=bootstrap_samples, seed=bootstrap_seed,
            ),
        }
    beneficial = sum(record.delta_success > 0 for record in records)
    secondary = {}
    for name, mask in (("outcome_only", outcome), ("counterfactual_utility", cf),
                       (strongest, baseline)):
        secondary[name] = {
            "unnecessary_continuation_count": sum(
                use and record.delta_success <= 0 for record, use in zip(records, mask)
            ),
            "missed_beneficial_count": sum(
                not use and record.delta_success > 0 for record, use in zip(records, mask)
            ),
            "beneficial_states": beneficial,
        }
    return {
        "diagnostic": sequential_diagnostic(records),
        "primary_call_rate": .25,
        "primary_strongest_uncertainty_baseline": strongest,
        "primary_accuracy": primary_accuracy,
        "primary_comparisons": comparisons,
        "secondary": secondary,
        "calibration": {
            name: calibration(score_sets[name], records)
            for name in ("outcome_only", "counterfactual_utility")
        },
        "frontier": frontiers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcome-report", required=True)
    parser.add_argument("--counterfactual-report", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if (config.get("schema") != "cv_method_evaluation_config_v1"
            or config.get("stage") not in ("phase_a_smoke", "phase_b_pilot", "phase_c_confirmation")
            or config.get("test_authorized") is not False
            or tuple(config.get("rates", ())) != RATES
            or tuple(config.get("lambdas", ())) != LAMBDAS):
        raise ValueError("invalid frozen evaluation config")
    outcome = load_report(args.outcome_report, "outcome_only")
    counterfactual = load_report(args.counterfactual_report, "counterfactual_utility")
    if outcome["stage"] != config["stage"] or counterfactual["stage"] != config["stage"]:
        raise ValueError("training and evaluation stages differ")
    if outcome["schedule_sha256"] != counterfactual["schedule_sha256"]:
        raise ValueError("training arms did not use the same state schedule")
    predictions = {}
    for method, report in (("outcome_only", outcome), ("counterfactual_utility", counterfactual)):
        predictions[method] = {benchmark: {} for benchmark in BENCHMARKS}
        for row in report["validation"]["predictions"]:
            key = (row["state_id"], row["replicate_id"])
            if key in predictions[method][row["benchmark"]]:
                raise ValueError("duplicate model prediction")
            predictions[method][row["benchmark"]][key] = row
    benchmarks = {}
    for benchmark in BENCHMARKS:
        spec = config["validation_rollouts"][benchmark]
        if sha256_file(spec["path"]) != spec["sha256"]:
            raise ValueError("validation rollout changed after evaluation freeze")
        records = read_records(spec["path"])
        outcome_ids = set(predictions["outcome_only"][benchmark])
        counterfactual_ids = set(predictions["counterfactual_utility"][benchmark])
        if outcome_ids != counterfactual_ids:
            raise ValueError("arm validation subsets differ")
        if config["stage"] == "phase_a_smoke":
            records = [record for record in records if record.decision_id in outcome_ids]
        benchmarks[benchmark] = evaluate_benchmark(
            benchmark, records,
            {method: predictions[method][benchmark] for method in predictions},
            bootstrap_samples=int(config["bootstrap_samples"]),
            bootstrap_seed=int(config["bootstrap_seed"]),
        )
    engineering_checks = {
        "outcome_only_smoke_passed": outcome.get("smoke_passed") is True,
        "counterfactual_smoke_passed": counterfactual.get("smoke_passed") is True,
        "matched_schedule": outcome["schedule_sha256"] == counterfactual["schedule_sha256"],
        "no_test_access": not outcome["test_accessed"] and not counterfactual["test_accessed"],
    }
    if config["stage"] == "phase_a_smoke":
        decision = "PHASE_A_PASS" if all(engineering_checks.values()) else "PHASE_A_FAIL"
        reason = "all engineering gates passed" if decision == "PHASE_A_PASS" else "one or more engineering gates failed"
    else:
        if config["stage"] == "phase_b_pilot":
            decision, reason = phase_b_decision(benchmarks)
        else:
            # Phase C aggregation across three seeds is intentionally separate;
            # one seed report cannot authorize scaling.
            decision, reason = "PHASE_C_SEED_COMPLETE", "await three-seed aggregation"
    report = {
        "schema": "cv_method_stage_evaluation_v1", "stage": config["stage"],
        "test_accessed": False, "primary_call_rate": .25,
        "rates": RATES, "lambdas": LAMBDAS,
        "decision": decision, "decision_reason": reason,
        "engineering_checks": engineering_checks, "benchmarks": benchmarks,
        "provenance": {
            "evaluation_config_sha256": sha256_file(args.config),
            "outcome_report_sha256": sha256_file(args.outcome_report),
            "counterfactual_report_sha256": sha256_file(args.counterfactual_report),
            "schedule_sha256": outcome["schedule_sha256"],
        },
    }
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    atomic_json_write_exclusive(output / "report.json", report)
    try:
        import matplotlib.pyplot as plt
        for benchmark in BENCHMARKS:
            frontier = benchmarks[benchmark]["frontier"]
            plt.figure(figsize=(6.5, 4.5))
            for policy in ("outcome_only", "counterfactual_utility"):
                xs = [row["realized_call_rate"] for row in frontier]
                ys = [row["policies_by_lambda"]["0.0"][policy]["accuracy"] for row in frontier]
                plt.plot(xs, ys, marker="o", label=policy)
            baseline_ys = []
            for row in frontier:
                name = row["strongest_uncertainty_baseline"]
                baseline_ys.append(row["policies_by_lambda"]["0.0"][name]["accuracy"])
            plt.plot([row["realized_call_rate"] for row in frontier], baseline_ys,
                     marker="o", label="strongest uncertainty")
            plt.xlabel("Average incremental tool calls")
            plt.ylabel("Accuracy")
            plt.title(benchmark)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(output / f"{benchmark}-accuracy-cost-frontier.png", dpi=180)
            plt.close()
    except ModuleNotFoundError:
        pass
    print(json.dumps({"report": str(output / "report.json"), "decision": decision,
                      "reason": reason}), flush=True)


if __name__ == "__main__":
    main()
