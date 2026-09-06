"""Frozen three-seed evaluation for Factorized Phase-C formal access.

The evaluator treats every trained seed as an independently deployable policy.
It averages policy outcomes across seeds; it never ensembles scores before action
selection.  Confidence intervals resample source clusters after first averaging
the paired per-decision policy difference across the three seeds.
"""
from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from statistics import mean
from typing import Any, Mapping, Sequence

from .phase_c_formal_transaction import FORMAL_MODES
from .phase_c_training import BENCHMARKS, METHODS, SEEDS
from .sequential_metrics import policy_metrics
from .sequential_schema import SequentialRolloutRecord


UNCERTAINTY_BASELINES = ("entropy", "confidence", "margin")
PRIMARY_METHOD = "factorized_potential_outcomes"
PRIMARY_CONTROL = "outcome_only"
DIRECT_CONTROL = "counterfactual_utility"
DecisionId = tuple[str, str]


def _decision_id(record: SequentialRolloutRecord) -> DecisionId:
    return record.state_id, record.replicate_id


def _top_rate_mask(
    records: Sequence[SequentialRolloutRecord], scores: Sequence[float], rate: float,
) -> list[bool]:
    if len(records) != len(scores) or not records or not 0.0 <= rate <= 1.0:
        raise ValueError("rate selection requires aligned records and scores")
    if not all(math.isfinite(float(value)) for value in scores):
        raise ValueError("selection scores must be finite")
    count = round(rate * len(records))
    order = sorted(
        range(len(records)),
        key=lambda index: (-float(scores[index]), _decision_id(records[index])),
    )
    selected = set(order[:count])
    return [index in selected for index in range(len(records))]


def _random_scores(
    records: Sequence[SequentialRolloutRecord], *, seed: int, benchmark: str,
) -> list[float]:
    return [
        float(
            int.from_bytes(
                hashlib.sha256(
                    f"phase-c-formal-random:{seed}:{benchmark}:"
                    f"{record.state_id}:{record.replicate_id}".encode()
                ).digest()[:8],
                "big",
            )
        )
        for record in records
    ]


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        average_rank = (start + stop - 1) / 2.0
        for position in range(start, stop):
            result[order[position]] = average_rank
        start = stop
    return result


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation requires aligned vectors with at least two values")
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    left_norm = sum((value - left_mean) ** 2 for value in left)
    right_norm = sum((value - right_mean) ** 2 for value in right)
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0 if list(left) == list(right) else 0.0
    return numerator / math.sqrt(left_norm * right_norm)


def spearman_rank_correlation(
    left: Sequence[float], right: Sequence[float],
) -> float:
    """Deterministic tied-rank Spearman correlation without scipy."""

    return _pearson(_rank(left), _rank(right))


def call_set_jaccard(left: Sequence[bool], right: Sequence[bool]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Jaccard masks must be aligned and non-empty")
    left_set = {index for index, value in enumerate(left) if value}
    right_set = {index for index, value in enumerate(right) if value}
    union = left_set | right_set
    return 1.0 if not union else len(left_set & right_set) / len(union)


def _action_regret(
    records: Sequence[SequentialRolloutRecord], mask: Sequence[bool],
) -> float:
    return mean(
        max(record.stop_correct, record.continue_correct)
        - (record.continue_correct if use else record.stop_correct)
        for record, use in zip(records, mask)
    )


def _metrics(
    records: Sequence[SequentialRolloutRecord], mask: Sequence[bool], *,
    lambda_cost: float, name: str,
) -> dict[str, Any]:
    values = policy_metrics(
        records, mask, lambda_cost=lambda_cost, policy_name=name,
    )
    values["top1_action_regret"] = _action_regret(records, mask)
    acquisition_rate = values["acquisition_rate"]
    if not isinstance(acquisition_rate, (int, float)):
        raise AssertionError("acquisition rate unexpectedly missing")
    values["answer_selection_probability"] = 1.0 - float(acquisition_rate)
    return values


def _mean_seed_metrics(values: Sequence[Mapping[str, Any]], *, name: str) -> dict[str, Any]:
    if not values:
        raise ValueError("seed aggregation requires metrics")
    keys = set(values[0])
    if any(set(item) != keys for item in values):
        raise ValueError("seed metric fields differ")
    result: dict[str, Any] = {"policy": name, "trained_seeds": len(values)}
    for key in sorted(keys - {"policy"}):
        entries = [item[key] for item in values]
        if key == "states":
            if len(set(entries)) != 1:
                raise ValueError("seed evaluations used different state counts")
            result[key] = entries[0]
        elif all(value is None for value in entries):
            result[key] = None
        elif any(value is None for value in entries):
            result[key] = None
        elif all(isinstance(value, (int, float)) for value in entries):
            result[key] = mean(float(value) for value in entries)
        else:
            raise ValueError(f"unsupported metric field for aggregation: {key}")
    return result


def _cluster_id(record: SequentialRolloutRecord, benchmark: str) -> str:
    # HRBench has multiple questions per image; the image is the independent
    # sampling unit frozen in the Phase-C allocation.
    return record.image_id if benchmark == "hrbench" else record.source_id


def seed_averaged_cluster_bootstrap_delta(
    records: Sequence[SequentialRolloutRecord],
    left_masks: Mapping[int, Sequence[bool]],
    right_masks: Mapping[int, Sequence[bool]],
    *, benchmark: str, lambda_cost: float, samples: int, seed: int,
) -> dict[str, Any]:
    """Bootstrap the mean independently-deployed seed policy difference."""

    if (
        benchmark not in BENCHMARKS
        or tuple(sorted(left_masks)) != SEEDS
        or tuple(sorted(right_masks)) != SEEDS
        or not records
        or samples < 10_000
        or not math.isfinite(lambda_cost)
        or lambda_cost < 0.0
    ):
        raise ValueError("invalid formal seed-averaged bootstrap inputs")
    if any(
        len(mask) != len(records)
        for masks in (left_masks, right_masks)
        for mask in masks.values()
    ):
        raise ValueError("formal bootstrap masks do not align with records")

    per_decision = []
    for index, record in enumerate(records):
        call_value = record.delta_success - lambda_cost * record.proposed_visual_cost
        per_decision.append(mean(
            (
                float(bool(left_masks[seed_value][index]))
                - float(bool(right_masks[seed_value][index]))
            ) * call_value
            for seed_value in SEEDS
        ))
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[_cluster_id(record, benchmark)].append(index)
    cluster_names = sorted(groups)
    cluster_sums = [sum(per_decision[index] for index in groups[name]) for name in cluster_names]
    cluster_counts = [len(groups[name]) for name in cluster_names]
    observed = sum(cluster_sums) / sum(cluster_counts)

    try:
        import numpy as np  # type: ignore[import-not-found]

        np_rng = np.random.default_rng(seed)
        sums = np.asarray(cluster_sums, dtype=np.float64)
        counts = np.asarray(cluster_counts, dtype=np.float64)
        values = np.empty(samples, dtype=np.float64)
        chunk = max(1, min(256, 4_000_000 // len(cluster_names)))
        for start in range(0, samples, chunk):
            size = min(chunk, samples - start)
            indices = np_rng.integers(
                0, len(cluster_names), size=(size, len(cluster_names))
            )
            values[start : start + size] = (
                sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
            )
        quantiles = np.asarray(
            np.quantile(values, (.025, .975), method="linear"), dtype=np.float64
        ).reshape(-1)
        ci_low, ci_high = (float(value) for value in quantiles.tolist())
    except ModuleNotFoundError:  # pragma: no cover - minimal install fallback
        py_rng = random.Random(seed)
        draws = []
        for _ in range(samples):
            sampled_indices = [
                py_rng.randrange(len(cluster_names)) for _ in cluster_names
            ]
            draws.append(
                sum(cluster_sums[index] for index in sampled_indices)
                / sum(cluster_counts[index] for index in sampled_indices)
            )
        draws.sort()
        ci_low = draws[int(0.025 * samples)]
        ci_high = draws[min(samples - 1, int(0.975 * samples))]
    return {
        "observed_delta": observed,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "seed_aggregation": "mean_of_independently_deployed_policies",
        "trained_seeds": len(SEEDS),
        "resampling_unit": "image_id" if benchmark == "hrbench" else "source_id",
        "clusters": len(cluster_names),
    }


def _calibration(
    records: Sequence[SequentialRolloutRecord], scores: Sequence[float],
) -> dict[str, Any]:
    order = sorted(range(len(records)), key=lambda index: (scores[index], _decision_id(records[index])))
    bins = []
    for bin_index in range(5):
        lower = round(bin_index * len(order) / 5)
        upper = round((bin_index + 1) * len(order) / 5)
        indices = order[lower:upper]
        if indices:
            bins.append({
                "bin": bin_index,
                "states": len(indices),
                "mean_score": mean(scores[index] for index in indices),
                "mean_gain": mean(records[index].delta_success for index in indices),
                "beneficial_rate": mean(
                    records[index].delta_success > 0 for index in indices
                ),
            })
    return {"quantile_bins": bins}


def index_formal_prediction_rows(
    records_by_benchmark: Mapping[str, Sequence[SequentialRolloutRecord]],
    prediction_payloads: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[int, dict[str, list[float]]]]]:
    """Strictly validate and align outcome-free prediction rows."""

    if tuple(sorted(records_by_benchmark)) != BENCHMARKS:
        raise ValueError("formal records must contain exactly three benchmarks")
    expected_ids = {
        benchmark: {_decision_id(record): record for record in records}
        for benchmark, records in records_by_benchmark.items()
    }
    if any(len(values) != len(records_by_benchmark[name]) for name, values in expected_ids.items()):
        raise ValueError("formal rollout decision IDs must be unique")

    rows_by_key: dict[tuple[str, int, str, str], dict[DecisionId, Mapping[str, Any]]] = {}
    seen_payload_seeds = set()
    for payload in prediction_payloads:
        if (
            payload.get("schema") != "factorized_phase_c_formal_predictions_v1"
            or payload.get("one_shot") is not True
            or payload.get("test_accessed") is not True
            or payload.get("formal_claim_eligible") is not True
            or int(payload.get("seed", -1)) not in SEEDS
        ):
            raise ValueError("invalid formal prediction payload")
        payload_seed = int(payload["seed"])
        if payload_seed in seen_payload_seeds:
            raise ValueError("duplicate formal prediction seed payload")
        seen_payload_seeds.add(payload_seed)
        for row in payload.get("predictions", ()):
            method = str(row.get("method"))
            mode = str(row.get("mode"))
            benchmark = str(row.get("benchmark"))
            if (
                method not in METHODS
                or benchmark not in BENCHMARKS
                or int(row.get("seed", -1)) != payload_seed
                or mode not in (FORMAL_MODES if method == PRIMARY_METHOD else ("original",))
            ):
                raise ValueError("formal prediction row has invalid method/domain/mode")
            common = {
                "method", "seed", "mode", "benchmark", "state_id", "replicate_id",
                "source_id", "continue_score", "action_logits", "measurement",
            }
            allowed = common | ({"factorized_probabilities"} if method == PRIMARY_METHOD else set())
            if set(row) != allowed:
                raise ValueError("formal prediction row fields differ from outcome-free contract")
            decision = str(row["state_id"]), str(row["replicate_id"])
            record = expected_ids[benchmark].get(decision)
            measurement = row["measurement"]
            score = float(row["continue_score"])
            if (
                record is None
                or str(row["source_id"]) != record.source_id
                or not math.isfinite(score)
                or measurement.get("proposed_crop_executions") != 0
                or measurement.get("already_acquired_crops") != 1
                or measurement.get("observed_images") != 2
            ):
                raise ValueError("formal prediction identity or leakage measurement failed")
            logits = row["action_logits"]
            if len(logits) != (3 if method == PRIMARY_METHOD else 2) or not all(
                math.isfinite(float(value)) for value in logits
            ):
                raise ValueError("formal prediction logits are invalid")
            if method == PRIMARY_METHOD:
                factors = row["factorized_probabilities"]
                factor_keys = {
                    "error_probability", "rescue_probability_given_error",
                    "harm_probability_given_correct", "expected_gain",
                }
                if (
                    set(factors) != factor_keys
                    or not all(math.isfinite(float(factors[key])) for key in factor_keys)
                    or not all(
                        0.0 <= float(factors[key]) <= 1.0
                        for key in factor_keys - {"expected_gain"}
                    )
                    or not math.isclose(
                        float(factors["expected_gain"]), score, abs_tol=1e-7
                    )
                    or not -1.0 <= float(factors["expected_gain"]) <= 1.0
                    or not math.isclose(
                        float(factors["expected_gain"]),
                        float(factors["error_probability"])
                        * float(factors["rescue_probability_given_error"])
                        - (1.0 - float(factors["error_probability"]))
                        * float(factors["harm_probability_given_correct"]),
                        abs_tol=1e-7,
                    )
                ):
                    raise ValueError("formal factorized probabilities are invalid")
            key = method, payload_seed, mode, benchmark
            indexed = rows_by_key.setdefault(key, {})
            if decision in indexed:
                raise ValueError("duplicate formal prediction decision")
            indexed[decision] = row
    if seen_payload_seeds != set(SEEDS):
        raise ValueError("formal predictions must contain all frozen seeds")

    result: dict[str, dict[str, dict[int, dict[str, list[float]]]]] = {
        benchmark: {method: {} for method in METHODS} for benchmark in BENCHMARKS
    }
    for benchmark in BENCHMARKS:
        ordered_records = list(records_by_benchmark[benchmark])
        ordered_ids = [_decision_id(record) for record in ordered_records]
        for method in METHODS:
            modes = FORMAL_MODES if method == PRIMARY_METHOD else ("original",)
            for seed_value in SEEDS:
                result[benchmark][method][seed_value] = {}
                for mode in modes:
                    indexed = rows_by_key.get((method, seed_value, mode, benchmark), {})
                    if set(indexed) != set(ordered_ids):
                        raise ValueError(
                            f"formal prediction coverage mismatch: {benchmark}/{method}/"
                            f"{seed_value}/{mode}"
                        )
                    result[benchmark][method][seed_value][mode] = [
                        float(indexed[decision]["continue_score"])
                        for decision in ordered_ids
                    ]
    return result


def evaluate_phase_c_formal(
    plan: Mapping[str, Any],
    records_by_benchmark: Mapping[str, Sequence[SequentialRolloutRecord]],
    prediction_payloads: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate the pre-registered formal protocol without model selection."""

    policy = plan["policy"]
    baselines = plan["baselines"]
    ablations = plan["ablations"]
    if (
        tuple(float(value) for value in policy["rates"])
        != (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
        or tuple(float(value) for value in policy["lambdas"])
        != (0.0, 0.025, 0.05, 0.1, 0.2)
        or tuple(baselines["uncertainty"]) != UNCERTAINTY_BASELINES
    ):
        raise ValueError("formal evaluation policy or baseline contract drifted")
    ordered_records = {
        benchmark: sorted(records_by_benchmark[benchmark], key=_decision_id)
        for benchmark in BENCHMARKS
    }
    for benchmark in BENCHMARKS:
        if len(ordered_records[benchmark]) != int(plan["benchmarks"][benchmark]["states"]):
            raise ValueError(f"formal rollout count mismatch for {benchmark}")
    scores = index_formal_prediction_rows(ordered_records, prediction_payloads)
    rates = tuple(float(value) for value in policy["rates"])
    lambdas = tuple(float(value) for value in policy["lambdas"])
    primary_rate = float(policy["primary_call_rate"])
    primary_lambda = float(policy["primary_lambda"])
    samples = int(policy["bootstrap_samples"])
    bootstrap_seed = int(policy["bootstrap_seed"])

    benchmark_reports: dict[str, Any] = {}
    semantic_domain_values: dict[str, dict[str, Any]] = {
        mode: {} for mode in FORMAL_MODES[1:]
    }
    for benchmark in BENCHMARKS:
        records = ordered_records[benchmark]
        baseline_scores = {
            "entropy": [record.stop_entropy for record in records],
            "confidence": [-record.stop_max_probability for record in records],
            "margin": [-record.stop_top1_top2_margin for record in records],
            "random_gate": _random_scores(
                records, seed=int(baselines["random_seed"]), benchmark=benchmark,
            ),
            "oracle_gain_rank": [record.delta_success for record in records],
        }
        masks: dict[float, dict[str, Any]] = {}
        frontiers = []
        for rate in rates:
            at_rate: dict[str, Any] = {
                "answer_only": [False] * len(records),
                **{
                    name: _top_rate_mask(records, values, rate)
                    for name, values in baseline_scores.items()
                },
            }
            for method in METHODS:
                at_rate[method] = {
                    seed_value: _top_rate_mask(
                        records, scores[benchmark][method][seed_value]["original"], rate,
                    )
                    for seed_value in SEEDS
                }
            masks[rate] = at_rate
            by_lambda = {}
            for lambda_cost in lambdas:
                metrics = {
                    name: _metrics(
                        records, mask, lambda_cost=lambda_cost, name=name,
                    )
                    for name, mask in at_rate.items()
                    if name not in METHODS
                }
                for method in METHODS:
                    metrics[method] = _mean_seed_metrics(
                        [
                            _metrics(
                                records, at_rate[method][seed_value],
                                lambda_cost=lambda_cost,
                                name=f"{method}/seed-{seed_value}",
                            )
                            for seed_value in SEEDS
                        ],
                        name=method,
                    )
                by_lambda[str(lambda_cost)] = metrics
            frontiers.append({
                "target_call_rate": rate,
                "calls_per_seed": round(rate * len(records)),
                "realized_call_rate": round(rate * len(records)) / len(records),
                "policies_by_lambda": by_lambda,
            })

        primary_masks = masks[primary_rate]
        primary_metrics = {
            name: _metrics(
                records, mask, lambda_cost=primary_lambda, name=name,
            )
            for name, mask in primary_masks.items()
            if name not in METHODS
        }
        for method in METHODS:
            primary_metrics[method] = _mean_seed_metrics(
                [
                    _metrics(
                        records, primary_masks[method][seed_value],
                        lambda_cost=primary_lambda,
                        name=f"{method}/seed-{seed_value}",
                    )
                    for seed_value in SEEDS
                ],
                name=method,
            )
        strongest = max(
            UNCERTAINTY_BASELINES,
            key=lambda name: (
                float(primary_metrics[name]["accuracy"]),
                -UNCERTAINTY_BASELINES.index(name),
            ),
        )
        comparisons = {}
        for comparator in (PRIMARY_CONTROL, DIRECT_CONTROL, strongest):
            right = (
                primary_masks[comparator]
                if comparator in METHODS
                else {seed_value: primary_masks[comparator] for seed_value in SEEDS}
            )
            label = (
                "strongest_uncertainty" if comparator == strongest else comparator
            )
            comparisons[f"{PRIMARY_METHOD}_minus_{label}"] = {
                "comparator": comparator,
                "accuracy": seed_averaged_cluster_bootstrap_delta(
                    records, primary_masks[PRIMARY_METHOD], right,
                    benchmark=benchmark, lambda_cost=0.0, samples=samples,
                    seed=bootstrap_seed,
                ),
                f"net_utility_lambda_{primary_lambda}": (
                    seed_averaged_cluster_bootstrap_delta(
                        records, primary_masks[PRIMARY_METHOD], right,
                        benchmark=benchmark, lambda_cost=primary_lambda,
                        samples=samples, seed=bootstrap_seed,
                    )
                ),
            }

        calibration = {
            method: {
                str(seed_value): _calibration(
                    records, scores[benchmark][method][seed_value]["original"],
                )
                for seed_value in SEEDS
            }
            for method in METHODS
        }
        semantic = {}
        original_masks = primary_masks[PRIMARY_METHOD]
        for mode in FORMAL_MODES[1:]:
            per_seed = {}
            for seed_value in SEEDS:
                original_scores = scores[benchmark][PRIMARY_METHOD][seed_value]["original"]
                ablated_scores = scores[benchmark][PRIMARY_METHOD][seed_value][mode]
                ablated_mask = _top_rate_mask(records, ablated_scores, primary_rate)
                original_accuracy = float(_metrics(
                    records, original_masks[seed_value], lambda_cost=primary_lambda,
                    name="original",
                )["accuracy"])
                ablated_accuracy = float(_metrics(
                    records, ablated_mask, lambda_cost=primary_lambda,
                    name=mode,
                )["accuracy"])
                per_seed[str(seed_value)] = {
                    "score_spearman": spearman_rank_correlation(
                        original_scores, ablated_scores,
                    ),
                    "primary_call_set_jaccard": call_set_jaccard(
                        original_masks[seed_value], ablated_mask,
                    ),
                    "accuracy_delta_original_minus_ablation": (
                        original_accuracy - ablated_accuracy
                    ),
                }
            summary: dict[str, Any] = {
                "per_seed": per_seed,
                "mean_score_spearman": mean(
                    item["score_spearman"] for item in per_seed.values()
                ),
                "mean_primary_call_set_jaccard": mean(
                    item["primary_call_set_jaccard"] for item in per_seed.values()
                ),
                "mean_accuracy_delta_original_minus_ablation": mean(
                    item["accuracy_delta_original_minus_ablation"]
                    for item in per_seed.values()
                ),
            }
            summary["ranking_changed"] = (
                summary["mean_score_spearman"]
                < float(ablations["semantic_score_correlation_max"])
                and summary["mean_primary_call_set_jaccard"]
                < float(ablations["semantic_call_set_jaccard_max"])
            )
            semantic[mode] = summary
            semantic_domain_values[mode][benchmark] = summary

        benchmark_reports[benchmark] = {
            "states": len(records),
            "source_clusters": len({_cluster_id(record, benchmark) for record in records}),
            "source_cluster_field": "image_id" if benchmark == "hrbench" else "source_id",
            "diagnostic": {
                "stop_accuracy": mean(record.stop_correct for record in records),
                "continue_accuracy": mean(record.continue_correct for record in records),
                "beneficial": sum(record.delta_success > 0 for record in records),
                "harmful": sum(record.delta_success < 0 for record in records),
                "neutral": sum(record.delta_success == 0 for record in records),
                "oracle_mean_positive_gain": mean(
                    max(record.delta_success, 0.0) for record in records
                ),
            },
            "primary_call_rate": primary_rate,
            "primary_lambda": primary_lambda,
            "primary_strongest_uncertainty_baseline": strongest,
            "primary_metrics": primary_metrics,
            "primary_comparisons": comparisons,
            "semantic_ablations": semantic,
            "calibration": calibration,
            "frontier": frontiers,
        }

    semantic_gates = {}
    for mode, domains in semantic_domain_values.items():
        changed = [name for name, value in domains.items() if value["ranking_changed"]]
        mean_accuracy_delta = mean(
            value["mean_accuracy_delta_original_minus_ablation"]
            for value in domains.values()
        )
        checks = {
            "positive_mean_accuracy_delta": mean_accuracy_delta > 0.0,
            "ranking_changed_in_required_domains": len(changed)
            >= int(ablations["required_domains_with_changed_ranking"]),
        }
        semantic_gates[mode] = {
            "mean_accuracy_delta_original_minus_ablation": mean_accuracy_delta,
            "domains_with_changed_ranking": changed,
            "checks": checks,
            "passed": all(checks.values()),
        }

    outcome_deltas = {
        benchmark: benchmark_reports[benchmark]["primary_comparisons"]
        [f"{PRIMARY_METHOD}_minus_{PRIMARY_CONTROL}"]["accuracy"]
        for benchmark in BENCHMARKS
    }
    positive_domains = [
        name for name, value in outcome_deltas.items()
        if value["observed_delta"] > 0.0
    ]
    significant_domains = [
        name for name, value in outcome_deltas.items() if value["ci_low"] > 0.0
    ]
    safe_domains = [
        name for name in positive_domains
        if (
            float(benchmark_reports[name]["primary_metrics"][PRIMARY_METHOD]["accuracy"])
            - float(benchmark_reports[name]["primary_metrics"][
                benchmark_reports[name]["primary_strongest_uncertainty_baseline"]
            ]["accuracy"])
        ) >= -float(plan["go_rule"]["successful_domain_max_regression_vs_strongest_uncertainty"])
    ]
    go_checks = {
        "positive_mean_delta_vs_outcome_in_required_domains": len(positive_domains)
        >= int(plan["go_rule"]["positive_mean_delta_vs_outcome_domains"]),
        "positive_source_bootstrap_ci_vs_outcome_in_required_domain": len(significant_domains)
        >= int(plan["go_rule"][
            "required_domain_source_bootstrap_ci_low_vs_outcome_positive"
        ]),
        "no_excess_regression_vs_strongest_uncertainty_on_successful_domains": (
            set(safe_domains) == set(positive_domains)
        ),
        "all_semantic_ablations_passed": all(
            value["passed"] for value in semantic_gates.values()
        ),
    }
    decision = "GO" if all(go_checks.values()) else "NO_GO"
    return {
        "schema": "factorized_phase_c_formal_evaluation_v1",
        "one_shot": True,
        "test_accessed": True,
        "formal_claim_eligible": True,
        "primary_method": PRIMARY_METHOD,
        "seed_aggregation": "mean_of_independently_deployed_policies",
        "decision": decision,
        "go_checks": go_checks,
        "positive_vs_outcome_domains": positive_domains,
        "significant_vs_outcome_domains": significant_domains,
        "safe_vs_uncertainty_domains": safe_domains,
        "mean_accuracy_delta_vs_outcome": mean(
            value["observed_delta"] for value in outcome_deltas.values()
        ),
        "semantic_gates": semantic_gates,
        "benchmarks": benchmark_reports,
    }
