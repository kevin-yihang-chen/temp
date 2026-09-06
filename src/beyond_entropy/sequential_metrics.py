"""Diagnostics, policy metrics, calibration, and paired source bootstrap."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import mean
from typing import Callable, Mapping, Sequence

from .sequential_schema import SequentialRolloutRecord


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2:
        return None
    lm, rm = mean(left), mean(right)
    numerator = sum((a - lm) * (b - rm) for a, b in zip(left, right))
    denominator = (
        sum((a - lm) ** 2 for a in left) * sum((b - rm) ** 2 for b in right)
    ) ** 0.5
    return None if denominator == 0 else numerator / denominator


def sequential_diagnostic(records: Sequence[SequentialRolloutRecord]) -> dict[str, float | int | None]:
    if not records:
        raise ValueError("sequential diagnostic requires records")
    gains = [item.delta_success for item in records]
    entropy_deltas = [item.delta_entropy for item in records]
    predicted_useful = [delta > 0 for delta in entropy_deltas]
    beneficial = [gain > 0 for gain in gains]
    useful_count = sum(predicted_useful)
    return {
        "states": len(records),
        "beneficial_count": sum(beneficial),
        "harmful_count": sum(gain < 0 for gain in gains),
        "neutral_count": sum(gain == 0 for gain in gains),
        "beneficial_rate": mean(beneficial),
        "harmful_rate": mean(gain < 0 for gain in gains),
        "neutral_rate": mean(gain == 0 for gain in gains),
        "mean_gain": mean(gains),
        "oracle_incremental_gain": mean(max(0.0, gain) for gain in gains),
        "delta_entropy_gain_pearson": _pearson(entropy_deltas, gains),
        "entropy_sign_mismatch_rate": mean(
            (entropy > 0) != (gain > 0)
            for entropy, gain in zip(entropy_deltas, gains)
        ),
        "entropy_useful_precision": (
            sum(p and y for p, y in zip(predicted_useful, beneficial)) / useful_count
            if useful_count
            else None
        ),
        "entropy_useful_recall": (
            sum(p and y for p, y in zip(predicted_useful, beneficial)) / sum(beneficial)
            if any(beneficial)
            else None
        ),
    }


def policy_metrics(
    records: Sequence[SequentialRolloutRecord],
    continue_mask: Sequence[bool],
    *,
    lambda_cost: float,
    policy_name: str,
) -> dict[str, float | int | str | None]:
    if not records or len(records) != len(continue_mask):
        raise ValueError("policy mask must align with non-empty records")
    if not math.isfinite(lambda_cost) or lambda_cost < 0:
        raise ValueError("lambda_cost must be finite and non-negative")
    outcomes = [
        item.continue_correct if use else item.stop_correct
        for item, use in zip(records, continue_mask)
    ]
    total_costs = [
        item.continue_total_visual_cost if use else item.stop_total_visual_cost
        for item, use in zip(records, continue_mask)
    ]
    incremental_costs = [
        item.proposed_visual_cost if use else 0.0
        for item, use in zip(records, continue_mask)
    ]
    gains = [item.delta_success if use else 0.0 for item, use in zip(records, continue_mask)]
    beneficial = [item.delta_success > 0 for item in records]
    calls = sum(continue_mask)
    true_calls = sum(use and useful for use, useful in zip(continue_mask, beneficial))
    oracle_utility = mean(
        max(0.0, item.incremental_utility(lambda_cost)) for item in records
    )
    incremental_utility = mean(
        gain - lambda_cost * cost for gain, cost in zip(gains, incremental_costs)
    )
    return {
        "policy": policy_name,
        "states": len(records),
        "accuracy": mean(outcomes),
        "stop_accuracy": mean(item.stop_correct for item in records),
        "accuracy_gain": mean(outcomes) - mean(item.stop_correct for item in records),
        "acquisition_rate": calls / len(records),
        "avg_incremental_visual_cost": mean(incremental_costs),
        "avg_total_visual_cost": mean(total_costs),
        "incremental_net_utility": incremental_utility,
        "total_net_utility": mean(outcomes) - lambda_cost * mean(total_costs),
        "oracle_incremental_utility": oracle_utility,
        "oracle_utility_gap": oracle_utility - incremental_utility,
        "beneficial_acquisition_precision": true_calls / calls if calls else None,
        "beneficial_acquisition_recall": (
            true_calls / sum(beneficial) if any(beneficial) else None
        ),
        "harmful_acquisition_rate": (
            sum(use and item.delta_success < 0 for item, use in zip(records, continue_mask))
            / calls
            if calls
            else None
        ),
        "unnecessary_acquisition_rate": (
            sum(use and item.delta_success <= 0 for item, use in zip(records, continue_mask))
            / calls
            if calls
            else None
        ),
    }


def binary_auroc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    if len(scores) != len(labels) or not scores:
        raise ValueError("scores and labels must align")
    positives = [score for score, label in zip(scores, labels) if label == 1]
    negatives = [score for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        return None
    wins = sum(p > n for p in positives for n in negatives)
    ties = sum(p == n for p in positives for n in negatives)
    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


def expected_calibration_error(
    probabilities: Sequence[float], labels: Sequence[int], *, bins: int = 10
) -> float:
    if len(probabilities) != len(labels) or not probabilities or bins <= 0:
        raise ValueError("invalid calibration inputs")
    total = len(labels)
    result = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = [
            i
            for i, value in enumerate(probabilities)
            if lower <= value < upper or (index == bins - 1 and value == 1.0)
        ]
        if members:
            result += len(members) / total * abs(
                mean(probabilities[i] for i in members)
                - mean(labels[i] for i in members)
            )
    return result


def risk_metrics(probabilities: Sequence[float], stop_correct: Sequence[float]) -> dict[str, object]:
    if len(probabilities) != len(stop_correct) or not probabilities:
        raise ValueError("risk predictions must align with labels")
    labels = [int(value < 1.0) for value in stop_correct]
    if any(not 0 <= value <= 1 for value in probabilities):
        raise ValueError("risk probabilities must lie in [0, 1]")
    order = sorted(range(len(labels)), key=lambda i: (probabilities[i], i))
    points = []
    for coverage in (0.1, 0.2, 0.4, 0.6, 0.8, 1.0):
        count = max(1, round(coverage * len(order)))
        accepted = order[:count]
        points.append(
            {
                "coverage": count / len(order),
                "risk": mean(labels[i] for i in accepted),
            }
        )
    aurc = 0.0
    previous_coverage = 0.0
    for point in points:
        aurc += (point["coverage"] - previous_coverage) * point["risk"]
        previous_coverage = point["coverage"]
    return {
        "auroc": binary_auroc(probabilities, labels),
        "brier": mean((p - y) ** 2 for p, y in zip(probabilities, labels)),
        "ece": expected_calibration_error(probabilities, labels),
        "aurc": aurc,
        "risk_coverage": points,
    }


def paired_source_bootstrap_delta(
    records: Sequence[SequentialRolloutRecord],
    left_mask: Sequence[bool],
    right_mask: Sequence[bool],
    *,
    statistic: Callable[[Sequence[SequentialRolloutRecord], Sequence[bool]], float],
    samples: int = 10_000,
    seed: int = 20260906,
) -> dict[str, float | int]:
    """Paired bootstrap over sources, preserving every decision within a source."""

    if samples < 10_000:
        raise ValueError("formal paired bootstrap requires at least 10,000 samples")
    if not records or len(records) != len(left_mask) or len(records) != len(right_mask):
        raise ValueError("bootstrap inputs must align")
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record.source_id, []).append(index)
    sources = sorted(groups)
    observed = statistic(records, left_mask) - statistic(records, right_mask)
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        sampled = [sources[rng.randrange(len(sources))] for _ in sources]
        indices = [index for source in sampled for index in groups[source]]
        subset = [records[index] for index in indices]
        left = [left_mask[index] for index in indices]
        right = [right_mask[index] for index in indices]
        deltas.append(statistic(subset, left) - statistic(subset, right))
    ordered = sorted(deltas)
    return {
        "observed_delta": observed,
        "ci_low": ordered[int(0.025 * samples)],
        "ci_high": ordered[min(samples - 1, int(0.975 * samples))],
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }
