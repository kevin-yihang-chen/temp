from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Any, Callable, Mapping, Sequence

from .predictability_audit import BinaryToolOutcome


DecisionId = tuple[str, str]


@dataclass(frozen=True)
class Prediction:
    state_id: str
    replicate_id: str
    score: float
    positive_net_probability: float
    rescue_probability: float
    harm_probability: float

    def __post_init__(self) -> None:
        if not self.state_id or not self.replicate_id:
            raise ValueError("prediction identity must be non-empty")
        if not math.isfinite(self.score):
            raise ValueError("prediction score must be finite")
        for name in (
            "positive_net_probability",
            "rescue_probability",
            "harm_probability",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")

    @property
    def decision_id(self) -> DecisionId:
        return self.state_id, self.replicate_id


def align_predictions(
    outcomes: Sequence[BinaryToolOutcome], predictions: Sequence[Prediction]
) -> list[Prediction]:
    expected = {(item.state_id, item.replicate_id) for item in outcomes}
    indexed = {item.decision_id: item for item in predictions}
    if len(indexed) != len(predictions):
        raise ValueError("prediction decision IDs must be unique")
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        extra = sorted(set(indexed) - expected)
        raise ValueError(
            f"prediction coverage mismatch; missing={missing[:3]}, extra={extra[:3]}"
        )
    return [indexed[(item.state_id, item.replicate_id)] for item in outcomes]


def _source_mean(
    outcomes: Sequence[BinaryToolOutcome], values: Sequence[float]
) -> float:
    by_source: dict[str, list[float]] = defaultdict(list)
    for outcome, value in zip(outcomes, values):
        by_source[outcome.source_id].append(float(value))
    if not by_source:
        raise ValueError("metric requires at least one source")
    return mean(mean(source_values) for source_values in by_source.values())


def policy_metrics(
    outcomes: Sequence[BinaryToolOutcome],
    calls: Sequence[bool],
    *,
    lambda_cost: float,
) -> dict[str, float | int | None]:
    if not outcomes or len(outcomes) != len(calls):
        raise ValueError("policy metrics require aligned non-empty outcomes and calls")
    if lambda_cost < 0.0:
        raise ValueError("lambda_cost must be non-negative")
    accuracies = [
        item.y_tool if call else item.y0 for item, call in zip(outcomes, calls)
    ]
    costs = [item.tool_cost if call else 0.0 for item, call in zip(outcomes, calls)]
    incremental = [
        item.incremental_utility(lambda_cost) if call else 0.0
        for item, call in zip(outcomes, calls)
    ]
    selected = [item for item, call in zip(outcomes, calls) if call]
    call_count = len(selected)
    return {
        "decisions": len(outcomes),
        "calls": call_count,
        "accuracy": _source_mean(outcomes, accuracies),
        "cost": _source_mean(outcomes, costs),
        "call_rate": _source_mean(outcomes, [float(value) for value in calls]),
        "incremental_utility": _source_mean(outcomes, incremental),
        "rescue_precision": (
            None if not selected else sum(item.rescue for item in selected) / call_count
        ),
        "harm_rate_per_call": (
            None if not selected else sum(item.harm for item in selected) / call_count
        ),
        "marginal_gain_per_call": (
            None if not selected else sum(item.gain for item in selected) / call_count
        ),
    }


def select_validation_threshold(
    outcomes: Sequence[BinaryToolOutcome],
    predictions: Sequence[Prediction],
    *,
    lambda_cost: float,
) -> dict[str, float | int]:
    aligned = align_predictions(outcomes, predictions)
    thresholds = [math.inf, *sorted({item.score for item in aligned}, reverse=True)]
    candidates: list[tuple[float, int, float, dict[str, float | int | None]]] = []
    for threshold in thresholds:
        calls = [item.score >= threshold for item in aligned]
        metrics = policy_metrics(outcomes, calls, lambda_cost=lambda_cost)
        utility_value = metrics["incremental_utility"]
        call_count = metrics["calls"]
        if not isinstance(utility_value, (float, int)) or not isinstance(
            call_count, int
        ):
            raise AssertionError("required policy metrics unexpectedly missing")
        candidates.append(
            (
                float(utility_value),
                -call_count,
                -threshold,
                metrics,
            )
        )
    utility, negative_calls, negative_threshold, metrics = max(
        candidates, key=lambda x: x[:3]
    )
    threshold = -negative_threshold
    call_rate = metrics["call_rate"]
    if not isinstance(call_rate, (float, int)):
        raise AssertionError("validation call rate unexpectedly missing")
    return {
        "threshold": threshold,
        "validation_utility": utility,
        "validation_calls": -negative_calls,
        "validation_call_rate": float(call_rate),
    }


def calls_at_threshold(
    predictions: Sequence[Prediction], threshold: float
) -> list[bool]:
    return [item.score >= threshold for item in predictions]


def calls_at_rate(predictions: Sequence[Prediction], call_rate: float) -> list[bool]:
    if not 0.0 <= call_rate <= 1.0:
        raise ValueError("call_rate must be in [0, 1]")
    count = round(call_rate * len(predictions))
    order = sorted(
        range(len(predictions)),
        key=lambda index: (
            -predictions[index].score,
            predictions[index].state_id,
            predictions[index].replicate_id,
        ),
    )
    selected = set(order[:count])
    return [index in selected for index in range(len(predictions))]


def _binary_metrics(
    labels: Sequence[bool], probabilities: Sequence[float]
) -> dict[str, float | None]:
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("binary metrics require aligned non-empty inputs")
    if len(set(labels)) < 2:
        return {"auroc": None, "auprc": None}
    from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore[import-untyped]

    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
    }


def _expected_calibration_error(
    labels: Sequence[bool], probabilities: Sequence[float], *, bins: int = 10
) -> float:
    if bins <= 0:
        raise ValueError("calibration bins must be positive")
    total = len(labels)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            item
            for item, probability in enumerate(probabilities)
            if lower <= probability < upper
            or (index == bins - 1 and probability == 1.0)
        ]
        if not members:
            continue
        confidence = mean(probabilities[item] for item in members)
        accuracy = mean(float(labels[item]) for item in members)
        error += len(members) / total * abs(accuracy - confidence)
    return error


def prediction_metrics(
    outcomes: Sequence[BinaryToolOutcome],
    predictions: Sequence[Prediction],
    *,
    lambda_cost: float,
    calibration_bins: int = 10,
) -> dict[str, float | None]:
    aligned = align_predictions(outcomes, predictions)
    positive_net = [item.incremental_utility(lambda_cost) > 0.0 for item in outcomes]
    rescue = [item.rescue for item in outcomes]
    harm = [item.harm for item in outcomes]
    probabilities = [item.positive_net_probability for item in aligned]
    primary = _binary_metrics(positive_net, probabilities)
    rescue_metrics = _binary_metrics(
        rescue, [item.rescue_probability for item in aligned]
    )
    harm_metrics = _binary_metrics(harm, [item.harm_probability for item in aligned])
    primary.update(
        {
            "brier": mean(
                (probability - float(label)) ** 2
                for probability, label in zip(probabilities, positive_net)
            ),
            "calibration_error": _expected_calibration_error(
                positive_net, probabilities, bins=calibration_bins
            ),
            "rescue_auprc": rescue_metrics["auprc"],
            "harm_auprc": harm_metrics["auprc"],
        }
    )
    return primary


def policy_curve(
    outcomes: Sequence[BinaryToolOutcome],
    predictions: Sequence[Prediction],
    *,
    lambda_cost: float,
    call_rates: Sequence[float],
) -> list[dict[str, float | int | None]]:
    aligned = align_predictions(outcomes, predictions)
    result: list[dict[str, float | int | None]] = []
    for call_rate in call_rates:
        metrics = policy_metrics(
            outcomes,
            calls_at_rate(aligned, call_rate),
            lambda_cost=lambda_cost,
        )
        result.append({"requested_call_rate": call_rate, **metrics})
    return result


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_source_bootstrap_utility(
    outcomes: Sequence[BinaryToolOutcome],
    candidate_calls: Sequence[bool],
    baseline_calls: Sequence[bool],
    *,
    lambda_cost: float,
    resamples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    if (
        not outcomes
        or len(outcomes) != len(candidate_calls)
        or len(outcomes) != len(baseline_calls)
    ):
        raise ValueError("paired bootstrap requires aligned non-empty inputs")
    if resamples <= 0 or not 0.0 < confidence_level < 1.0:
        raise ValueError("invalid bootstrap configuration")
    by_source: dict[str, list[float]] = defaultdict(list)
    for outcome, candidate, baseline in zip(outcomes, candidate_calls, baseline_calls):
        net = outcome.incremental_utility(lambda_cost)
        by_source[outcome.source_id].append(net * (float(candidate) - float(baseline)))
    source_differences = {name: mean(values) for name, values in by_source.items()}
    sources = sorted(source_differences)
    rng = random.Random(seed)
    samples = [
        mean(source_differences[rng.choice(sources)] for _ in sources)
        for _ in range(resamples)
    ]
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "point": mean(source_differences.values()),
        "lower": _percentile(samples, alpha),
        "upper": _percentile(samples, 1.0 - alpha),
        "confidence_level": confidence_level,
        "resamples": resamples,
        "seed": seed,
        "resampling_unit": "source_id",
        "sources": len(sources),
    }
