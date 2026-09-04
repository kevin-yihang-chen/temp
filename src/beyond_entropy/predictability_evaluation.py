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
    positive_gain_probability: float
    rescue_probability: float
    harm_probability: float

    def __post_init__(self) -> None:
        if not self.state_id or not self.replicate_id:
            raise ValueError("prediction identity must be non-empty")
        if not math.isfinite(self.score):
            raise ValueError("prediction score must be finite")
        for name in (
            "positive_gain_probability",
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


def _decision_id(outcome: BinaryToolOutcome) -> DecisionId:
    return outcome.state_id, outcome.replicate_id


def align_policy_outcomes(
    reference: Sequence[BinaryToolOutcome],
    outcomes: Sequence[BinaryToolOutcome],
    calls: Sequence[bool],
) -> tuple[list[BinaryToolOutcome], list[bool]]:
    """Align an independent policy ledger to a reference decision order.

    Different policies may execute different tool actions and incur different
    costs. They must nevertheless refer to the same paired decisions and the
    same answer-now outcome. This function makes that contract explicit before
    any policy difference is computed.
    """

    if not reference or len(outcomes) != len(calls):
        raise ValueError("policy alignment requires non-empty aligned inputs")
    reference_ids = [_decision_id(item) for item in reference]
    if len(set(reference_ids)) != len(reference_ids):
        raise ValueError("reference policy decision IDs must be unique")
    indexed: dict[DecisionId, tuple[BinaryToolOutcome, bool]] = {}
    for outcome, call in zip(outcomes, calls):
        decision_id = _decision_id(outcome)
        if decision_id in indexed:
            raise ValueError("comparison policy decision IDs must be unique")
        indexed[decision_id] = outcome, bool(call)
    if set(indexed) != set(reference_ids):
        missing = sorted(set(reference_ids) - set(indexed))
        extra = sorted(set(indexed) - set(reference_ids))
        raise ValueError(
            f"policy coverage mismatch; missing={missing[:3]}, extra={extra[:3]}"
        )

    aligned_outcomes: list[BinaryToolOutcome] = []
    aligned_calls: list[bool] = []
    for expected in reference:
        actual, call = indexed[_decision_id(expected)]
        if (
            actual.image_id != expected.image_id
            or actual.source_id != expected.source_id
            or not math.isclose(actual.y0, expected.y0, abs_tol=1e-12)
        ):
            raise ValueError(
                f"paired policy identity or Y0 mismatch for {_decision_id(expected)!r}"
            )
        aligned_outcomes.append(actual)
        aligned_calls.append(call)
    return aligned_outcomes, aligned_calls


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


def _finite_no_call_threshold(scores: Sequence[float]) -> float:
    maximum = max(scores)
    threshold = math.nextafter(maximum, math.inf)
    if not math.isfinite(threshold):
        raise ValueError("scores are too large to construct a finite no-call threshold")
    return threshold


def select_score_threshold(
    outcomes: Sequence[BinaryToolOutcome],
    scores: Sequence[float],
    *,
    lambda_cost: float,
) -> dict[str, float | int]:
    """Tune a high-score call threshold on validation outcomes only."""

    if not outcomes or len(outcomes) != len(scores):
        raise ValueError("threshold selection requires aligned non-empty inputs")
    normalized_scores = [float(value) for value in scores]
    if not all(math.isfinite(value) for value in normalized_scores):
        raise ValueError("threshold scores must be finite")
    thresholds = [
        _finite_no_call_threshold(normalized_scores),
        *sorted(set(normalized_scores), reverse=True),
    ]
    candidates: list[tuple[float, int, float, dict[str, float | int | None]]] = []
    for threshold in thresholds:
        calls = [score >= threshold for score in normalized_scores]
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
        candidates, key=lambda item: item[:3]
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


def select_validation_threshold(
    outcomes: Sequence[BinaryToolOutcome],
    predictions: Sequence[Prediction],
    *,
    lambda_cost: float,
) -> dict[str, float | int]:
    aligned = align_predictions(outcomes, predictions)
    return select_score_threshold(
        outcomes,
        [item.score for item in aligned],
        lambda_cost=lambda_cost,
    )


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
    calibration_bins: int = 10,
) -> dict[str, float | None]:
    aligned = align_predictions(outcomes, predictions)
    positive_gain = [item.gain > 0.0 for item in outcomes]
    rescue = [item.rescue for item in outcomes]
    harm = [item.harm for item in outcomes]
    probabilities = [item.positive_gain_probability for item in aligned]
    primary = _binary_metrics(positive_gain, probabilities)
    rescue_metrics = _binary_metrics(
        rescue, [item.rescue_probability for item in aligned]
    )
    harm_metrics = _binary_metrics(harm, [item.harm_probability for item in aligned])
    primary.update(
        {
            "brier": mean(
                (probability - float(label)) ** 2
                for probability, label in zip(probabilities, positive_gain)
            ),
            "calibration_error": _expected_calibration_error(
                positive_gain, probabilities, bins=calibration_bins
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
    return paired_source_bootstrap_policy_difference(
        outcomes,
        candidate_calls,
        outcomes,
        baseline_calls,
        lambda_cost=lambda_cost,
        resamples=resamples,
        confidence_level=confidence_level,
        seed=seed,
    )


def paired_source_bootstrap_policy_difference(
    candidate_outcomes: Sequence[BinaryToolOutcome],
    candidate_calls: Sequence[bool],
    baseline_outcomes: Sequence[BinaryToolOutcome],
    baseline_calls: Sequence[bool],
    *,
    lambda_cost: float,
    resamples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap candidate-minus-baseline utility from independent ledgers."""

    if (
        not candidate_outcomes
        or len(candidate_outcomes) != len(candidate_calls)
        or len(baseline_outcomes) != len(baseline_calls)
    ):
        raise ValueError("paired bootstrap requires aligned non-empty inputs")
    if resamples <= 0 or not 0.0 < confidence_level < 1.0:
        raise ValueError("invalid bootstrap configuration")
    if lambda_cost < 0.0:
        raise ValueError("lambda_cost must be non-negative")
    aligned_baseline, aligned_baseline_calls = align_policy_outcomes(
        candidate_outcomes, baseline_outcomes, baseline_calls
    )
    by_source: dict[str, list[float]] = defaultdict(list)
    candidate_values: list[float] = []
    baseline_values: list[float] = []
    for candidate_outcome, candidate, baseline_outcome, baseline in zip(
        candidate_outcomes,
        candidate_calls,
        aligned_baseline,
        aligned_baseline_calls,
    ):
        candidate_value = (
            candidate_outcome.incremental_utility(lambda_cost) if candidate else 0.0
        )
        baseline_value = (
            baseline_outcome.incremental_utility(lambda_cost) if baseline else 0.0
        )
        candidate_values.append(candidate_value)
        baseline_values.append(baseline_value)
        by_source[candidate_outcome.source_id].append(candidate_value - baseline_value)
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
        "candidate_utility": _source_mean(candidate_outcomes, candidate_values),
        "baseline_utility": _source_mean(candidate_outcomes, baseline_values),
        "lower": _percentile(samples, alpha),
        "upper": _percentile(samples, 1.0 - alpha),
        "confidence_level": confidence_level,
        "resamples": resamples,
        "seed": seed,
        "resampling_unit": "source_id",
        "sources": len(sources),
    }
