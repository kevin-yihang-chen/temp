from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Literal, Sequence


RiskKind = Literal[
    "induced_harm",
    "net_negative_call_mass",
    "negative_net_value",
]
SelectionObjective = Literal["source_utility", "source_call_rate"]


@dataclass(frozen=True)
class AcquisitionCalibrationRow:
    """One frozen top-action prediction and its paired calibration outcome."""

    source_id: str
    score: float
    gain: float
    tool_cost: float = 1.0

    def __post_init__(self) -> None:
        if not str(self.source_id).strip():
            raise ValueError("calibration source_id must be non-empty")
        if not math.isfinite(self.score):
            raise ValueError("calibration score must be finite")
        if not math.isfinite(self.gain) or not -1.0 <= self.gain <= 1.0:
            raise ValueError("calibration gain must be finite and in [-1, 1]")
        if not math.isfinite(self.tool_cost) or self.tool_cost < 0.0:
            raise ValueError("calibration tool_cost must be finite and non-negative")


@dataclass(frozen=True)
class RiskConstraint:
    """A source-level bounded acquisition risk and its allowed population mean."""

    kind: RiskKind
    limit: float

    def __post_init__(self) -> None:
        if self.kind not in {
            "induced_harm",
            "net_negative_call_mass",
            "negative_net_value",
        }:
            raise ValueError(f"unsupported acquisition risk: {self.kind!r}")
        if not math.isfinite(self.limit) or self.limit <= 0.0:
            raise ValueError("risk limit must be finite and strictly positive")


def threshold_grid_from_training_scores(
    scores: Sequence[float],
    *,
    max_thresholds: int = 128,
) -> list[float]:
    """Build a deterministic finite threshold family without outcome access.

    The returned thresholds are observed training-score order statistics. The
    maximum and minimum are retained, and a deterministic evenly spaced subset
    is used when the unique family is larger than ``max_thresholds``.
    """

    if max_thresholds <= 0:
        raise ValueError("max_thresholds must be positive")
    values = sorted({float(value) for value in scores}, reverse=True)
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("training scores must be non-empty and finite")
    if len(values) <= max_thresholds:
        return values
    if max_thresholds == 1:
        return [values[0]]
    indices = {
        round(position * (len(values) - 1) / (max_thresholds - 1))
        for position in range(max_thresholds)
    }
    return [values[index] for index in sorted(indices)]


def threshold_grid_from_target_call_rates(
    scores: Sequence[float],
    target_call_rates: Sequence[float],
) -> list[float]:
    """Freeze strict-to-permissive score thresholds at training quantiles.

    The first threshold is one floating-point step above the largest observed
    score.  Remaining thresholds target increasing call rates and are deduped
    when tied scores induce the same call set.
    """

    ordered = sorted((float(value) for value in scores), reverse=True)
    if not ordered or any(not math.isfinite(value) for value in ordered):
        raise ValueError("training scores must be non-empty and finite")
    rates = [float(value) for value in target_call_rates]
    if (
        not rates
        or any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in rates)
        or rates != sorted(set(rates))
    ):
        raise ValueError("target call rates must be sorted, unique, and in (0,1]")
    thresholds = [math.nextafter(ordered[0], math.inf)]
    for rate in rates:
        index = min(len(ordered) - 1, max(0, math.ceil(rate * len(ordered)) - 1))
        threshold = ordered[index]
        if threshold < thresholds[-1]:
            thresholds.append(threshold)
    return thresholds


def bernoulli_relative_entropy(observed: float, null: float) -> float:
    """Return KL(Bernoulli(observed) || Bernoulli(null))."""

    if not 0.0 <= observed <= 1.0 or not 0.0 < null < 1.0:
        raise ValueError("Bernoulli means must lie in [0,1] and null in (0,1)")
    if observed == 0.0:
        return -math.log1p(-null)
    if observed == 1.0:
        return -math.log(null)
    return observed * math.log(observed / null) + (1.0 - observed) * math.log(
        (1.0 - observed) / (1.0 - null)
    )


def bounded_mean_lower_tail_pvalue(
    sample_mean: float,
    *,
    null_mean: float,
    n_samples: int,
    upper_bound: float = 1.0,
) -> float:
    """Conservative p-value for ``H0: E[X] >= null_mean``.

    For independent ``X`` in ``[0, upper_bound]``, the Bernoulli-KL Chernoff
    bound gives ``P(mean <= x) <= exp(-n KL(x/B || mu/B))``. Inverting this
    monotone tail bound yields a super-uniform one-sided p-value. Calibration
    calls this function on independent source-level means, never question rows.
    """

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if not math.isfinite(upper_bound) or upper_bound <= 0.0:
        raise ValueError("upper_bound must be finite and positive")
    if not math.isfinite(sample_mean) or not 0.0 <= sample_mean <= upper_bound:
        raise ValueError("sample_mean must lie in [0, upper_bound]")
    if not math.isfinite(null_mean) or not 0.0 < null_mean < upper_bound:
        raise ValueError("null_mean must lie strictly inside the bounded range")
    if sample_mean >= null_mean:
        return 1.0
    divergence = bernoulli_relative_entropy(
        sample_mean / upper_bound,
        null_mean / upper_bound,
    )
    exponent = -n_samples * divergence
    return 0.0 if exponent < -745.0 else math.exp(exponent)


def _risk_upper_bound(
    kind: RiskKind,
    *,
    lambda_cost: float,
    max_tool_cost: float,
) -> float:
    if kind in {"induced_harm", "net_negative_call_mass"}:
        return 1.0
    if kind == "negative_net_value":
        return 1.0 + lambda_cost * max_tool_cost
    raise ValueError(f"unsupported acquisition risk: {kind!r}")


def _risk_loss(
    row: AcquisitionCalibrationRow,
    *,
    called: bool,
    kind: RiskKind,
    lambda_cost: float,
) -> float:
    if not called:
        return 0.0
    net_value = row.gain - lambda_cost * row.tool_cost
    if kind == "induced_harm":
        return max(-row.gain, 0.0)
    if kind == "net_negative_call_mass":
        return float(net_value < 0.0)
    if kind == "negative_net_value":
        return max(-net_value, 0.0)
    raise ValueError(f"unsupported acquisition risk: {kind!r}")


def _threshold_summary(
    rows: Sequence[AcquisitionCalibrationRow],
    *,
    threshold: float | None,
    constraints: Sequence[RiskConstraint],
    lambda_cost: float,
    max_tool_cost: float,
    adjusted_p_cutoff: float,
) -> dict[str, object]:
    by_source: dict[str, list[AcquisitionCalibrationRow]] = {}
    for row in rows:
        by_source.setdefault(row.source_id, []).append(row)
    source_calls: list[float] = []
    source_utilities: list[float] = []
    source_losses: dict[str, list[float]] = {
        constraint.kind: [] for constraint in constraints
    }
    pooled_calls = 0
    pooled_utility = 0.0
    pooled_losses = {constraint.kind: 0.0 for constraint in constraints}
    for source_rows in by_source.values():
        calls: list[float] = []
        utilities: list[float] = []
        losses = {constraint.kind: [] for constraint in constraints}
        for row in source_rows:
            called = threshold is not None and row.score >= threshold
            call = float(called)
            utility = call * (row.gain - lambda_cost * row.tool_cost)
            calls.append(call)
            utilities.append(utility)
            pooled_calls += int(called)
            pooled_utility += utility
            for constraint in constraints:
                loss = _risk_loss(
                    row,
                    called=called,
                    kind=constraint.kind,
                    lambda_cost=lambda_cost,
                )
                losses[constraint.kind].append(loss)
                pooled_losses[constraint.kind] += loss
        source_calls.append(mean(calls))
        source_utilities.append(mean(utilities))
        for constraint in constraints:
            source_losses[constraint.kind].append(mean(losses[constraint.kind]))
    risks: dict[str, dict[str, object]] = {}
    accepted = True
    for constraint in constraints:
        risk_mean = mean(source_losses[constraint.kind])
        upper_bound = _risk_upper_bound(
            constraint.kind,
            lambda_cost=lambda_cost,
            max_tool_cost=max_tool_cost,
        )
        p_value = (
            0.0
            if threshold is None
            else bounded_mean_lower_tail_pvalue(
                risk_mean,
                null_mean=constraint.limit,
                n_samples=len(by_source),
                upper_bound=upper_bound,
            )
        )
        passed = threshold is None or p_value <= adjusted_p_cutoff
        accepted = accepted and passed
        risks[constraint.kind] = {
            "limit": constraint.limit,
            "upper_bound": upper_bound,
            "source_balanced_mean": risk_mean,
            "pooled_decision_mean": pooled_losses[constraint.kind] / len(rows),
            "p_value": p_value,
            "passed": passed,
        }
    return {
        "threshold": threshold,
        "answer_now_only": threshold is None,
        "n_sources": len(by_source),
        "n_decisions": len(rows),
        "source_call_rate": mean(source_calls),
        "pooled_call_rate": pooled_calls / len(rows),
        "source_utility": mean(source_utilities),
        "pooled_utility": pooled_utility / len(rows),
        "risks": risks,
        "risk_accepted": accepted,
    }


def calibrate_source_risk_threshold(
    rows: Sequence[AcquisitionCalibrationRow],
    thresholds: Sequence[float],
    *,
    constraints: Sequence[RiskConstraint],
    lambda_cost: float = 0.05,
    max_tool_cost: float = 1.0,
    family_error: float = 0.05,
    min_source_call_rate: float = 0.01,
    min_source_utility: float = 0.0,
    selection_objective: SelectionObjective = "source_utility",
) -> dict[str, object]:
    """Select a non-degenerate threshold with source-level LTT risk control.

    Thresholds must be constructed before calibration outcomes are read. Every
    threshold-risk pair is tested with a bounded-mean KL p-value and a
    Bonferroni cutoff, so any data-dependent selection from the jointly accepted
    set retains family-wise risk validity under independent exchangeable source
    groups. ``min_source_utility`` is an empirical selection constraint, not a
    finite-sample utility guarantee.
    """

    materialized = list(rows)
    if not materialized:
        raise ValueError("risk calibration rows must be non-empty")
    source_count = len({row.source_id for row in materialized})
    if source_count < 2:
        raise ValueError("risk calibration requires at least two source groups")
    frozen_thresholds = [float(value) for value in thresholds]
    if not frozen_thresholds or any(
        not math.isfinite(value) for value in frozen_thresholds
    ):
        raise ValueError("frozen thresholds must be non-empty and finite")
    if len(set(frozen_thresholds)) != len(frozen_thresholds):
        raise ValueError("frozen thresholds must be unique")
    frozen_constraints = list(constraints)
    if not frozen_constraints:
        raise ValueError("at least one risk constraint is required")
    kinds = [constraint.kind for constraint in frozen_constraints]
    if len(set(kinds)) != len(kinds):
        raise ValueError("risk constraint kinds must be unique")
    if lambda_cost < 0.0 or not math.isfinite(lambda_cost):
        raise ValueError("lambda_cost must be finite and non-negative")
    if max_tool_cost <= 0.0 or not math.isfinite(max_tool_cost):
        raise ValueError("max_tool_cost must be finite and positive")
    if any(row.tool_cost > max_tool_cost for row in materialized):
        raise ValueError("calibration tool_cost exceeds the frozen maximum")
    if not 0.0 < family_error < 1.0:
        raise ValueError("family_error must lie in (0,1)")
    if not 0.0 <= min_source_call_rate <= 1.0:
        raise ValueError("min_source_call_rate must lie in [0,1]")
    if not math.isfinite(min_source_utility):
        raise ValueError("min_source_utility must be finite")
    if selection_objective not in {"source_utility", "source_call_rate"}:
        raise ValueError(f"unsupported selection objective: {selection_objective}")
    for constraint in frozen_constraints:
        upper = _risk_upper_bound(
            constraint.kind,
            lambda_cost=lambda_cost,
            max_tool_cost=max_tool_cost,
        )
        if constraint.limit >= upper:
            raise ValueError(
                f"risk limit for {constraint.kind} must be below {upper}"
            )

    hypothesis_count = len(frozen_thresholds) * len(frozen_constraints)
    adjusted_p_cutoff = family_error / hypothesis_count
    candidates = [
        _threshold_summary(
            materialized,
            threshold=threshold,
            constraints=frozen_constraints,
            lambda_cost=lambda_cost,
            max_tool_cost=max_tool_cost,
            adjusted_p_cutoff=adjusted_p_cutoff,
        )
        for threshold in frozen_thresholds
    ]
    answer_now = _threshold_summary(
        materialized,
        threshold=None,
        constraints=frozen_constraints,
        lambda_cost=lambda_cost,
        max_tool_cost=max_tool_cost,
        adjusted_p_cutoff=adjusted_p_cutoff,
    )
    eligible = [
        candidate
        for candidate in candidates
        if bool(candidate["risk_accepted"])
        and float(candidate["source_call_rate"]) >= min_source_call_rate
        and float(candidate["source_utility"]) >= min_source_utility
    ]
    selected = None
    if eligible:
        primary = (
            "source_utility"
            if selection_objective == "source_utility"
            else "source_call_rate"
        )
        selected = max(
            eligible,
            key=lambda candidate: (
                float(candidate[primary]),
                float(candidate["source_utility"]),
                float(candidate["source_call_rate"]),
                float(candidate["threshold"]),
            ),
        )
    return {
        "scientific_status": (
            "source-level finite-threshold risk calibration; thresholds are "
            "frozen before calibration outcomes"
        ),
        "method": "bonferroni_bounded_mean_kl_ltt_v1",
        "lambda_cost": lambda_cost,
        "max_tool_cost": max_tool_cost,
        "family_error": family_error,
        "hypothesis_count": hypothesis_count,
        "adjusted_p_cutoff": adjusted_p_cutoff,
        "n_sources": source_count,
        "n_decisions": len(materialized),
        "constraints": [
            {"kind": constraint.kind, "limit": constraint.limit}
            for constraint in frozen_constraints
        ],
        "min_source_call_rate": min_source_call_rate,
        "min_source_utility": min_source_utility,
        "selection_objective": selection_objective,
        "selected_threshold": (
            float(selected["threshold"]) if selected is not None else None
        ),
        "selection_status": (
            "selected_non_degenerate_safe_threshold"
            if selected is not None
            else "no_non_degenerate_safe_threshold"
        ),
        "selected": selected,
        "answer_now": answer_now,
        "candidates": candidates,
    }


def calibrate_source_risk_threshold_fixed_sequence(
    rows: Sequence[AcquisitionCalibrationRow],
    thresholds: Sequence[float],
    *,
    constraints: Sequence[RiskConstraint],
    lambda_cost: float = 0.05,
    max_tool_cost: float = 1.0,
    family_error: float = 0.05,
    min_source_call_rate: float = 0.01,
    min_source_utility: float = 0.0,
) -> dict[str, object]:
    """Calibrate a nested threshold family with fixed-sequence LTT.

    Thresholds are tested from strict to permissive.  Each step Bonferroni-
    corrects only across the registered risk constraints; testing stops at the
    first joint failure and no more permissive threshold is summarized.  The
    selected policy is the most permissive preceding threshold that also meets
    the empirical non-degeneracy conditions.
    """

    materialized = list(rows)
    if not materialized:
        raise ValueError("risk calibration rows must be non-empty")
    source_count = len({row.source_id for row in materialized})
    if source_count < 2:
        raise ValueError("risk calibration requires at least two source groups")
    frozen_thresholds = [float(value) for value in thresholds]
    if not frozen_thresholds or any(
        not math.isfinite(value) for value in frozen_thresholds
    ):
        raise ValueError("frozen thresholds must be non-empty and finite")
    if any(
        strict <= permissive
        for strict, permissive in zip(frozen_thresholds, frozen_thresholds[1:])
    ):
        raise ValueError(
            "fixed-sequence thresholds must be unique and strictly descending"
        )
    frozen_constraints = list(constraints)
    if not frozen_constraints:
        raise ValueError("at least one risk constraint is required")
    kinds = [constraint.kind for constraint in frozen_constraints]
    if len(set(kinds)) != len(kinds):
        raise ValueError("risk constraint kinds must be unique")
    if lambda_cost < 0.0 or not math.isfinite(lambda_cost):
        raise ValueError("lambda_cost must be finite and non-negative")
    if max_tool_cost <= 0.0 or not math.isfinite(max_tool_cost):
        raise ValueError("max_tool_cost must be finite and positive")
    if any(row.tool_cost > max_tool_cost for row in materialized):
        raise ValueError("calibration tool_cost exceeds the frozen maximum")
    if not 0.0 < family_error < 1.0:
        raise ValueError("family_error must lie in (0,1)")
    if not 0.0 <= min_source_call_rate <= 1.0:
        raise ValueError("min_source_call_rate must lie in [0,1]")
    if not math.isfinite(min_source_utility):
        raise ValueError("min_source_utility must be finite")
    for constraint in frozen_constraints:
        upper = _risk_upper_bound(
            constraint.kind,
            lambda_cost=lambda_cost,
            max_tool_cost=max_tool_cost,
        )
        if constraint.limit >= upper:
            raise ValueError(
                f"risk limit for {constraint.kind} must be below {upper}"
            )

    adjusted_p_cutoff = family_error / len(frozen_constraints)
    tested: list[dict[str, object]] = []
    stopping_threshold = None
    for threshold in frozen_thresholds:
        candidate = _threshold_summary(
            materialized,
            threshold=threshold,
            constraints=frozen_constraints,
            lambda_cost=lambda_cost,
            max_tool_cost=max_tool_cost,
            adjusted_p_cutoff=adjusted_p_cutoff,
        )
        tested.append(candidate)
        if not bool(candidate["risk_accepted"]):
            stopping_threshold = threshold
            break
    eligible = [
        candidate
        for candidate in tested
        if bool(candidate["risk_accepted"])
        and float(candidate["source_call_rate"]) >= min_source_call_rate
        and float(candidate["source_utility"]) >= min_source_utility
    ]
    selected = eligible[-1] if eligible else None
    answer_now = _threshold_summary(
        materialized,
        threshold=None,
        constraints=frozen_constraints,
        lambda_cost=lambda_cost,
        max_tool_cost=max_tool_cost,
        adjusted_p_cutoff=adjusted_p_cutoff,
    )
    tested_count = len(tested)
    return {
        "scientific_status": (
            "source-level fixed-sequence risk calibration; nested thresholds "
            "are frozen before calibration outcomes"
        ),
        "method": "fixed_sequence_bounded_mean_kl_ltt_v1",
        "threshold_order": "strict_to_permissive_descending",
        "lambda_cost": lambda_cost,
        "max_tool_cost": max_tool_cost,
        "family_error": family_error,
        "per_step_hypothesis_count": len(frozen_constraints),
        "adjusted_p_cutoff": adjusted_p_cutoff,
        "n_sources": source_count,
        "n_decisions": len(materialized),
        "constraints": [
            {"kind": constraint.kind, "limit": constraint.limit}
            for constraint in frozen_constraints
        ],
        "min_source_call_rate": min_source_call_rate,
        "min_source_utility": min_source_utility,
        "selection_objective": (
            "most_permissive_pre_failure_with_non_degeneracy"
        ),
        "selected_threshold": (
            float(selected["threshold"]) if selected is not None else None
        ),
        "selection_status": (
            "selected_non_degenerate_safe_threshold"
            if selected is not None
            else "no_non_degenerate_safe_threshold"
        ),
        "selected": selected,
        "answer_now": answer_now,
        "tested_threshold_count": tested_count,
        "stopping_threshold": stopping_threshold,
        "candidates": tested,
        "untested_thresholds": frozen_thresholds[tested_count:],
    }
