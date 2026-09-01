from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, cast

from .dataset import DecisionKey, group_by_decision
from .infographicvqa_decar import DECAR_ACTION_IDS
from .schema import ActionRecord


DECAR_EVALUATION_SCHEMA = "infographicvqa_decar_oof_evaluation_v1"
DECAR_HYBRID_EVALUATION_SCHEMA = (
    "infographicvqa_decar_entropy_where_hybrid_evaluation_v1"
)
DECAR_ORACLE_WHERE_EVALUATION_SCHEMA = (
    "infographicvqa_entropy_oracle_where_factorization_evaluation_v1"
)
DECAR_BOOTSTRAP_SEED = 20_260_917
DECAR_BOOTSTRAP_RESAMPLES = 20_000
DECAR_BOOTSTRAP_CONFIDENCE = 0.95
DECAR_CALL_RATES = (0.005, 0.01, 0.02, 0.05, 0.10)
DECAR_LAMBDA_COST = 0.05
DECAR_VARIANTS = ("decar", "task_value_only", "loss_only", "no_harm_head")
DECAR_NON_ORACLE_BASELINES = (
    "answer_now",
    "entropy_random",
    "entropy_fixed_ug_grid_00",
    "entropy_gated_ug",
)
DECAR_STATIC_REFERENCES = (
    "charged_exhaustive_ug",
    "task_oracle_one_crop",
    "task_oracle_stopping",
)

_ADDITIVE_METRICS = (
    "baseline_anls",
    "final_anls",
    "anls_gain",
    "baseline_exact_accuracy",
    "final_exact_accuracy",
    "exact_accuracy_gain",
    "utility",
    "executed_crops",
    "call",
    "helpful_call",
    "helpful_state",
    "induced_harm",
    "harmful_call",
    "negative_utility_call",
    "negative_utility_magnitude",
    "gate_false_negative",
    "missed_positive_utility",
    "action_selection_regret",
    "oracle_stop_regret",
    "entropy_disagreement",
    "scgr",
)
_RATIO_METRICS = {
    "gain_per_call": ("anls_gain", "call"),
    "helpful_call_precision": ("helpful_call", "call"),
    "helpful_state_recovery": ("helpful_call", "helpful_state"),
    "entropy_disagreement_per_call": ("entropy_disagreement", "call"),
    "scgr_per_call": ("scgr", "call"),
}
_FORBIDDEN_PREDICTION_FIELDS = {
    "answer_after",
    "answer_before",
    "correct_after",
    "correct_before",
    "delta",
    "entropy_after",
    "gain",
    "harm",
    "oracle_action_id",
    "reward",
    "target",
    "teacher_nll",
    "utility",
}


@dataclass(frozen=True)
class DecarOutcome:
    key: DecisionKey
    source_id: str
    image_id: str
    baseline: ActionRecord
    crops: tuple[ActionRecord, ...]

    @property
    def entropy_action(self) -> ActionRecord:
        return min(self.crops, key=lambda row: (row.entropy_after, row.action_id))

    @property
    def task_action(self) -> ActionRecord:
        return min(self.crops, key=lambda row: (-row.delta_success, row.action_id))

    @property
    def helpful_state(self) -> float:
        return float(any(row.delta_success > 0.0 for row in self.crops))

    @property
    def oracle_stop_utility(self) -> float:
        return max(0.0, self.task_action.delta_success - DECAR_LAMBDA_COST)


@dataclass(frozen=True)
class PredictionVariant:
    action_id: str
    predicted_gap: float
    predicted_margin: float
    score: float
    eligible: bool


@dataclass(frozen=True)
class DecarPrediction:
    key: DecisionKey
    source_id: str
    image_id: str
    outer_fold: int
    variants: Mapping[str, PredictionVariant]


def _finite(value: object, name: str) -> float:
    number = float(cast(Any, value))
    if not math.isfinite(number):
        raise ValueError(f"DECAR evaluation {name} must be finite")
    return number


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0.0 else None


def _prediction_forbidden_fields(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).lower()
            if key in _FORBIDDEN_PREDICTION_FIELDS:
                found.add(key)
            found.update(_prediction_forbidden_fields(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            found.update(_prediction_forbidden_fields(child))
    return found


def build_decar_outcomes(
    records: Sequence[ActionRecord],
    *,
    expected_decisions: int | None = None,
    expected_sources: int | None = None,
) -> dict[DecisionKey, DecarOutcome]:
    grouped = group_by_decision(records)
    outcomes: dict[DecisionKey, DecarOutcome] = {}
    for key, siblings in grouped.items():
        answers = [row for row in siblings if row.action_type == "ANSWER"]
        crops = tuple(
            sorted(
                (row for row in siblings if row.action_type == "ZOOM"),
                key=lambda row: row.action_id,
            )
        )
        if len(answers) != 1 or tuple(row.action_id for row in crops) != tuple(
            DECAR_ACTION_IDS
        ):
            raise ValueError(f"DECAR evaluation sibling contract failed for {key!r}")
        if any(
            not math.isfinite(value)
            for row in siblings
            for value in (
                row.correct_before,
                row.correct_after,
                row.entropy_before,
                row.entropy_after,
            )
        ):
            raise ValueError("DECAR evaluation rollout contains non-finite outcomes")
        outcomes[key] = DecarOutcome(
            key=key,
            source_id=answers[0].source_id,
            image_id=answers[0].image_id,
            baseline=answers[0],
            crops=crops,
        )
    if expected_decisions is not None and len(outcomes) != expected_decisions:
        raise ValueError("DECAR evaluation decision population changed")
    source_count = len({row.source_id for row in outcomes.values()})
    if expected_sources is not None and source_count != expected_sources:
        raise ValueError("DECAR evaluation source population changed")
    return outcomes


def _validate_probability(value: object, name: str) -> float:
    probability = _finite(value, name)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"DECAR evaluation {name} is outside [0,1]")
    return probability


def _parse_variant(name: str, value: object) -> PredictionVariant:
    if not isinstance(value, Mapping):
        raise ValueError(f"DECAR evaluation {name} prediction is not a mapping")
    common = {
        "selected_action_id",
        "predicted_gap",
        "predicted_margin",
        "score",
        "eligible",
    }
    probabilities: tuple[float, ...]
    if name in {"decar", "task_value_only"}:
        expected = common | {
            "rescue_probability",
            "neutral_probability",
            "harm_probability",
            "predicted_delta",
        }
        probabilities = (
            _validate_probability(
                value.get("rescue_probability"), "rescue probability"
            ),
            _validate_probability(
                value.get("neutral_probability"), "neutral probability"
            ),
            _validate_probability(value.get("harm_probability"), "harm probability"),
        )
    elif name == "no_harm_head":
        expected = common | {
            "rescue_probability",
            "other_probability",
            "predicted_delta",
        }
        probabilities = (
            _validate_probability(
                value.get("rescue_probability"), "rescue probability"
            ),
            _validate_probability(value.get("other_probability"), "other probability"),
        )
    elif name == "loss_only":
        expected = common
        probabilities = ()
    else:  # pragma: no cover - caller checks the exact variant family
        raise ValueError(f"unknown DECAR evaluation variant: {name}")
    if set(value) != expected:
        raise ValueError(f"DECAR evaluation {name} prediction schema changed")
    if probabilities and not math.isclose(sum(probabilities), 1.0, abs_tol=1e-5):
        raise ValueError(f"DECAR evaluation {name} probabilities do not sum to one")
    if name != "loss_only":
        _finite(value["predicted_delta"], f"{name} predicted delta")
    action_id = str(value.get("selected_action_id", ""))
    eligible = value.get("eligible")
    if action_id not in DECAR_ACTION_IDS or not isinstance(eligible, bool):
        raise ValueError(f"DECAR evaluation {name} action/eligibility is invalid")
    if name == "loss_only" and eligible is not True:
        raise ValueError("DECAR evaluation loss_only must always be eligible")
    return PredictionVariant(
        action_id=action_id,
        predicted_gap=_finite(value.get("predicted_gap"), f"{name} predicted gap"),
        predicted_margin=_finite(
            value.get("predicted_margin"), f"{name} predicted margin"
        ),
        score=_finite(value.get("score"), f"{name} score"),
        eligible=eligible,
    )


def parse_decar_predictions(
    rows: Sequence[Mapping[str, Any]],
    outcomes: Mapping[DecisionKey, DecarOutcome],
) -> dict[DecisionKey, DecarPrediction]:
    parsed: dict[DecisionKey, DecarPrediction] = {}
    source_fold: dict[str, int] = {}
    expected_top_level = {
        "schema",
        "state_id",
        "replicate_id",
        "image_id",
        "source_id",
        "outer_fold",
        "variants",
    }
    for row in rows:
        forbidden = _prediction_forbidden_fields(row)
        if forbidden:
            raise ValueError(
                f"DECAR prediction rows contain forbidden outcomes: {sorted(forbidden)}"
            )
        if set(row) != expected_top_level or row.get("schema") != (
            "infographicvqa_decar_oof_prediction_v1"
        ):
            raise ValueError("DECAR evaluation prediction row schema changed")
        key = (str(row.get("state_id", "")), str(row.get("replicate_id", "")))
        if not all(key) or key in parsed or key not in outcomes:
            raise ValueError("DECAR evaluation prediction key is invalid")
        outcome = outcomes[key]
        source_id = str(row.get("source_id", ""))
        image_id = str(row.get("image_id", ""))
        outer_fold = int(row.get("outer_fold", -1))
        if (
            source_id != outcome.source_id
            or image_id != outcome.image_id
            or not 0 <= outer_fold < 5
        ):
            raise ValueError("DECAR evaluation prediction identity mismatch")
        previous_fold = source_fold.setdefault(source_id, outer_fold)
        if previous_fold != outer_fold:
            raise ValueError("DECAR evaluation source crosses outer folds")
        variants = row.get("variants")
        if not isinstance(variants, Mapping) or set(variants) != set(DECAR_VARIANTS):
            raise ValueError("DECAR evaluation variant family changed")
        parsed[key] = DecarPrediction(
            key=key,
            source_id=source_id,
            image_id=image_id,
            outer_fold=outer_fold,
            variants={
                name: _parse_variant(name, variants[name]) for name in DECAR_VARIANTS
            },
        )
    if set(parsed) != set(outcomes):
        raise ValueError("DECAR evaluation prediction coverage is incomplete")
    return parsed


def complete_tie_top_keys(
    scores: Mapping[DecisionKey, float],
    *,
    target_calls: int,
) -> tuple[set[DecisionKey], dict[str, Any]]:
    """Select a score threshold, retaining every exact boundary tie."""

    if not 0 <= target_calls <= len(scores):
        raise ValueError("DECAR complete-tie target is invalid")
    values = {key: _finite(value, "ranking score") for key, value in scores.items()}
    if target_calls == 0:
        return set(), {
            "threshold": None,
            "target_calls": 0,
            "actual_calls": 0,
            "boundary_ties": 0,
            "ties_preserved": True,
        }
    ordered = sorted(values.values(), reverse=True)
    threshold = ordered[min(target_calls, len(ordered)) - 1]
    selected = {key for key, value in values.items() if value >= threshold}
    return selected, {
        "threshold": threshold,
        "target_calls": target_calls,
        "actual_calls": len(selected),
        "boundary_ties": sum(value == threshold for value in values.values()),
        "ties_preserved": True,
    }


def _ranked_exact_keys(
    variants: Mapping[DecisionKey, PredictionVariant],
    *,
    target_calls: int,
) -> tuple[set[DecisionKey], dict[str, Any]]:
    eligible = [(key, value) for key, value in variants.items() if value.eligible]
    ordered = sorted(
        eligible,
        key=lambda item: (
            -item[1].score,
            -item[1].predicted_gap,
            item[0][0],
            item[0][1],
        ),
    )
    selected = {key for key, _ in ordered[:target_calls]}
    return selected, {
        "target_calls": target_calls,
        "actual_calls": len(selected),
        "eligible_calls": len(eligible),
        "matched_call_count": len(selected) == target_calls,
        "tie_break": "score_desc_gap_desc_state_id_replicate_id",
        "selection_uses_outcomes": False,
    }


def _complete_tie_exact_match(
    scores: Mapping[DecisionKey, float],
    *,
    target_calls: int,
) -> tuple[set[DecisionKey], dict[str, Any]]:
    if not scores or not 0 <= target_calls <= len(scores):
        raise ValueError("DECAR entropy-gate target is invalid")
    if target_calls == 0:
        return set(), {
            "threshold": None,
            "target_calls": 0,
            "actual_calls": 0,
            "matched_call_count": True,
            "ties_preserved": True,
            "selection_uses_outcomes": False,
        }
    candidates: list[tuple[int, float, set[DecisionKey]]] = []
    for threshold in sorted(
        {_finite(value, "entropy gate score") for value in scores.values()},
        reverse=True,
    ):
        selected = {key for key, value in scores.items() if value >= threshold}
        candidates.append((abs(len(selected) - target_calls), threshold, selected))
    _, threshold, selected = min(
        candidates,
        key=lambda item: (
            item[0],
            int(len(item[2]) > target_calls),
            -len(item[2]),
            -item[1],
        ),
    )
    return selected, {
        "threshold": threshold,
        "target_calls": target_calls,
        "actual_calls": len(selected),
        "matched_call_count": len(selected) == target_calls,
        "ties_preserved": True,
        "selection_uses_outcomes": False,
    }


def _decision_metrics(
    outcome: DecarOutcome,
    *,
    action_probabilities: Mapping[str, float] | None,
    executed_crops: float,
) -> dict[str, float]:
    probabilities = dict(action_probabilities or {})
    called = bool(probabilities)
    if called:
        if set(probabilities) - set(DECAR_ACTION_IDS):
            raise ValueError("DECAR policy selected an unknown crop")
        if any(
            value < 0.0 or not math.isfinite(value) for value in probabilities.values()
        ):
            raise ValueError("DECAR policy action probabilities are invalid")
        if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-12):
            raise ValueError("DECAR policy action probabilities do not sum to one")
        if not math.isfinite(executed_crops) or executed_crops <= 0.0:
            raise ValueError("DECAR called policy has invalid execution cost")
    elif executed_crops != 0.0:
        raise ValueError("DECAR stopped policy has nonzero execution cost")
    crop_by_action = {row.action_id: row for row in outcome.crops}
    gain = sum(
        probability * crop_by_action[action_id].delta_success
        for action_id, probability in probabilities.items()
    )
    utility = gain - DECAR_LAMBDA_COST * executed_crops
    baseline_exact = float(math.isclose(outcome.baseline.correct_before, 1.0))
    final_exact = (
        sum(
            probability
            * float(math.isclose(crop_by_action[action_id].correct_after, 1.0))
            for action_id, probability in probabilities.items()
        )
        if called
        else baseline_exact
    )
    best_crop_gain = outcome.task_action.delta_success
    entropy_action_id = outcome.entropy_action.action_id
    negative_utility_call = sum(
        probability
        * float(
            crop_by_action[action_id].delta_success - DECAR_LAMBDA_COST * executed_crops
            < 0.0
        )
        for action_id, probability in probabilities.items()
    )
    return {
        "baseline_anls": outcome.baseline.correct_before,
        "final_anls": outcome.baseline.correct_before + gain,
        "anls_gain": gain,
        "baseline_exact_accuracy": baseline_exact,
        "final_exact_accuracy": final_exact,
        "exact_accuracy_gain": final_exact - baseline_exact,
        "utility": utility,
        "executed_crops": executed_crops,
        "call": float(called),
        "helpful_call": sum(
            probability * float(crop_by_action[action_id].delta_success > 0.0)
            for action_id, probability in probabilities.items()
        ),
        "helpful_state": outcome.helpful_state,
        "induced_harm": sum(
            probability * max(-crop_by_action[action_id].delta_success, 0.0)
            for action_id, probability in probabilities.items()
        ),
        "harmful_call": sum(
            probability * float(crop_by_action[action_id].delta_success < 0.0)
            for action_id, probability in probabilities.items()
        ),
        "negative_utility_call": negative_utility_call,
        "negative_utility_magnitude": (max(-utility, 0.0) if called else 0.0),
        "gate_false_negative": float(not called and outcome.oracle_stop_utility > 0.0),
        "missed_positive_utility": (outcome.oracle_stop_utility if not called else 0.0),
        "action_selection_regret": (best_crop_gain - gain if called else 0.0),
        "oracle_stop_regret": outcome.oracle_stop_utility - utility,
        "entropy_disagreement": sum(
            probability * float(action_id != entropy_action_id)
            for action_id, probability in probabilities.items()
        ),
        "scgr": sum(
            probability
            * float(
                crop_by_action[action_id].delta_entropy > 0.0
                and crop_by_action[action_id].delta_success < 0.0
            )
            for action_id, probability in probabilities.items()
        ),
    }


def _policy_metrics(
    outcomes: Mapping[DecisionKey, DecarOutcome],
    *,
    called_keys: set[DecisionKey],
    action_by_key: Mapping[DecisionKey, str] | None = None,
    random_action: bool = False,
    entropy_action: bool = False,
    task_action: bool = False,
    executions_per_call: float = 1.0,
) -> dict[DecisionKey, dict[str, float]]:
    result: dict[DecisionKey, dict[str, float]] = {}
    for key, outcome in outcomes.items():
        probabilities: dict[str, float] | None = None
        if key in called_keys:
            if random_action:
                probabilities = {action_id: 0.25 for action_id in DECAR_ACTION_IDS}
            elif entropy_action:
                probabilities = {outcome.entropy_action.action_id: 1.0}
            elif task_action:
                probabilities = {outcome.task_action.action_id: 1.0}
            elif action_by_key is not None and key in action_by_key:
                probabilities = {action_by_key[key]: 1.0}
            else:
                raise ValueError("DECAR called policy has no action")
        result[key] = _decision_metrics(
            outcome,
            action_probabilities=probabilities,
            executed_crops=executions_per_call if probabilities else 0.0,
        )
    return result


def _aggregate_policy(
    values: Mapping[DecisionKey, Mapping[str, float]],
    outcomes: Mapping[DecisionKey, DecarOutcome],
    sources: Sequence[str],
) -> dict[str, Any]:
    if set(values) != set(outcomes):
        raise ValueError("DECAR policy metric coverage changed")
    grouped: dict[str, dict[str, list[float]]] = {
        source: {name: [] for name in _ADDITIVE_METRICS} for source in sources
    }
    for key in sorted(outcomes):
        source = outcomes[key].source_id
        row = values[key]
        if set(row) != set(_ADDITIVE_METRICS):
            raise ValueError("DECAR policy metric schema changed")
        for name in _ADDITIVE_METRICS:
            grouped[source][name].append(_finite(row[name], name))
    source_values = {
        source: {
            name: sum(grouped[source][name]) / len(grouped[source][name])
            for name in _ADDITIVE_METRICS
        }
        for source in sources
    }
    question_point: dict[str, float | None] = {
        name: sum(float(values[key][name]) for key in values) / len(values)
        for name in _ADDITIVE_METRICS
    }
    source_point: dict[str, float | None] = {
        name: sum(source_values[source][name] for source in sources) / len(sources)
        for name in _ADDITIVE_METRICS
    }
    for points in (question_point, source_point):
        for name, (numerator, denominator) in _RATIO_METRICS.items():
            points[name] = _safe_ratio(
                cast(float, points[numerator]),
                cast(float, points[denominator]),
            )
    called_sources = {
        outcomes[key].source_id for key, row in values.items() if row["call"] > 0.0
    }
    return {
        "question_balanced": question_point,
        "source_balanced": source_point,
        "raw_calls": sum(row["call"] for row in values.values()),
        "distinct_called_sources": len(called_sources),
        "source_values": source_values,
    }


def _source_concentration(
    values: Mapping[DecisionKey, Mapping[str, float]],
    outcomes: Mapping[DecisionKey, DecarOutcome],
) -> dict[str, Any]:
    import numpy as np  # type: ignore[import-not-found]

    calls_by_source: dict[str, float] = defaultdict(float)
    utility_by_source: dict[str, list[float]] = defaultdict(list)
    for key, row in values.items():
        source = outcomes[key].source_id
        calls_by_source[source] += float(row["call"])
        utility_by_source[source].append(float(row["utility"]))
    sources = sorted(utility_by_source)
    calls = np.asarray(
        [calls_by_source[source] for source in sources], dtype=np.float64
    )
    source_utilities = np.asarray(
        [
            sum(utility_by_source[source]) / len(utility_by_source[source])
            for source in sources
        ],
        dtype=np.float64,
    )
    total_calls = float(calls.sum())
    shares = calls / total_calls if total_calls > 0.0 else np.zeros_like(calls)
    ordered = np.sort(shares)[::-1]

    def top_fraction(fraction: float) -> float:
        count = max(1, math.ceil(fraction * len(sources)))
        return float(ordered[:count].sum())

    return {
        "sources": len(sources),
        "sources_with_calls": int((calls > 0.0).sum()),
        "sources_with_negative_mean_utility": int((source_utilities < 0.0).sum()),
        "call_source_hhi": float((shares**2).sum()),
        "top_1pct_sources_call_fraction": top_fraction(0.01),
        "top_5pct_sources_call_fraction": top_fraction(0.05),
        "top_10pct_sources_call_fraction": top_fraction(0.10),
        "source_utility_quantiles": {
            "q00": float(np.quantile(source_utilities, 0.00)),
            "q10": float(np.quantile(source_utilities, 0.10)),
            "q25": float(np.quantile(source_utilities, 0.25)),
            "q50": float(np.quantile(source_utilities, 0.50)),
            "q75": float(np.quantile(source_utilities, 0.75)),
            "q90": float(np.quantile(source_utilities, 0.90)),
            "q100": float(np.quantile(source_utilities, 1.00)),
        },
    }


def _bootstrap_all_policies(
    aggregates: Mapping[str, Mapping[str, Any]],
    sources: Sequence[str],
    bootstrap_indices: Any,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    import numpy as np  # type: ignore[import-not-found]

    indices = np.asarray(bootstrap_indices)
    if (
        indices.ndim != 2
        or indices.shape[1] != len(sources)
        or indices.shape[0] <= 0
        or not np.issubdtype(indices.dtype, np.integer)
        or int(indices.min()) < 0
        or int(indices.max()) >= len(sources)
    ):
        raise ValueError("DECAR source-bootstrap indices are invalid")
    columns: list[tuple[str, str]] = []
    arrays: list[Any] = []
    for policy_name in sorted(aggregates):
        source_values = aggregates[policy_name]["source_values"]
        for metric_name in _ADDITIVE_METRICS:
            columns.append((policy_name, metric_name))
            arrays.append(
                np.asarray(
                    [source_values[source][metric_name] for source in sources],
                    dtype=np.float64,
                )
            )
    matrix = np.stack(arrays, axis=1)
    draws = np.empty((indices.shape[0], matrix.shape[1]), dtype=np.float64)
    batch_size = 256
    for start in range(0, indices.shape[0], batch_size):
        stop = min(indices.shape[0], start + batch_size)
        current = indices[start:stop]
        counts = np.zeros((stop - start, len(sources)), dtype=np.float64)
        for offset, row in enumerate(current):
            counts[offset] = np.bincount(row, minlength=len(sources))
        draws[start:stop] = counts @ matrix / len(sources)
    alpha = 1.0 - DECAR_BOOTSTRAP_CONFIDENCE
    position = {column: index for index, column in enumerate(columns)}

    def interval(values: Any) -> dict[str, float | None]:
        return {
            "ci_low": float(np.quantile(values, alpha / 2.0)),
            "ci_high": float(np.quantile(values, 1.0 - alpha / 2.0)),
        }

    policy_bootstrap: dict[str, Any] = {}
    utility_draws: dict[str, Any] = {}
    for policy_name in sorted(aggregates):
        additive: dict[str, Any] = {}
        for metric_name in _ADDITIVE_METRICS:
            values = draws[:, position[(policy_name, metric_name)]]
            additive[metric_name] = {
                "point_estimate": float(
                    aggregates[policy_name]["source_balanced"][metric_name]
                ),
                **interval(values),
            }
            if metric_name == "utility":
                utility_draws[policy_name] = values
        ratios: dict[str, Any] = {}
        for metric_name, (numerator, denominator) in _RATIO_METRICS.items():
            numerator_values = draws[:, position[(policy_name, numerator)]]
            denominator_values = draws[:, position[(policy_name, denominator)]]
            valid = denominator_values > 0.0
            ratio_values = numerator_values[valid] / denominator_values[valid]
            ratios[metric_name] = {
                "point_estimate": aggregates[policy_name]["source_balanced"][
                    metric_name
                ],
                "valid_resamples": int(valid.sum()),
                **(
                    interval(ratio_values)
                    if ratio_values.size
                    else {"ci_low": None, "ci_high": None}
                ),
            }
        policy_bootstrap[policy_name] = {
            "additive": additive,
            "ratios": ratios,
        }
    paired_differences: dict[str, dict[str, Any]] = {}
    for policy_name in sorted(aggregates):
        if policy_name.endswith(
            (
                "/decar",
                "/entropy_when_decar_where",
                "/entropy_when_task_oracle_where",
                "/relative_teacher_entropy",
            )
        ):
            prefix = policy_name.rsplit("/", 1)[0]
            oracle_name = f"{prefix}/entropy_when_task_oracle_where"
            if oracle_name in aggregates and policy_name != oracle_name:
                continue
            paired_differences[prefix] = {}
            for comparator in sorted(aggregates):
                if comparator.startswith(prefix + "/") and comparator != policy_name:
                    difference = utility_draws[policy_name] - utility_draws[comparator]
                    point = (
                        aggregates[policy_name]["source_balanced"]["utility"]
                        - aggregates[comparator]["source_balanced"]["utility"]
                    )
                    paired_differences[prefix][comparator.rsplit("/", 1)[1]] = {
                        "point_estimate": point,
                        **interval(difference),
                    }
    metadata = {
        "method": "paired_iid_whole_source_percentile_bootstrap",
        "n_sources": len(sources),
        "n_resamples": int(indices.shape[0]),
        "confidence_level": DECAR_BOOTSTRAP_CONFIDENCE,
        "seed": DECAR_BOOTSTRAP_SEED,
        "same_indices_for_all_policies_and_differences": True,
    }
    return {"metadata": metadata, "policies": policy_bootstrap}, paired_differences


def evaluate_decar_oof(
    records: Sequence[ActionRecord],
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_indices: Any,
    target_call_rates: Sequence[float] = DECAR_CALL_RATES,
    expected_decisions: int | None = 23_946,
    expected_sources: int | None = 2_204,
) -> dict[str, Any]:
    """Evaluate frozen outcome-free DECAR OOF predictions on train siblings."""

    rates = tuple(float(value) for value in target_call_rates)
    if (
        not rates
        or rates != tuple(sorted(set(rates)))
        or any(not 0.0 < value <= 1.0 for value in rates)
    ):
        raise ValueError("DECAR evaluation call-rate family is invalid")
    outcomes = build_decar_outcomes(
        records,
        expected_decisions=expected_decisions,
        expected_sources=expected_sources,
    )
    predictions = parse_decar_predictions(prediction_rows, outcomes)
    keys = sorted(outcomes)
    sources = sorted({row.source_id for row in outcomes.values()})
    entropy_scores = {key: outcomes[key].baseline.entropy_before for key in keys}
    variant_predictions = {
        name: {key: predictions[key].variants[name] for key in keys}
        for name in DECAR_VARIANTS
    }
    action_by_variant = {
        name: {key: value.action_id for key, value in variants.items()}
        for name, variants in variant_predictions.items()
    }

    policy_values: dict[str, dict[DecisionKey, dict[str, float]]] = {}
    operating: list[dict[str, Any]] = []
    all_keys = set(keys)
    static_definitions = {
        "answer_now": _policy_metrics(outcomes, called_keys=set()),
        "charged_exhaustive_ug": _policy_metrics(
            outcomes,
            called_keys=all_keys,
            entropy_action=True,
            executions_per_call=4.0,
        ),
        "task_oracle_one_crop": _policy_metrics(
            outcomes,
            called_keys=all_keys,
            task_action=True,
        ),
        "task_oracle_stopping": _policy_metrics(
            outcomes,
            called_keys={
                key for key in keys if outcomes[key].oracle_stop_utility > 0.0
            },
            task_action=True,
        ),
    }
    for name, values in static_definitions.items():
        policy_values[f"static/{name}"] = values

    for rate in rates:
        point_name = f"rate-{rate:.3f}"
        nominal_calls = math.ceil(rate * len(keys))
        primary_scores = {
            key: value.score
            for key, value in variant_predictions["decar"].items()
            if value.eligible
        }
        primary_calls, primary_selection = complete_tie_top_keys(
            primary_scores,
            target_calls=min(nominal_calls, len(primary_scores)),
        )
        actual_calls = len(primary_calls)
        selections: dict[str, set[DecisionKey]] = {"decar": primary_calls}
        selection_audits: dict[str, Any] = {"decar": primary_selection}
        for name in ("task_value_only", "loss_only", "no_harm_head"):
            selected, audit = _ranked_exact_keys(
                variant_predictions[name], target_calls=actual_calls
            )
            selections[name] = selected
            selection_audits[name] = audit
        entropy_calls, entropy_audit = _complete_tie_exact_match(
            entropy_scores,
            target_calls=actual_calls,
        )
        ug_target_calls = actual_calls // 4
        entropy_ug_calls, entropy_ug_audit = _complete_tie_exact_match(
            entropy_scores,
            target_calls=ug_target_calls,
        )
        selection_audits["entropy_gate_random_and_fixed"] = entropy_audit
        selection_audits["entropy_gated_ug"] = entropy_ug_audit

        point_values: dict[str, dict[DecisionKey, dict[str, float]]] = {}
        for name in DECAR_VARIANTS:
            point_values[name] = _policy_metrics(
                outcomes,
                called_keys=selections[name],
                action_by_key=action_by_variant[name],
            )
        point_values["answer_now"] = static_definitions["answer_now"]
        point_values["entropy_random"] = _policy_metrics(
            outcomes,
            called_keys=entropy_calls,
            random_action=True,
        )
        point_values["entropy_fixed_ug_grid_00"] = _policy_metrics(
            outcomes,
            called_keys=entropy_calls,
            action_by_key={key: "ug-grid-00" for key in keys},
        )
        point_values["entropy_gated_ug"] = _policy_metrics(
            outcomes,
            called_keys=entropy_ug_calls,
            entropy_action=True,
            executions_per_call=4.0,
        )
        for name, values in point_values.items():
            policy_values[f"{point_name}/{name}"] = values
        operating.append(
            {
                "name": point_name,
                "nominal_question_call_rate": rate,
                "nominal_calls": nominal_calls,
                "primary_actual_calls": actual_calls,
                "selection_audits": selection_audits,
            }
        )

    aggregates = {
        name: _aggregate_policy(values, outcomes, sources)
        for name, values in policy_values.items()
    }
    bootstrap, paired_differences = _bootstrap_all_policies(
        aggregates, sources, bootstrap_indices
    )
    public_aggregates = {
        name: {key: value for key, value in aggregate.items() if key != "source_values"}
        for name, aggregate in aggregates.items()
    }

    points: list[dict[str, Any]] = []
    for registered in operating:
        point_name = str(registered["name"])
        names = [
            *DECAR_VARIANTS,
            *DECAR_NON_ORACLE_BASELINES,
        ]
        policies = {name: public_aggregates[f"{point_name}/{name}"] for name in names}
        policy_bootstrap = {
            name: bootstrap["policies"][f"{point_name}/{name}"] for name in names
        }
        primary = policies["decar"]["source_balanced"]
        primary_budget = float(primary["executed_crops"])
        feasible_baselines = [
            name
            for name in DECAR_NON_ORACLE_BASELINES
            if float(policies[name]["source_balanced"]["executed_crops"])
            <= primary_budget + 1e-15
        ]
        strongest_baseline = max(
            feasible_baselines,
            key=lambda name: (
                float(policies[name]["source_balanced"]["utility"]),
                name,
            ),
        )
        selection_audits = registered["selection_audits"]
        audit_passed = all(
            bool(selection_audits[name].get("matched_call_count", True))
            for name in selection_audits
        ) and int(registered["primary_actual_calls"]) == int(
            selection_audits["decar"]["actual_calls"]
        )
        primary_utility_interval = policy_bootstrap["decar"]["additive"]["utility"]
        rules = {
            "minimum_calls_and_sources": (
                float(policies["decar"]["raw_calls"]) >= 100.0
                and int(policies["decar"]["distinct_called_sources"]) >= 50
            ),
            "source_utility_ci_low_strictly_positive": float(
                primary_utility_interval["ci_low"]
            )
            > 0.0,
            "strictly_above_every_feasible_non_oracle_baseline": all(
                float(primary["utility"])
                > float(policies[name]["source_balanced"]["utility"])
                for name in feasible_baselines
            ),
            "strictly_above_every_registered_ablation": all(
                float(primary["utility"])
                > float(policies[name]["source_balanced"]["utility"])
                for name in ("task_value_only", "loss_only", "no_harm_head")
            ),
            "harm_no_greater_than_no_harm_and_strongest_baseline": all(
                float(primary[metric])
                <= float(policies[comparator]["source_balanced"][metric]) + 1e-15
                for metric in ("induced_harm", "negative_utility_call")
                for comparator in ("no_harm_head", strongest_baseline)
            ),
            "all_audits_passed": audit_passed,
        }
        decomposition_metrics = {
            "action_choice_regret": "action_selection_regret",
            "gate_false_positive_mass": "negative_utility_call",
            "gate_false_positive_loss": "negative_utility_magnitude",
            "gate_false_negative_mass": "gate_false_negative",
            "gate_false_negative_loss": "missed_positive_utility",
        }
        failure_decomposition = {
            balance: {
                name: float(policies["decar"][balance][metric])
                for name, metric in decomposition_metrics.items()
            }
            for balance in ("question_balanced", "source_balanced")
        }
        failure_decomposition["source_concentration"] = _source_concentration(
            policy_values[f"{point_name}/decar"], outcomes
        )
        points.append(
            {
                **registered,
                "policies": policies,
                "source_bootstrap": policy_bootstrap,
                "paired_source_utility_differences": paired_differences[point_name],
                "feasible_non_oracle_baselines": feasible_baselines,
                "strongest_feasible_non_oracle_baseline": strongest_baseline,
                "qualification_rules": rules,
                "qualified": all(rules.values()),
                "failure_decomposition": failure_decomposition,
            }
        )
    qualified = [row for row in points if row["qualified"]]
    selected_point = (
        min(
            qualified,
            key=lambda row: (
                -float(row["policies"]["decar"]["source_balanced"]["utility"]),
                float(row["policies"]["decar"]["source_balanced"]["induced_harm"]),
                float(row["nominal_question_call_rate"]),
            ),
        )
        if qualified
        else None
    )
    static = {
        name: {
            **public_aggregates[f"static/{name}"],
            "source_bootstrap": bootstrap["policies"][f"static/{name}"],
        }
        for name in ("answer_now", *DECAR_STATIC_REFERENCES)
    }
    return {
        "schema": DECAR_EVALUATION_SCHEMA,
        "scientific_status": "registered official-train source-OOF evaluation",
        "population": {
            "decisions": len(outcomes),
            "sources": len(sources),
            "images": len({row.image_id for row in outcomes.values()}),
        },
        "lambda_cost": DECAR_LAMBDA_COST,
        "registered_call_rates": list(rates),
        "operating_points": points,
        "static_references": static,
        "bootstrap": bootstrap["metadata"],
        "all_prediction_rows_outcome_free": True,
        "decision": (
            "decar_advanced_to_sealed_validation"
            if selected_point is not None
            else "decar_not_advanced"
        ),
        "selected_operating_point": (
            None
            if selected_point is None
            else {
                "name": selected_point["name"],
                "nominal_question_call_rate": selected_point[
                    "nominal_question_call_rate"
                ],
                "primary_actual_calls": selected_point["primary_actual_calls"],
                "source_balanced_utility": selected_point["policies"]["decar"][
                    "source_balanced"
                ]["utility"],
                "source_balanced_induced_harm": selected_point["policies"]["decar"][
                    "source_balanced"
                ]["induced_harm"],
            }
        ),
        "validation_or_test_inputs_used": False,
    }


def evaluate_entropy_where_hybrid(
    records: Sequence[ActionRecord],
    prediction_rows: Sequence[Mapping[str, Any]],
    formal_evaluation: Mapping[str, Any],
    *,
    bootstrap_indices: Any,
    expected_decisions: int | None = 23_946,
    expected_sources: int | None = 2_204,
    expected_action_disagreements: Mapping[str, int] | None = None,
    expected_bootstrap_resamples: int = DECAR_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Evaluate the frozen entropy-when, source-OOF-action-where diagnostic."""

    outcomes = build_decar_outcomes(
        records,
        expected_decisions=expected_decisions,
        expected_sources=expected_sources,
    )
    predictions = parse_decar_predictions(prediction_rows, outcomes)
    keys = sorted(outcomes)
    sources = sorted({row.source_id for row in outcomes.values()})
    population = formal_evaluation.get("population")
    formal_bootstrap = formal_evaluation.get("bootstrap")
    formal_points = formal_evaluation.get("operating_points")
    if (
        formal_evaluation.get("schema") != DECAR_EVALUATION_SCHEMA
        or formal_evaluation.get("decision") != "decar_not_advanced"
        or formal_evaluation.get("selected_operating_point") is not None
        or formal_evaluation.get("validation_or_test_inputs_used") is not False
        or not isinstance(population, Mapping)
        or population.get("decisions") != len(outcomes)
        or population.get("sources") != len(sources)
        or not isinstance(formal_bootstrap, Mapping)
        or formal_bootstrap.get("n_resamples") != expected_bootstrap_resamples
        or formal_bootstrap.get("n_sources") != len(sources)
        or formal_bootstrap.get("same_indices_for_all_policies_and_differences")
        is not True
        or tuple(formal_evaluation.get("registered_call_rates", ())) != DECAR_CALL_RATES
        or not isinstance(formal_points, list)
        or len(formal_points) != len(DECAR_CALL_RATES)
    ):
        raise ValueError("DECAR hybrid formal-evaluation contract failed")
    if tuple(bootstrap_indices.shape) != (expected_bootstrap_resamples, len(sources)):
        raise ValueError("DECAR hybrid bootstrap shape changed")
    if str(bootstrap_indices.dtype) != "int32":
        raise ValueError("DECAR hybrid bootstrap dtype changed")

    entropy_scores = {key: outcomes[key].baseline.entropy_before for key in keys}
    variants = {
        name: {key: predictions[key].variants[name] for key in keys}
        for name in DECAR_VARIANTS
    }
    actions = {
        name: {key: value.action_id for key, value in values.items()}
        for name, values in variants.items()
    }
    disagreements = {
        name: sum(actions["decar"][key] != actions[name][key] for key in keys)
        for name in ("loss_only", "no_harm_head", "task_value_only")
    }
    if expected_action_disagreements is not None and disagreements != dict(
        expected_action_disagreements
    ):
        raise ValueError("DECAR hybrid OOF action-family audit changed")

    policy_values: dict[str, dict[DecisionKey, dict[str, float]]] = {}
    operating: list[dict[str, Any]] = []
    answer_now = _policy_metrics(outcomes, called_keys=set())
    for formal_point, rate in zip(formal_points, DECAR_CALL_RATES, strict=True):
        if not isinstance(formal_point, Mapping):
            raise ValueError("DECAR hybrid formal operating point is invalid")
        nominal_calls = math.ceil(rate * len(keys))
        point_name = f"rate-{rate:.3f}"
        if (
            formal_point.get("name") != point_name
            or formal_point.get("nominal_calls") != nominal_calls
            or _finite(
                formal_point.get("nominal_question_call_rate"), "hybrid formal rate"
            )
            != rate
        ):
            raise ValueError("DECAR hybrid formal operating-point family changed")
        formal_selection = formal_point.get("selection_audits")
        if not isinstance(formal_selection, Mapping):
            raise ValueError("DECAR hybrid formal selection audit is missing")

        primary_scores = {
            key: value.score
            for key, value in variants["decar"].items()
            if value.eligible
        }
        original_calls, original_audit = complete_tie_top_keys(
            primary_scores,
            target_calls=min(nominal_calls, len(primary_scores)),
        )
        actual_calls = len(original_calls)
        if actual_calls != formal_point.get(
            "primary_actual_calls"
        ) or original_audit != formal_selection.get("decar"):
            raise ValueError("DECAR hybrid original-policy identity audit failed")
        entropy_calls, entropy_audit = _complete_tie_exact_match(
            entropy_scores,
            target_calls=actual_calls,
        )
        entropy_ug_calls, entropy_ug_audit = _complete_tie_exact_match(
            entropy_scores,
            target_calls=actual_calls // 4,
        )
        if entropy_audit != formal_selection.get(
            "entropy_gate_random_and_fixed"
        ) or entropy_ug_audit != formal_selection.get("entropy_gated_ug"):
            raise ValueError("DECAR hybrid entropy identity audit failed")

        point_values = {
            "entropy_when_decar_where": _policy_metrics(
                outcomes,
                called_keys=entropy_calls,
                action_by_key=actions["decar"],
            ),
            "entropy_when_task_value_where": _policy_metrics(
                outcomes,
                called_keys=entropy_calls,
                action_by_key=actions["task_value_only"],
            ),
            "original_decar": _policy_metrics(
                outcomes,
                called_keys=original_calls,
                action_by_key=actions["decar"],
            ),
            "entropy_random": _policy_metrics(
                outcomes,
                called_keys=entropy_calls,
                random_action=True,
            ),
            "entropy_fixed_ug_grid_00": _policy_metrics(
                outcomes,
                called_keys=entropy_calls,
                action_by_key={key: "ug-grid-00" for key in keys},
            ),
            "entropy_gated_ug": _policy_metrics(
                outcomes,
                called_keys=entropy_ug_calls,
                entropy_action=True,
                executions_per_call=4.0,
            ),
            "answer_now": answer_now,
        }
        for name, values in point_values.items():
            policy_values[f"{point_name}/{name}"] = values
        operating.append(
            {
                "name": point_name,
                "nominal_question_call_rate": rate,
                "nominal_calls": nominal_calls,
                "actual_calls": actual_calls,
                "selection_audits": {
                    "original_decar": original_audit,
                    "entropy_one_crop": entropy_audit,
                    "entropy_four_crop": entropy_ug_audit,
                    "matches_formal_identities": True,
                },
            }
        )

    aggregates = {
        name: _aggregate_policy(values, outcomes, sources)
        for name, values in policy_values.items()
    }
    bootstrap, paired_differences = _bootstrap_all_policies(
        aggregates,
        sources,
        bootstrap_indices,
    )
    public_aggregates = {
        name: {key: value for key, value in aggregate.items() if key != "source_values"}
        for name, aggregate in aggregates.items()
    }
    policy_names = (
        "entropy_when_decar_where",
        "entropy_when_task_value_where",
        "original_decar",
        "answer_now",
        "entropy_random",
        "entropy_fixed_ug_grid_00",
        "entropy_gated_ug",
    )
    comparator_names = tuple(
        name for name in policy_names if name != "entropy_when_decar_where"
    )
    points: list[dict[str, Any]] = []
    for registered in operating:
        point_name = str(registered["name"])
        policies = {
            name: public_aggregates[f"{point_name}/{name}"] for name in policy_names
        }
        policy_bootstrap = {
            name: bootstrap["policies"][f"{point_name}/{name}"] for name in policy_names
        }
        primary = policies["entropy_when_decar_where"]["source_balanced"]
        primary_interval = policy_bootstrap["entropy_when_decar_where"]["additive"][
            "utility"
        ]
        rules = {
            "minimum_calls_and_sources": (
                float(policies["entropy_when_decar_where"]["raw_calls"]) >= 100.0
                and int(policies["entropy_when_decar_where"]["distinct_called_sources"])
                >= 50
            ),
            "source_utility_ci_low_strictly_positive": float(primary_interval["ci_low"])
            > 0.0,
            "strictly_above_every_feasible_non_oracle_comparator": all(
                float(primary["utility"])
                > float(policies[name]["source_balanced"]["utility"])
                for name in comparator_names
                if name != "entropy_when_task_value_where"
            ),
            "strictly_above_task_value_where_ablation": float(primary["utility"])
            > float(
                policies["entropy_when_task_value_where"]["source_balanced"]["utility"]
            ),
            "harm_no_greater_than_one_crop_entropy_baselines": all(
                float(primary[metric])
                <= float(policies[name]["source_balanced"][metric]) + 1e-15
                for metric in ("induced_harm", "negative_utility_call")
                for name in ("entropy_random", "entropy_fixed_ug_grid_00")
            ),
            "all_audits_passed": bool(
                registered["selection_audits"]["matches_formal_identities"]
            )
            and all(
                bool(registered["selection_audits"][name]["matched_call_count"])
                for name in ("entropy_one_crop", "entropy_four_crop")
            ),
        }
        points.append(
            {
                **registered,
                "policies": policies,
                "source_bootstrap": policy_bootstrap,
                "paired_source_utility_differences": paired_differences[point_name],
                "qualification_rules": rules,
                "qualified": all(rules.values()),
                "failure_decomposition": _source_concentration(
                    policy_values[f"{point_name}/entropy_when_decar_where"], outcomes
                ),
            }
        )
    qualified = [row for row in points if row["qualified"]]
    selected = (
        min(
            qualified,
            key=lambda row: (
                -float(
                    row["policies"]["entropy_when_decar_where"]["source_balanced"][
                        "utility"
                    ]
                ),
                float(
                    row["policies"]["entropy_when_decar_where"]["source_balanced"][
                        "induced_harm"
                    ]
                ),
                float(row["nominal_question_call_rate"]),
            ),
        )
        if qualified
        else None
    )
    return {
        "schema": DECAR_HYBRID_EVALUATION_SCHEMA,
        "scientific_status": "post-DECAR-v1 official-train frozen hybrid diagnostic",
        "population": {
            "decisions": len(outcomes),
            "sources": len(sources),
            "images": len({row.image_id for row in outcomes.values()}),
        },
        "lambda_cost": DECAR_LAMBDA_COST,
        "registered_call_rates": list(DECAR_CALL_RATES),
        "action_family_audit": {
            "disagreements_from_decar": disagreements,
            "passed": expected_action_disagreements is None
            or disagreements == dict(expected_action_disagreements),
        },
        "operating_points": points,
        "bootstrap": bootstrap["metadata"],
        "decision": (
            "hybrid_train_supported"
            if selected is not None
            else "hybrid_train_not_supported"
        ),
        "selected_operating_point": (
            None
            if selected is None
            else {
                "name": selected["name"],
                "nominal_question_call_rate": selected["nominal_question_call_rate"],
                "actual_calls": selected["actual_calls"],
                "source_balanced_utility": selected["policies"][
                    "entropy_when_decar_where"
                ]["source_balanced"]["utility"],
                "source_balanced_induced_harm": selected["policies"][
                    "entropy_when_decar_where"
                ]["source_balanced"]["induced_harm"],
            }
        ),
        "validation_or_test_inputs_used": False,
    }


def evaluate_entropy_oracle_where_factorization(
    records: Sequence[ActionRecord],
    prediction_rows: Sequence[Mapping[str, Any]],
    hybrid_evaluation: Mapping[str, Any],
    *,
    bootstrap_indices: Any,
    expected_decisions: int | None = 23_946,
    expected_sources: int | None = 2_204,
    expected_action_disagreements: Mapping[str, int] | None = None,
    expected_bootstrap_resamples: int = DECAR_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Factor entropy-when from crop selection with an outcome-oracle where."""

    outcomes = build_decar_outcomes(
        records,
        expected_decisions=expected_decisions,
        expected_sources=expected_sources,
    )
    predictions = parse_decar_predictions(prediction_rows, outcomes)
    keys = sorted(outcomes)
    sources = sorted({row.source_id for row in outcomes.values()})
    hybrid_population = hybrid_evaluation.get("population")
    hybrid_bootstrap = hybrid_evaluation.get("bootstrap")
    hybrid_points = hybrid_evaluation.get("operating_points")
    if (
        hybrid_evaluation.get("schema") != DECAR_HYBRID_EVALUATION_SCHEMA
        or hybrid_evaluation.get("decision") != "hybrid_train_not_supported"
        or hybrid_evaluation.get("selected_operating_point") is not None
        or hybrid_evaluation.get("validation_or_test_inputs_used") is not False
        or not isinstance(hybrid_population, Mapping)
        or hybrid_population.get("decisions") != len(outcomes)
        or hybrid_population.get("sources") != len(sources)
        or not isinstance(hybrid_bootstrap, Mapping)
        or hybrid_bootstrap.get("n_resamples") != expected_bootstrap_resamples
        or hybrid_bootstrap.get("n_sources") != len(sources)
        or hybrid_bootstrap.get("same_indices_for_all_policies_and_differences")
        is not True
        or tuple(hybrid_evaluation.get("registered_call_rates", ())) != DECAR_CALL_RATES
        or not isinstance(hybrid_points, list)
        or len(hybrid_points) != len(DECAR_CALL_RATES)
    ):
        raise ValueError("oracle-where frozen hybrid contract failed")
    if tuple(bootstrap_indices.shape) != (expected_bootstrap_resamples, len(sources)):
        raise ValueError("oracle-where bootstrap shape changed")
    if str(bootstrap_indices.dtype) != "int32":
        raise ValueError("oracle-where bootstrap dtype changed")

    entropy_scores = {key: outcomes[key].baseline.entropy_before for key in keys}
    variants = {
        name: {key: predictions[key].variants[name] for key in keys}
        for name in DECAR_VARIANTS
    }
    actions = {
        name: {key: value.action_id for key, value in values.items()}
        for name, values in variants.items()
    }
    disagreements = {
        name: sum(actions["decar"][key] != actions[name][key] for key in keys)
        for name in ("loss_only", "no_harm_head", "task_value_only")
    }
    if expected_action_disagreements is not None and disagreements != dict(
        expected_action_disagreements
    ):
        raise ValueError("oracle-where OOF action-family audit changed")

    policy_values: dict[str, dict[DecisionKey, dict[str, float]]] = {}
    operating: list[dict[str, Any]] = []
    answer_now = _policy_metrics(outcomes, called_keys=set())
    for hybrid_point, rate in zip(hybrid_points, DECAR_CALL_RATES, strict=True):
        if not isinstance(hybrid_point, Mapping):
            raise ValueError("oracle-where hybrid operating point is invalid")
        point_name = f"rate-{rate:.3f}"
        actual_calls = int(hybrid_point.get("actual_calls", -1))
        selection_audits = hybrid_point.get("selection_audits")
        if (
            hybrid_point.get("name") != point_name
            or _finite(hybrid_point.get("nominal_question_call_rate"), "oracle rate")
            != rate
            or actual_calls <= 0
            or not isinstance(selection_audits, Mapping)
        ):
            raise ValueError("oracle-where hybrid operating-point family changed")
        entropy_calls, entropy_audit = _complete_tie_exact_match(
            entropy_scores,
            target_calls=actual_calls,
        )
        if len(entropy_calls) != actual_calls or entropy_audit != selection_audits.get(
            "entropy_one_crop"
        ):
            raise ValueError("oracle-where entropy identity audit failed")

        point_values = {
            "entropy_when_task_oracle_where": _policy_metrics(
                outcomes,
                called_keys=entropy_calls,
                task_action=True,
            ),
            "entropy_when_decar_where": _policy_metrics(
                outcomes,
                called_keys=entropy_calls,
                action_by_key=actions["decar"],
            ),
            "entropy_when_task_value_where": _policy_metrics(
                outcomes,
                called_keys=entropy_calls,
                action_by_key=actions["task_value_only"],
            ),
            "entropy_random": _policy_metrics(
                outcomes,
                called_keys=entropy_calls,
                random_action=True,
            ),
            "entropy_fixed_ug_grid_00": _policy_metrics(
                outcomes,
                called_keys=entropy_calls,
                action_by_key={key: "ug-grid-00" for key in keys},
            ),
            "answer_now": answer_now,
        }
        oracle_values = point_values["entropy_when_task_oracle_where"]
        learned_values = point_values["entropy_when_decar_where"]
        per_state_consistent = all(
            math.isclose(
                oracle_values[key]["utility"] - learned_values[key]["utility"],
                learned_values[key]["action_selection_regret"],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for key in keys
        )
        if not per_state_consistent:
            raise ValueError("oracle-where per-state regret identity changed")
        for name, values in point_values.items():
            policy_values[f"{point_name}/{name}"] = values
        operating.append(
            {
                "name": point_name,
                "nominal_question_call_rate": rate,
                "actual_calls": actual_calls,
                "selection_audit": entropy_audit,
                "per_state_regret_identity_passed": per_state_consistent,
            }
        )

    aggregates = {
        name: _aggregate_policy(values, outcomes, sources)
        for name, values in policy_values.items()
    }
    bootstrap, paired_differences = _bootstrap_all_policies(
        aggregates,
        sources,
        bootstrap_indices,
    )
    public_aggregates = {
        name: {key: value for key, value in aggregate.items() if key != "source_values"}
        for name, aggregate in aggregates.items()
    }
    policy_names = (
        "entropy_when_task_oracle_where",
        "entropy_when_decar_where",
        "entropy_when_task_value_where",
        "entropy_random",
        "entropy_fixed_ug_grid_00",
        "answer_now",
    )
    points: list[dict[str, Any]] = []
    for registered, hybrid_point in zip(operating, hybrid_points, strict=True):
        point_name = str(registered["name"])
        policies = {
            name: public_aggregates[f"{point_name}/{name}"] for name in policy_names
        }
        policy_bootstrap = {
            name: bootstrap["policies"][f"{point_name}/{name}"] for name in policy_names
        }
        for learned_name in (
            "entropy_when_decar_where",
            "entropy_when_task_value_where",
            "entropy_random",
            "entropy_fixed_ug_grid_00",
            "answer_now",
        ):
            if policies[learned_name] != hybrid_point["policies"][learned_name]:
                raise ValueError(
                    f"oracle-where frozen {learned_name} aggregate changed"
                )
        oracle = policies["entropy_when_task_oracle_where"]
        learned = policies["entropy_when_decar_where"]
        arithmetic_consistent = all(
            math.isclose(
                float(oracle[balance][metric]),
                float(learned[balance][metric])
                + float(learned[balance]["action_selection_regret"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for balance in ("question_balanced", "source_balanced")
            for metric in ("utility", "anls_gain")
        ) and all(
            math.isclose(
                float(oracle[balance]["call"]),
                float(learned[balance]["call"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            for balance in ("question_balanced", "source_balanced")
        )
        differences = paired_differences[point_name]
        primary_interval = policy_bootstrap["entropy_when_task_oracle_where"][
            "additive"
        ]["utility"]
        rules = {
            "minimum_calls_and_sources": (
                float(oracle["raw_calls"]) >= 100.0
                and int(oracle["distinct_called_sources"]) >= 50
            ),
            "oracle_source_utility_ci_low_strictly_positive": float(
                primary_interval["ci_low"]
            )
            > 0.0,
            "paired_above_decar_where_ci_low_strictly_positive": float(
                differences["entropy_when_decar_where"]["ci_low"]
            )
            > 0.0,
            "paired_above_task_value_where_ci_low_strictly_positive": float(
                differences["entropy_when_task_value_where"]["ci_low"]
            )
            > 0.0,
            "exact_arithmetic_consistency": arithmetic_consistent,
            "all_audits_passed": bool(
                registered["per_state_regret_identity_passed"]
                and registered["selection_audit"]["matched_call_count"]
            ),
        }
        points.append(
            {
                **registered,
                "policies": policies,
                "source_bootstrap": policy_bootstrap,
                "paired_source_utility_differences": differences,
                "qualification_rules": rules,
                "qualified": all(rules.values()),
            }
        )
    qualified = [row for row in points if row["qualified"]]
    selected = (
        min(
            qualified,
            key=lambda row: (
                -float(
                    row["policies"]["entropy_when_task_oracle_where"][
                        "source_balanced"
                    ]["utility"]
                ),
                float(row["nominal_question_call_rate"]),
            ),
        )
        if qualified
        else None
    )
    return {
        "schema": DECAR_ORACLE_WHERE_EVALUATION_SCHEMA,
        "scientific_status": "post-hybrid official-train outcome-oracle diagnostic",
        "population": {
            "decisions": len(outcomes),
            "sources": len(sources),
            "images": len({row.image_id for row in outcomes.values()}),
        },
        "lambda_cost": DECAR_LAMBDA_COST,
        "registered_call_rates": list(DECAR_CALL_RATES),
        "action_family_audit": {
            "disagreements_from_decar": disagreements,
            "passed": expected_action_disagreements is None
            or disagreements == dict(expected_action_disagreements),
        },
        "operating_points": points,
        "bootstrap": bootstrap["metadata"],
        "decision": (
            "where_bottleneck_supported"
            if selected is not None
            else "where_bottleneck_not_supported"
        ),
        "selected_operating_point": (
            None
            if selected is None
            else {
                "name": selected["name"],
                "nominal_question_call_rate": selected["nominal_question_call_rate"],
                "actual_calls": selected["actual_calls"],
                "source_balanced_oracle_utility": selected["policies"][
                    "entropy_when_task_oracle_where"
                ]["source_balanced"]["utility"],
            }
        ),
        "outcome_oracle_used": True,
        "deployable_method_evidence": False,
        "validation_or_test_inputs_used": False,
    }
