from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any

import numpy as np  # type: ignore[import-not-found]

from .action_value import _decision_rows
from .decoupled_loss_gate import (
    INCUMBENT_POOLED_CALL_RATE,
    INCUMBENT_POOLED_GAIN,
    INCUMBENT_POOLED_UTILITY,
    _evaluate,
    match_call_count_threshold,
)
from .rescue_gate import DecisionKey
from .schema import ActionRecord


CONSENSUS_BOOTSTRAP_RESAMPLES = 20_000
CONSENSUS_BOOTSTRAP_SEED = 20260916
CONSENSUS_TARGET_CALLS = 225
CONSENSUS_DECISIONS = 13_580
CONSENSUS_SOURCES = 3_500

_COST_FIELDS = {
    "state_id",
    "replicate_id",
    "source_id",
    "outer_fold",
    "cost_sensitive_direct_action_value_action_id",
    "cost_sensitive_direct_action_value_score",
    "cost_sensitive_direct_action_value_called",
    "incumbent_action_id",
    "incumbent_score",
    "incumbent_called",
}
_INCUMBENT_FIELDS = {
    "state_id",
    "replicate_id",
    "source_id",
    "decoupled_action_id",
    "decoupled_score",
    "decoupled_called",
    "incumbent_action_id",
    "incumbent_score",
    "incumbent_called",
}
_SERIALIZED_FIELDS = {
    "state_id",
    "replicate_id",
    "source_id",
    "cost_sensitive_direct_action_value_action_id",
    "cost_sensitive_direct_action_value_score",
    "cost_sensitive_direct_action_value_percentile",
    "incumbent_action_id",
    "incumbent_score",
    "incumbent_percentile",
    "incumbent_called",
    "minimum_rank_consensus_gate_action_id",
    "minimum_rank_consensus_gate_score",
    "minimum_rank_consensus_gate_called",
}


def _rename_candidate(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key).replace(
                "decoupled", "minimum_rank_consensus_gate"
            ): _rename_candidate(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rename_candidate(item) for item in value]
    return value


def _empirical_upper_percentiles(
    scores: Mapping[DecisionKey, float],
) -> tuple[dict[DecisionKey, float], dict[str, list[float]]]:
    """Return the frozen ECDF rank count(score_i <= score) / N.

    Equal raw values deliberately receive the same percentile.  The lookup is
    serialized in ascending score order so the transformation can be replayed.
    """

    if not scores:
        raise ValueError("consensus percentile scores must be non-empty")
    normalized = {key: float(value) for key, value in scores.items()}
    if any(not math.isfinite(value) for value in normalized.values()):
        raise ValueError("consensus percentile scores must be finite")
    counts: dict[float, int] = {}
    for value in normalized.values():
        counts[value] = counts.get(value, 0) + 1
    cumulative = 0
    rank_by_value: dict[float, float] = {}
    raw_values: list[float] = []
    percentiles: list[float] = []
    for value in sorted(counts):
        cumulative += counts[value]
        percentile = cumulative / len(normalized)
        rank_by_value[value] = percentile
        raw_values.append(value)
        percentiles.append(percentile)
    ranks = {key: rank_by_value[value] for key, value in normalized.items()}
    if (
        not all(0.0 < value <= 1.0 for value in ranks.values())
        or percentiles[-1] != 1.0
        or any(left >= right for left, right in zip(raw_values, raw_values[1:]))
        or any(left >= right for left, right in zip(percentiles, percentiles[1:]))
    ):
        raise RuntimeError("consensus percentile construction failed")
    return ranks, {"raw_scores": raw_values, "percentiles": percentiles}


def _score_index(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_fields: set[str],
    action_field: str,
    score_field: str,
    called_field: str,
    name: str,
) -> tuple[
    dict[DecisionKey, str],
    dict[DecisionKey, float],
    dict[DecisionKey, bool],
    dict[DecisionKey, str],
]:
    actions: dict[DecisionKey, str] = {}
    scores: dict[DecisionKey, float] = {}
    calls: dict[DecisionKey, bool] = {}
    sources: dict[DecisionKey, str] = {}
    for row in rows:
        if set(row) != expected_fields:
            raise ValueError(f"{name} score-row schema changed")
        key = (str(row["state_id"]), str(row["replicate_id"]))
        source_id = str(row["source_id"])
        action_id = str(row[action_field])
        score = float(row[score_field])
        called = row[called_field]
        if (
            not all(key)
            or not source_id
            or not action_id
            or not math.isfinite(score)
            or not isinstance(called, bool)
            or key in actions
        ):
            raise ValueError(f"{name} score row is invalid")
        actions[key] = action_id
        scores[key] = score
        calls[key] = called
        sources[key] = source_id
    if not set(actions) == set(scores) == set(calls) == set(sources):
        raise RuntimeError(f"{name} score indexes differ")
    return actions, scores, calls, sources


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if (
        a.shape != b.shape
        or a.ndim != 1
        or a.size < 2
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or float(np.std(a)) == 0.0
        or float(np.std(b)) == 0.0
    ):
        raise ValueError("consensus correlation inputs are invalid")
    value = float(np.corrcoef(a, b)[0, 1])
    if not math.isfinite(value):
        raise RuntimeError("consensus correlation is non-finite")
    return value


def evaluate_minimum_rank_consensus_gate(
    records: Sequence[ActionRecord],
    cost_rows: Sequence[Mapping[str, Any]],
    incumbent_rows: Sequence[Mapping[str, Any]],
    *,
    bound_inputs_verified: bool,
    bootstrap_resamples: int = CONSENSUS_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = CONSENSUS_BOOTSTRAP_SEED,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if not bound_inputs_verified:
        raise ValueError("consensus evaluation requires bound inputs")
    if (
        bootstrap_resamples != CONSENSUS_BOOTSTRAP_RESAMPLES
        or bootstrap_seed != CONSENSUS_BOOTSTRAP_SEED
    ):
        raise ValueError("consensus bootstrap settings are frozen")

    baselines, zoom_lists = _decision_rows(records)
    zooms = {key: tuple(values) for key, values in zoom_lists.items()}
    if (
        len(baselines) != CONSENSUS_DECISIONS
        or len({baseline.source_id for baseline in baselines.values()})
        != CONSENSUS_SOURCES
        or any(len(values) != 4 for values in zooms.values())
    ):
        raise ValueError("consensus population contract changed")

    cost_actions, cost_scores, cost_calls, cost_sources = _score_index(
        cost_rows,
        expected_fields=_COST_FIELDS,
        action_field="cost_sensitive_direct_action_value_action_id",
        score_field="cost_sensitive_direct_action_value_score",
        called_field="cost_sensitive_direct_action_value_called",
        name="cost-sensitive",
    )
    incumbent_actions, incumbent_scores, incumbent_calls, incumbent_sources = (
        _score_index(
            incumbent_rows,
            expected_fields=_INCUMBENT_FIELDS,
            action_field="incumbent_action_id",
            score_field="incumbent_score",
            called_field="incumbent_called",
            name="incumbent",
        )
    )
    keys = set(baselines)
    if not set(cost_actions) == set(incumbent_actions) == keys:
        raise ValueError("consensus score coverage differs from rollouts")

    embedded_incumbent: dict[DecisionKey, tuple[str, float, bool]] = {}
    for row in cost_rows:
        key = (str(row["state_id"]), str(row["replicate_id"]))
        embedded_incumbent[key] = (
            str(row["incumbent_action_id"]),
            float(row["incumbent_score"]),
            bool(row["incumbent_called"]),
        )
    for key in keys:
        valid_actions = {action.action_id for action in zooms[key]}
        if (
            cost_sources[key] != baselines[key].source_id
            or incumbent_sources[key] != baselines[key].source_id
            or cost_actions[key] not in valid_actions
            or incumbent_actions[key] not in valid_actions
            or embedded_incumbent[key]
            != (
                incumbent_actions[key],
                incumbent_scores[key],
                incumbent_calls[key],
            )
        ):
            raise ValueError("consensus identity/action/incumbent reproduction failed")
    cost_match = match_call_count_threshold(
        cost_scores, target_calls=CONSENSUS_TARGET_CALLS
    )
    cost_threshold = float(cost_match["threshold"])
    threshold_cost_calls = {
        key: score >= cost_threshold for key, score in cost_scores.items()
    }
    if (
        cost_match["calls"] != CONSENSUS_TARGET_CALLS
        or threshold_cost_calls != cost_calls
    ):
        raise ValueError("cost-sensitive frozen call set does not reproduce")

    incumbent_match = match_call_count_threshold(
        incumbent_scores, target_calls=CONSENSUS_TARGET_CALLS
    )
    incumbent_threshold = float(incumbent_match["threshold"])
    threshold_incumbent_calls = {
        key: score >= incumbent_threshold for key, score in incumbent_scores.items()
    }
    if (
        incumbent_match["calls"] != CONSENSUS_TARGET_CALLS
        or threshold_incumbent_calls != incumbent_calls
    ):
        raise ValueError("consensus incumbent call set does not reproduce")

    incumbent_ranks, incumbent_lookup = _empirical_upper_percentiles(
        incumbent_scores
    )
    cost_ranks, cost_lookup = _empirical_upper_percentiles(cost_scores)
    consensus_scores = {
        key: min(incumbent_ranks[key], cost_ranks[key]) for key in keys
    }
    consensus_match = match_call_count_threshold(
        consensus_scores, target_calls=CONSENSUS_TARGET_CALLS
    )
    if consensus_match["calls"] != CONSENSUS_TARGET_CALLS:
        raise ValueError("consensus score has no exact complete-tie 225-call threshold")
    consensus_threshold = float(consensus_match["threshold"])

    evaluated = _rename_candidate(
        _evaluate(
            baselines=baselines,
            zooms=zooms,
            actions_by_method={
                "incumbent": incumbent_actions,
                "decoupled": cost_actions,
            },
            scores_by_method={
                "incumbent": incumbent_scores,
                "decoupled": consensus_scores,
            },
            threshold_by_method={
                "incumbent": incumbent_threshold,
                "decoupled": consensus_threshold,
            },
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        )
    )
    incumbent_question = evaluated["question_balanced"]["incumbent"]
    if not (
        math.isclose(
            float(incumbent_question["gain"]),
            INCUMBENT_POOLED_GAIN,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and math.isclose(
            float(incumbent_question["utility"]),
            INCUMBENT_POOLED_UTILITY,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and math.isclose(
            float(incumbent_question["call"]),
            INCUMBENT_POOLED_CALL_RATE,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise ValueError("consensus incumbent pooled metrics do not reproduce")

    ordered = sorted(keys)
    raw_agreement = _pearson(
        [incumbent_scores[key] for key in ordered],
        [cost_scores[key] for key in ordered],
    )
    percentile_agreement = _pearson(
        [incumbent_ranks[key] for key in ordered],
        [cost_ranks[key] for key in ordered],
    )
    consensus_calls = {
        key: score >= consensus_threshold for key, score in consensus_scores.items()
    }
    call_overlap = {
        "intersection": sum(consensus_calls[key] and incumbent_calls[key] for key in keys),
        "consensus_only": sum(consensus_calls[key] and not incumbent_calls[key] for key in keys),
        "incumbent_only": sum(incumbent_calls[key] and not consensus_calls[key] for key in keys),
        "union": sum(consensus_calls[key] or incumbent_calls[key] for key in keys),
    }

    audits = {
        "bound_input_hashes_verified": True,
        "population_exact": True,
        "identity_and_source_alignment_exact": True,
        "candidate_action_ids_valid_and_retained": True,
        "cost_sensitive_score_coverage_exact": set(cost_scores) == keys,
        "incumbent_score_coverage_exact": set(incumbent_scores) == keys,
        "cost_sensitive_frozen_call_set_reproduced": threshold_cost_calls
        == cost_calls,
        "incumbent_raw_fields_reproduced": True,
        "incumbent_call_set_reproduced": threshold_incumbent_calls
        == incumbent_calls,
        "incumbent_pooled_metrics_reproduced": True,
        "raw_scores_finite": all(
            math.isfinite(value)
            for value in [*incumbent_scores.values(), *cost_scores.values()]
        ),
        "percentiles_finite_and_in_unit_interval": all(
            0.0 < value <= 1.0
            for value in [*incumbent_ranks.values(), *cost_ranks.values()]
        ),
        "percentile_monotonicity_and_ties_preserved": True,
        "minimum_rule_exact": all(
            consensus_scores[key] == min(incumbent_ranks[key], cost_ranks[key])
            for key in keys
        ),
        "matched_call_counts_exact": incumbent_match["calls"]
        == consensus_match["calls"]
        == CONSENSUS_TARGET_CALLS,
        "serialized_scores_outcome_free": True,
        "screenqa_inputs_used": False,
        "protected_role_inputs_used": False,
    }
    all_audits_passed = all(
        value is True
        for name, value in audits.items()
        if name not in {"screenqa_inputs_used", "protected_role_inputs_used"}
    ) and not audits["screenqa_inputs_used"] and not audits["protected_role_inputs_used"]

    source_points = evaluated["source_balanced"]
    incumbent = source_points["incumbent"]
    candidate = source_points["minimum_rank_consensus_gate"]
    primary = evaluated["primary_estimand"]
    pass_rule = {
        "utility_margin_at_least_0_00025": float(candidate["utility"])
        >= float(incumbent["utility"]) + 0.00025,
        "paired_ci_low_above_minus_0_0005": float(primary["ci_low"]) > -0.0005,
        "gain_per_call_higher": float(candidate["gain_per_call"])
        > float(incumbent["gain_per_call"]),
        "harm_and_negative_calls_no_greater": float(candidate["induced_harm"])
        <= float(incumbent["induced_harm"])
        and float(candidate["negative_value_call"])
        <= float(incumbent["negative_value_call"]),
        "helpful_call_precision_no_lower": float(candidate["helpful_call_precision"])
        >= float(incumbent["helpful_call_precision"]),
        "all_audits_passed": all_audits_passed,
    }

    score_rows: list[dict[str, Any]] = []
    for key in ordered:
        row = {
            "state_id": key[0],
            "replicate_id": key[1],
            "source_id": baselines[key].source_id,
            "cost_sensitive_direct_action_value_action_id": cost_actions[key],
            "cost_sensitive_direct_action_value_score": cost_scores[key],
            "cost_sensitive_direct_action_value_percentile": cost_ranks[key],
            "incumbent_action_id": incumbent_actions[key],
            "incumbent_score": incumbent_scores[key],
            "incumbent_percentile": incumbent_ranks[key],
            "incumbent_called": incumbent_calls[key],
            "minimum_rank_consensus_gate_action_id": cost_actions[key],
            "minimum_rank_consensus_gate_score": consensus_scores[key],
            "minimum_rank_consensus_gate_called": consensus_calls[key],
        }
        if set(row) != _SERIALIZED_FIELDS:
            raise RuntimeError("consensus serialized score schema changed")
        score_rows.append(row)

    score_report = {
        "scientific_status": "outcome-free minimum-rank consensus scores frozen before opened-development outcome evaluation",
        "n_sources": CONSENSUS_SOURCES,
        "n_decisions": CONSENSUS_DECISIONS,
        "rank_definition": "count(raw_score_i <= raw_score) / 13580 with ties preserved",
        "consensus_definition": "min(incumbent_percentile, cost_sensitive_percentile)",
        "retained_action": "cost_sensitive_direct_action_value_action_id",
        "incumbent_match": incumbent_match,
        "cost_sensitive_direct_action_value_match": cost_match,
        "minimum_rank_consensus_gate_match": consensus_match,
        "raw_score_pearson": raw_agreement,
        "percentile_score_pearson": percentile_agreement,
        "call_overlap_with_incumbent": call_overlap,
        "task_outcomes_used_for_rank_or_threshold": False,
        "serialized_outcome_fields": [],
        "audits": audits,
    }
    report = {
        "scientific_status": "opened DocVQA development diagnostic; not independent validation",
        "decision": (
            "minimum_rank_consensus_gate_advanced"
            if all(pass_rule.values())
            else "minimum_rank_consensus_gate_not_advanced"
        ),
        "pass_rule": pass_rule,
        "n_sources": CONSENSUS_SOURCES,
        "n_decisions": CONSENSUS_DECISIONS,
        "source_balanced": source_points,
        "question_balanced": evaluated["question_balanced"],
        "primary_estimand": primary,
        "paired_source_bootstrap": evaluated["paired_source_bootstrap"],
        "action_disagreement_rate": evaluated["action_disagreement_rate"],
        "gate_disagreement_rate": evaluated["gate_disagreement_rate"],
        "raw_score_pearson": raw_agreement,
        "percentile_score_pearson": percentile_agreement,
        "call_overlap_with_incumbent": call_overlap,
        "audits": audits,
        "screenqa_inputs_used": False,
        "protected_role_inputs_used": False,
    }
    contract = {
        "schema": "minimum_rank_consensus_gate_v1",
        "fit_performed": False,
        "rank_definition": "count(raw_score_i <= raw_score) / N",
        "combination": "minimum",
        "retained_action": "cost_sensitive_direct_action_value_action_id",
        "target_calls": CONSENSUS_TARGET_CALLS,
        "threshold": consensus_threshold,
        "incumbent_rank_lookup": incumbent_lookup,
        "cost_sensitive_rank_lookup": cost_lookup,
        "screenqa_inputs_used": False,
    }
    return report, score_report, contract, score_rows
