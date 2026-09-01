from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np  # type: ignore[import-not-found]

from .decoupled_loss_gate import (
    INCUMBENT_POOLED_CALL_RATE,
    INCUMBENT_POOLED_GAIN,
    INCUMBENT_POOLED_UTILITY,
    _evaluate,
    match_call_count_threshold,
)
from .external_pairwise_signed_value import _incumbent_index
from .rescue_gate import DecisionKey
from .scaled_action_value import (
    _PreparedDecisions,
    _prepare_decisions,
    _serialize_linear,
    _source_folds,
)
from .schema import ActionRecord


COST_SENSITIVE_SEED = 20260914
COST_SENSITIVE_FOLDS = 5
COST_SENSITIVE_C = 0.01
COST_SENSITIVE_MAX_ITER = 2000
COST_SENSITIVE_TARGET_CALLS = 225
COST_SENSITIVE_BOOTSTRAP_RESAMPLES = 20_000
COST_SENSITIVE_LAMBDA_COST = 0.05
COST_SENSITIVE_FEATURE_MODE = "semantic-context"
COST_SENSITIVE_ACTION_FEATURE_COUNT = 46
COST_SENSITIVE_DECISIONS = 13_580
COST_SENSITIVE_SOURCES = 3_500
COST_SENSITIVE_ACTION_ROWS = 54_320
COST_SENSITIVE_POSITIVE_UTILITY_ROWS = 1_442
COST_SENSITIVE_NEGATIVE_UTILITY_ROWS = 52_878
COST_SENSITIVE_POSITIVE_GAIN_ROWS = 1_604
COST_SENSITIVE_NEGATIVE_GAIN_ROWS = 1_535
COST_SENSITIVE_NEUTRAL_GAIN_ROWS = 51_181


_FORBIDDEN_OUTPUT_FIELDS = {
    "correct_before",
    "correct_after",
    "target",
    "reward",
    "gain",
    "harm",
    "answer_before",
    "answer_after",
    "oracle_action_id",
    "entropy_after",
    "delta_success",
    "utility",
}


def _rename_candidate(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key).replace(
                "decoupled", "cost_sensitive_direct_action_value"
            ): _rename_candidate(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rename_candidate(item) for item in value]
    return value


def _source_utility_weights(
    utilities: Sequence[float], source_ids: Sequence[str]
) -> np.ndarray:
    if not utilities or len(utilities) != len(source_ids):
        raise ValueError("cost-sensitive weights require aligned non-empty rows")
    values = np.asarray(utilities, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values == 0.0):
        raise ValueError("cost-sensitive utilities must be finite and nonzero")
    totals: dict[str, float] = {}
    for source_id, utility in zip(source_ids, values.tolist()):
        totals[source_id] = totals.get(source_id, 0.0) + abs(float(utility))
    if not totals or any(total <= 0.0 for total in totals.values()):
        raise ValueError("cost-sensitive source utility mass is invalid")
    weights = np.asarray(
        [
            abs(float(utility)) / totals[source_id]
            for source_id, utility in zip(source_ids, values.tolist())
        ],
        dtype=np.float64,
    )
    weights *= len(weights) / float(weights.sum())
    if not math.isclose(
        float(weights.sum()), float(len(weights)), rel_tol=0.0, abs_tol=1e-8
    ):
        raise RuntimeError("cost-sensitive global weight mass differs")
    return weights


def _training_rows(
    prepared: _PreparedDecisions, keys: Sequence[DecisionKey]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[float]]:
    features: list[tuple[float, ...]] = []
    labels: list[int] = []
    utilities: list[float] = []
    source_ids: list[str] = []
    for key in keys:
        for action in prepared.zooms[key]:
            utility = float(action.voi(COST_SENSITIVE_LAMBDA_COST))
            features.append(prepared.action_features[(key, action.action_id)])
            labels.append(int(utility > 0.0))
            utilities.append(utility)
            source_ids.append(prepared.baselines[key].source_id)
    array = np.asarray(features, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    if (
        array.ndim != 2
        or array.shape[1] == 0
        or not np.isfinite(array).all()
        or set(label_array.tolist()) != {0, 1}
    ):
        raise ValueError("cost-sensitive training rows are invalid")
    weights = _source_utility_weights(utilities, source_ids)
    return array, label_array, weights, source_ids, utilities


def _fit_head(
    prepared: _PreparedDecisions,
    keys: Sequence[DecisionKey],
    *,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

    features, labels, weights, source_ids, utilities = _training_rows(
        prepared, keys
    )
    scaler = StandardScaler().fit(features)
    model = LogisticRegression(
        C=COST_SENSITIVE_C,
        penalty="l2",
        solver="liblinear",
        max_iter=COST_SENSITIVE_MAX_ITER,
        random_state=seed,
    ).fit(scaler.transform(features), labels, sample_weight=weights)
    if int(model.n_iter_[0]) >= COST_SENSITIVE_MAX_ITER:
        raise RuntimeError("cost-sensitive direct action-value head did not converge")
    source_mass: dict[str, float] = {}
    for source_id, weight in zip(source_ids, weights.tolist()):
        source_mass[source_id] = source_mass.get(source_id, 0.0) + float(weight)
    expected_source_mass = len(weights) / len(source_mass)
    if any(
        not math.isclose(
            mass, expected_source_mass, rel_tol=0.0, abs_tol=1e-8
        )
        for mass in source_mass.values()
    ):
        raise RuntimeError("cost-sensitive source weight masses differ")
    return (
        {"scaler": scaler, "model": model},
        {
            "train_decisions": len(keys),
            "train_rows": int(features.shape[0]),
            "feature_count": int(features.shape[1]),
            "positive_utility_rows": int(np.sum(labels == 1)),
            "negative_utility_rows": int(np.sum(labels == 0)),
            "utility_min": float(min(utilities)),
            "utility_max": float(max(utilities)),
            "weight_mass": float(weights.sum()),
            "expected_weight_mass": int(features.shape[0]),
            "source_count": len(source_mass),
            "source_mass_min": float(min(source_mass.values())),
            "source_mass_max": float(max(source_mass.values())),
            "expected_source_mass": float(expected_source_mass),
            "iterations": int(model.n_iter_[0]),
            "class_weight": None,
            "weighting": "equal_source_then_absolute_net_utility",
        },
    )


def _score_head(
    prepared: _PreparedDecisions,
    keys: Sequence[DecisionKey],
    *,
    head: Mapping[str, Any],
) -> tuple[dict[DecisionKey, str], dict[DecisionKey, float]]:
    actions: dict[DecisionKey, str] = {}
    scores: dict[DecisionKey, float] = {}
    scaler = head["scaler"]
    model = head["model"]
    for key in keys:
        candidates = prepared.zooms[key]
        features = np.asarray(
            [
                prepared.action_features[(key, action.action_id)]
                for action in candidates
            ],
            dtype=np.float64,
        )
        values = np.asarray(
            model.decision_function(scaler.transform(features)), dtype=np.float64
        )
        if values.shape != (len(candidates),) or not np.isfinite(values).all():
            raise RuntimeError("cost-sensitive direct action scores are invalid")
        selected_index = min(
            range(len(candidates)),
            key=lambda index: (-float(values[index]), candidates[index].action_id),
        )
        actions[key] = candidates[selected_index].action_id
        scores[key] = float(values[selected_index])
    return actions, scores


def _fit_oof(
    prepared: _PreparedDecisions,
    *,
    n_folds: int,
    seed: int,
) -> tuple[
    dict[DecisionKey, str],
    dict[DecisionKey, float],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[DecisionKey, int],
]:
    fold_by_key = _source_folds(
        prepared.keys, prepared.baselines, n_folds=n_folds, seed=seed
    )
    actions: dict[DecisionKey, str] = {}
    scores: dict[DecisionKey, float] = {}
    serialized_folds: list[dict[str, Any]] = []
    fold_audits: list[dict[str, Any]] = []
    for fold in range(n_folds):
        train_keys = [key for key in prepared.keys if fold_by_key[key] != fold]
        test_keys = [key for key in prepared.keys if fold_by_key[key] == fold]
        train_sources = {prepared.baselines[key].source_id for key in train_keys}
        test_sources = {prepared.baselines[key].source_id for key in test_keys}
        overlap = train_sources & test_sources
        if overlap:
            raise RuntimeError("cost-sensitive folds leak sources")
        head, training_audit = _fit_head(
            prepared, train_keys, seed=seed + fold
        )
        fold_actions, fold_scores = _score_head(
            prepared, test_keys, head=head
        )
        actions.update(fold_actions)
        scores.update(fold_scores)
        fold_audit = {
            "fold": fold,
            "test_decisions": len(test_keys),
            "test_sources": len(test_sources),
            "source_overlap": len(overlap),
            "source_exclusion_passed": True,
            **training_audit,
        }
        fold_audits.append(fold_audit)
        serialized_folds.append(
            {
                "fold": fold,
                "head": _serialize_linear(head["scaler"], head["model"]),
                "training_audit": fold_audit,
            }
        )
    if set(actions) != set(prepared.keys) or set(scores) != set(prepared.keys):
        raise RuntimeError("cost-sensitive OOF scores are incomplete")
    return actions, scores, serialized_folds, fold_audits, fold_by_key


def evaluate_cost_sensitive_direct_action_value(
    records: Sequence[ActionRecord],
    audited_score_rows: Sequence[Mapping[str, Any]],
    *,
    semantic_decisions: Mapping[DecisionKey, Mapping[str, Any]],
    bound_inputs_verified: bool,
    n_folds: int = COST_SENSITIVE_FOLDS,
    bootstrap_resamples: int = COST_SENSITIVE_BOOTSTRAP_RESAMPLES,
    seed: int = COST_SENSITIVE_SEED,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if not bound_inputs_verified:
        raise ValueError("cost-sensitive fitting requires bound inputs")
    if n_folds != COST_SENSITIVE_FOLDS or seed != COST_SENSITIVE_SEED:
        raise ValueError("cost-sensitive folds and seed are frozen")
    if bootstrap_resamples != COST_SENSITIVE_BOOTSTRAP_RESAMPLES:
        raise ValueError("cost-sensitive bootstrap count is frozen")
    prepared = _prepare_decisions(
        records,
        feature_mode=COST_SENSITIVE_FEATURE_MODE,
        semantic_decisions=semantic_decisions,
    )
    utilities = [
        float(action.voi(COST_SENSITIVE_LAMBDA_COST))
        for key in prepared.keys
        for action in prepared.zooms[key]
    ]
    gains = [
        float(action.delta_success)
        for key in prepared.keys
        for action in prepared.zooms[key]
    ]
    first_key = prepared.keys[0]
    action_feature_count = len(
        prepared.action_features[(first_key, prepared.zooms[first_key][0].action_id)]
    )
    if (
        len(prepared.keys) != COST_SENSITIVE_DECISIONS
        or len({prepared.baselines[key].source_id for key in prepared.keys})
        != COST_SENSITIVE_SOURCES
        or len(utilities) != COST_SENSITIVE_ACTION_ROWS
        or any(len(prepared.zooms[key]) != 4 for key in prepared.keys)
        or action_feature_count != COST_SENSITIVE_ACTION_FEATURE_COUNT
        or sum(value > 0.0 for value in utilities)
        != COST_SENSITIVE_POSITIVE_UTILITY_ROWS
        or sum(value < 0.0 for value in utilities)
        != COST_SENSITIVE_NEGATIVE_UTILITY_ROWS
        or any(value == 0.0 for value in utilities)
        or sum(value > 0.0 for value in gains) != COST_SENSITIVE_POSITIVE_GAIN_ROWS
        or sum(value < 0.0 for value in gains) != COST_SENSITIVE_NEGATIVE_GAIN_ROWS
        or sum(value == 0.0 for value in gains) != COST_SENSITIVE_NEUTRAL_GAIN_ROWS
    ):
        raise ValueError("cost-sensitive population or utility contract changed")
    candidate_actions, candidate_scores, serialized_folds, fold_audits, fold_by_key = _fit_oof(
        prepared, n_folds=n_folds, seed=seed
    )
    incumbent_actions, incumbent_scores, audited_incumbent_calls = _incumbent_index(
        audited_score_rows, baselines=prepared.baselines
    )
    incumbent_match = match_call_count_threshold(
        incumbent_scores, target_calls=COST_SENSITIVE_TARGET_CALLS
    )
    candidate_match = match_call_count_threshold(
        candidate_scores, target_calls=COST_SENSITIVE_TARGET_CALLS
    )
    incumbent_call_keys = {
        key
        for key, score in incumbent_scores.items()
        if score >= float(incumbent_match["threshold"])
    }
    audited_call_keys = {
        key for key, called in audited_incumbent_calls.items() if called
    }
    if (
        incumbent_match["calls"] != COST_SENSITIVE_TARGET_CALLS
        or candidate_match["calls"] != COST_SENSITIVE_TARGET_CALLS
        or incumbent_call_keys != audited_call_keys
    ):
        raise ValueError("cost-sensitive matched-call contract changed")
    evaluated = _rename_candidate(
        _evaluate(
            baselines=prepared.baselines,
            zooms=prepared.zooms,
            actions_by_method={
                "incumbent": incumbent_actions,
                "decoupled": candidate_actions,
            },
            scores_by_method={
                "incumbent": incumbent_scores,
                "decoupled": candidate_scores,
            },
            threshold_by_method={
                "incumbent": float(incumbent_match["threshold"]),
                "decoupled": float(candidate_match["threshold"]),
            },
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=seed,
        )
    )
    incumbent_question = evaluated["question_balanced"]["incumbent"]
    if not (
        math.isclose(float(incumbent_question["gain"]), INCUMBENT_POOLED_GAIN, rel_tol=0.0, abs_tol=1e-15)
        and math.isclose(float(incumbent_question["utility"]), INCUMBENT_POOLED_UTILITY, rel_tol=0.0, abs_tol=1e-15)
        and math.isclose(float(incumbent_question["call"]), INCUMBENT_POOLED_CALL_RATE, rel_tol=0.0, abs_tol=1e-15)
    ):
        raise ValueError("cost-sensitive incumbent metrics do not reproduce")
    full_head, full_audit = _fit_head(prepared, prepared.keys, seed=seed)
    source_points = evaluated["source_balanced"]
    incumbent = source_points["incumbent"]
    candidate = source_points["cost_sensitive_direct_action_value"]
    primary = evaluated["primary_estimand"]
    weight_audits_pass = all(
        math.isclose(float(fold["weight_mass"]), float(fold["expected_weight_mass"]), rel_tol=0.0, abs_tol=1e-8)
        and math.isclose(float(fold["source_mass_min"]), float(fold["expected_source_mass"]), rel_tol=0.0, abs_tol=1e-8)
        and math.isclose(float(fold["source_mass_max"]), float(fold["expected_source_mass"]), rel_tol=0.0, abs_tol=1e-8)
        for fold in [*fold_audits, full_audit]
    )
    audits = {
        "bound_input_hashes_verified": True,
        "population_and_utility_counts_exact": True,
        "semantic_action_alignment_exact": True,
        "action_feature_count_exact": action_feature_count == COST_SENSITIVE_ACTION_FEATURE_COUNT,
        "source_exclusion": all(fold["source_exclusion_passed"] for fold in fold_audits),
        "source_and_global_weight_mass_exact": weight_audits_pass,
        "class_weight_disabled": all(fold["class_weight"] is None for fold in [*fold_audits, full_audit]),
        "all_heads_converged": all(int(fold["iterations"]) < COST_SENSITIVE_MAX_ITER for fold in [*fold_audits, full_audit]),
        "oof_score_coverage_exact": set(candidate_scores) == set(prepared.keys),
        "matched_call_counts_exact": incumbent_match["calls"] == candidate_match["calls"] == COST_SENSITIVE_TARGET_CALLS,
        "incumbent_call_set_reproduced": incumbent_call_keys == audited_call_keys,
        "incumbent_pooled_metrics_reproduced": True,
        "serialized_scores_finite": all(math.isfinite(value) for value in candidate_scores.values()),
        "serialized_scores_outcome_free": True,
        "screenqa_inputs_used": False,
        "protected_role_inputs_used": False,
    }
    all_audits_passed = all(
        value is True
        for key, value in audits.items()
        if key not in {"screenqa_inputs_used", "protected_role_inputs_used"}
    ) and not audits["screenqa_inputs_used"] and not audits["protected_role_inputs_used"]
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
    threshold = float(candidate_match["threshold"])
    score_rows: list[dict[str, Any]] = []
    for key in prepared.keys:
        row = {
            "state_id": key[0],
            "replicate_id": key[1],
            "source_id": prepared.baselines[key].source_id,
            "outer_fold": int(fold_by_key[key]),
            "cost_sensitive_direct_action_value_action_id": candidate_actions[key],
            "cost_sensitive_direct_action_value_score": float(candidate_scores[key]),
            "cost_sensitive_direct_action_value_called": bool(candidate_scores[key] >= threshold),
            "incumbent_action_id": incumbent_actions[key],
            "incumbent_score": float(incumbent_scores[key]),
            "incumbent_called": bool(audited_incumbent_calls[key]),
        }
        if _FORBIDDEN_OUTPUT_FIELDS.intersection(row):
            raise RuntimeError("cost-sensitive serialized rows leak outcomes")
        score_rows.append(row)
    score_report = {
        "scientific_status": "outcome-free cost-sensitive direct action-value OOF scores frozen before opened-development outcome evaluation",
        "n_sources": COST_SENSITIVE_SOURCES,
        "n_decisions": COST_SENSITIVE_DECISIONS,
        "n_action_rows": COST_SENSITIVE_ACTION_ROWS,
        "feature_mode": COST_SENSITIVE_FEATURE_MODE,
        "action_feature_count": action_feature_count,
        "C": COST_SENSITIVE_C,
        "fold_training": fold_audits,
        "incumbent_match": incumbent_match,
        "cost_sensitive_direct_action_value_match": candidate_match,
        "task_outcomes_used_for_thresholds": False,
        "serialized_outcome_fields": [],
        "audits": audits,
    }
    report = {
        "scientific_status": "opened DocVQA development diagnostic; not independent validation",
        "decision": "cost_sensitive_direct_action_value_advanced" if all(pass_rule.values()) else "cost_sensitive_direct_action_value_not_advanced",
        "pass_rule": pass_rule,
        "n_sources": COST_SENSITIVE_SOURCES,
        "n_decisions": COST_SENSITIVE_DECISIONS,
        "lambda_cost": COST_SENSITIVE_LAMBDA_COST,
        "source_balanced": source_points,
        "question_balanced": evaluated["question_balanced"],
        "primary_estimand": primary,
        "paired_source_bootstrap": evaluated["paired_source_bootstrap"],
        "action_disagreement_rate": evaluated["action_disagreement_rate"],
        "gate_disagreement_rate": evaluated["gate_disagreement_rate"],
        "audits": audits,
        "screenqa_inputs_used": False,
        "protected_role_inputs_used": False,
    }
    model = {
        "schema": "cost_sensitive_direct_action_value_v1",
        "feature_mode": COST_SENSITIVE_FEATURE_MODE,
        "action_feature_count": action_feature_count,
        "seed": seed,
        "n_folds": n_folds,
        "lambda_cost": COST_SENSITIVE_LAMBDA_COST,
        "C": COST_SENSITIVE_C,
        "weighting": "equal_source_then_absolute_net_utility",
        "oof_folds": serialized_folds,
        "full_refit": {
            "head": _serialize_linear(full_head["scaler"], full_head["model"]),
            "training_audit": full_audit,
        },
        "screenqa_inputs_used": False,
    }
    return report, score_report, model, score_rows
