from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any

import numpy as np  # type: ignore[import-not-found]

from .action_value import (
    _action_features,
    _semantic_feature_index,
    _state_features,
    _validate_domains,
)
from .oof_action_value import _fit_heads, _score_heads, _source_folds
from .rescue_gate import DecisionKey
from .scaled_evaluation import bootstrap_source_balanced_metrics
from .schema import ActionRecord


DECOUPLED_SEED = 20260905
DECOUPLED_BOOTSTRAP_RESAMPLES = 20000
DECOUPLED_CONFIDENCE = 0.95
DECOUPLED_TARGET_CALLS = 225
DECOUPLED_LAMBDA_COST = 0.05
INCUMBENT_SEED = 20260829
INCUMBENT_FOLDS = 5
INCUMBENT_ALPHA = 1.0
INCUMBENT_THRESHOLD = -0.0136405068067658
INCUMBENT_POOLED_GAIN = 0.003653651657979981
INCUMBENT_POOLED_UTILITY = 0.00282522750481356
INCUMBENT_POOLED_CALL_RATE = 0.016568483063328424


def match_call_count_threshold(
    scores: Mapping[DecisionKey, float], *, target_calls: int
) -> dict[str, Any]:
    """Choose a complete-tie score threshold without reading any outcome."""

    if not scores or target_calls < 0 or target_calls > len(scores):
        raise ValueError("matched-call scores or target count are invalid")
    normalized = {key: float(value) for key, value in scores.items()}
    if any(not math.isfinite(value) for value in normalized.values()):
        raise ValueError("matched-call scores must be finite")
    candidates: list[dict[str, Any]] = []
    for threshold in sorted(set(normalized.values()), reverse=True):
        calls = sum(value >= threshold for value in normalized.values())
        candidates.append(
            {
                "threshold": threshold,
                "calls": calls,
                "absolute_call_error": abs(calls - target_calls),
                "pooled_call_rate": calls / len(normalized),
            }
        )
    selected = min(
        candidates,
        key=lambda item: (
            item["absolute_call_error"],
            item["calls"],
            -item["threshold"],
        ),
    )
    return {
        **selected,
        "target_calls": target_calls,
        "selection_uses_outcomes": False,
        "ties_preserved": True,
    }


def _forced_factorized_scores(
    heads: Mapping[str, Any],
    keys: Sequence[DecisionKey],
    *,
    forced_actions: Mapping[DecisionKey, str],
    baselines: Mapping[DecisionKey, ActionRecord],
    zooms: Mapping[DecisionKey, Sequence[ActionRecord]],
    feature_mode: str,
    semantic_by_key: Mapping[DecisionKey, Mapping[str, Any]],
) -> dict[DecisionKey, float]:
    result: dict[DecisionKey, float] = {}
    for key in keys:
        state = np.asarray(
            [
                _state_features(
                    baselines[key],
                    feature_mode=feature_mode,
                    semantic_decision=semantic_by_key.get(key),
                )
            ],
            dtype=np.float64,
        )
        error_probability = float(
            heads["error_model"].predict_proba(
                heads["error_scaler"].transform(state)
            )[0, 1]
        )
        matches = [
            action
            for action in zooms[key]
            if action.action_id == forced_actions[key]
        ]
        if len(matches) != 1:
            raise ValueError(f"forced loss action is invalid for {key!r}")
        action = matches[0]
        features = np.asarray(
            [
                _action_features(
                    baselines[key],
                    action,
                    feature_mode=feature_mode,
                    semantic_decision=semantic_by_key.get(key),
                )
            ],
            dtype=np.float64,
        )
        rescue_probability = float(
            heads["rescue_model"].predict_proba(
                heads["rescue_scaler"].transform(features)
            )[0, 1]
        )
        harm_probability = float(
            heads["harm_model"].predict_proba(
                heads["harm_scaler"].transform(features)
            )[0, 1]
        )
        result[key] = (
            error_probability
            * rescue_probability
            * float(heads["rescue_magnitude"])
            - (1.0 - error_probability)
            * harm_probability
            * float(heads["harm_magnitude"])
            - DECOUPLED_LAMBDA_COST * action.tool_cost
        )
    return result


def _prediction_index(
    rows: Sequence[Mapping[str, Any]],
    keys: set[DecisionKey],
) -> tuple[dict[DecisionKey, str], dict[DecisionKey, float]]:
    forbidden = {
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
    }
    actions: dict[DecisionKey, str] = {}
    scores: dict[DecisionKey, float] = {}
    for row in rows:
        if forbidden.intersection(row):
            raise ValueError("loss-proposer predictions contain forbidden outcomes")
        key = (str(row.get("state_id", "")), str(row.get("replicate_id", "")))
        action_id = str(row.get("loss_only_action_id", ""))
        score = float(row.get("loss_only_score", math.nan))
        if not all(key) or not action_id or not math.isfinite(score) or key in actions:
            raise ValueError("loss-proposer prediction row is invalid")
        actions[key] = action_id
        scores[key] = score
    if set(actions) != keys or set(scores) != keys:
        raise ValueError("loss-proposer predictions do not exactly cover decisions")
    return actions, scores


def _source_means(
    values: Mapping[DecisionKey, float], source_by_key: Mapping[DecisionKey, str]
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for key, value in values.items():
        grouped.setdefault(source_by_key[key], []).append(float(value))
    return {source: mean(items) for source, items in grouped.items()}


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0.0 else None


def _evaluate(
    *,
    baselines: Mapping[DecisionKey, ActionRecord],
    zooms: Mapping[DecisionKey, Sequence[ActionRecord]],
    actions_by_method: Mapping[str, Mapping[DecisionKey, str]],
    scores_by_method: Mapping[str, Mapping[DecisionKey, float]],
    threshold_by_method: Mapping[str, float],
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    source_by_key = {key: baseline.source_id for key, baseline in baselines.items()}
    helpful_state = {
        key: float(any(action.delta_success > 0.0 for action in zooms[key]))
        for key in baselines
    }
    values: dict[str, dict[str, dict[DecisionKey, float]]] = {}
    score_rows: list[dict[str, Any]] = []
    for method in ("incumbent", "decoupled"):
        metrics = {
            name: {}
            for name in (
                "gain",
                "utility",
                "call",
                "induced_harm",
                "negative_value_call",
                "helpful_proposal",
                "helpful_call",
            )
        }
        values[method] = metrics
        for key in sorted(baselines):
            matches = [
                action
                for action in zooms[key]
                if action.action_id == actions_by_method[method][key]
            ]
            if len(matches) != 1:
                raise ValueError(f"{method} action is invalid for {key!r}")
            action = matches[0]
            called = scores_by_method[method][key] >= threshold_by_method[method]
            gain = action.delta_success if called else 0.0
            utility = (
                gain - DECOUPLED_LAMBDA_COST * action.tool_cost if called else 0.0
            )
            metrics["gain"][key] = gain
            metrics["utility"][key] = utility
            metrics["call"][key] = float(called)
            metrics["induced_harm"][key] = max(-gain, 0.0)
            metrics["negative_value_call"][key] = float(called and utility < 0.0)
            metrics["helpful_proposal"][key] = float(action.delta_success > 0.0)
            metrics["helpful_call"][key] = float(
                called and action.delta_success > 0.0
            )

    for key in sorted(baselines):
        score_rows.append(
            {
                "state_id": key[0],
                "replicate_id": key[1],
                "source_id": baselines[key].source_id,
                "incumbent_action_id": actions_by_method["incumbent"][key],
                "incumbent_score": float(scores_by_method["incumbent"][key]),
                "incumbent_called": bool(values["incumbent"]["call"][key]),
                "decoupled_action_id": actions_by_method["decoupled"][key],
                "decoupled_score": float(scores_by_method["decoupled"][key]),
                "decoupled_called": bool(values["decoupled"]["call"][key]),
            }
        )

    source_metrics: dict[str, dict[str, float]] = {
        source: {} for source in sorted(set(source_by_key.values()))
    }
    source_points: dict[str, dict[str, float | None]] = {}
    question_points: dict[str, dict[str, float | None]] = {}
    source_helpful_state = _source_means(helpful_state, source_by_key)
    helpful_state_source_mass = mean(source_helpful_state.values())
    helpful_state_question_mass = mean(helpful_state.values())
    for method, metrics in values.items():
        per_source = {
            name: _source_means(metric, source_by_key)
            for name, metric in metrics.items()
        }
        for source in source_metrics:
            for name in metrics:
                source_metrics[source][f"{method}_{name}"] = per_source[name][source]
        source_points[method] = {
            name: mean(per_source[name].values()) for name in metrics
        }
        question_points[method] = {
            name: mean(metric.values()) for name, metric in metrics.items()
        }
        for points in (source_points[method], question_points[method]):
            points["gain_per_call"] = _safe_ratio(
                float(points["gain"]), float(points["call"])
            )
            points["helpful_call_precision"] = _safe_ratio(
                float(points["helpful_call"]), float(points["call"])
            )
            points["helpful_call_recall"] = _safe_ratio(
                float(points["helpful_call"]), float(points["helpful_proposal"])
            )
            points["proposal_helpful_state_recovery"] = _safe_ratio(
                float(points["helpful_proposal"]),
                helpful_state_source_mass
                if points is source_points[method]
                else helpful_state_question_mass,
            )

    for source in source_metrics:
        source_metrics[source]["decoupled_minus_incumbent_utility"] = (
            source_metrics[source]["decoupled_utility"]
            - source_metrics[source]["incumbent_utility"]
        )
    bootstrap = bootstrap_source_balanced_metrics(
        source_metrics,
        n_resamples=bootstrap_resamples,
        confidence_level=DECOUPLED_CONFIDENCE,
        seed=bootstrap_seed,
    )
    action_disagreement = mean(
        actions_by_method["incumbent"][key]
        != actions_by_method["decoupled"][key]
        for key in baselines
    )
    gate_disagreement = mean(
        bool(values["incumbent"]["call"][key])
        != bool(values["decoupled"]["call"][key])
        for key in baselines
    )
    return {
        "source_balanced": source_points,
        "question_balanced": question_points,
        "primary_estimand": bootstrap["metrics"][
            "decoupled_minus_incumbent_utility"
        ],
        "paired_source_bootstrap": bootstrap,
        "action_disagreement_rate": action_disagreement,
        "gate_disagreement_rate": gate_disagreement,
        "score_rows": score_rows,
    }


def evaluate_decoupled_loss_proposal_gate(
    records_by_domain: Mapping[str, Sequence[ActionRecord]],
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    semantic_decisions_by_domain: Mapping[
        str, Mapping[DecisionKey, Mapping[str, Any]]
    ],
    feature_mode: str = "hybrid-context-semantic",
    bootstrap_resamples: int = DECOUPLED_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DECOUPLED_SEED,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if feature_mode != "hybrid-context-semantic":
        raise ValueError("decoupled protocol requires hybrid-context-semantic")
    domain_by_key, baselines, zooms = _validate_domains(records_by_domain)
    semantic_by_key = _semantic_feature_index(
        feature_mode=feature_mode,
        records_by_domain=records_by_domain,
        domain_by_key=domain_by_key,
        semantic_decisions_by_domain=semantic_decisions_by_domain,
    )
    keys = sorted(baselines)
    loss_actions, loss_scores = _prediction_index(prediction_rows, set(keys))
    fold_by_key, fold_source_counts = _source_folds(
        domain_by_key,
        baselines,
        n_folds=INCUMBENT_FOLDS,
        seed=INCUMBENT_SEED,
    )
    incumbent_actions: dict[DecisionKey, str] = {}
    incumbent_scores: dict[DecisionKey, float] = {}
    decoupled_scores: dict[DecisionKey, float] = {}
    fold_counts: list[dict[str, int]] = []
    for fold in range(INCUMBENT_FOLDS):
        train_keys = [key for key in keys if fold_by_key[key] != fold]
        test_keys = [key for key in keys if fold_by_key[key] == fold]
        heads = _fit_heads(
            train_keys,
            alpha=INCUMBENT_ALPHA,
            seed=INCUMBENT_SEED + fold,
            feature_mode=feature_mode,
            baselines=baselines,
            zooms=zooms,
            domain_by_key=domain_by_key,
            semantic_by_key=semantic_by_key,
        )
        values, actions = _score_heads(
            heads,
            test_keys,
            lambda_cost=DECOUPLED_LAMBDA_COST,
            feature_mode=feature_mode,
            baselines=baselines,
            zooms=zooms,
            semantic_by_key=semantic_by_key,
        )
        forced_values = _forced_factorized_scores(
            heads,
            test_keys,
            forced_actions=loss_actions,
            baselines=baselines,
            zooms=zooms,
            feature_mode=feature_mode,
            semantic_by_key=semantic_by_key,
        )
        incumbent_actions.update(actions)
        incumbent_scores.update(values)
        decoupled_scores.update(forced_values)
        fold_counts.append(
            {
                "fold": fold,
                "train_decisions": len(train_keys),
                "test_decisions": len(test_keys),
            }
        )
    if not (
        set(incumbent_actions)
        == set(incumbent_scores)
        == set(decoupled_scores)
        == set(loss_actions)
        == set(keys)
    ):
        raise RuntimeError("decoupled OOF reconstruction is incomplete")

    incumbent_match = match_call_count_threshold(
        incumbent_scores, target_calls=DECOUPLED_TARGET_CALLS
    )
    decoupled_match = match_call_count_threshold(
        decoupled_scores, target_calls=DECOUPLED_TARGET_CALLS
    )
    incumbent_matched_call_keys = {
        key
        for key, score in incumbent_scores.items()
        if score >= float(incumbent_match["threshold"])
    }
    incumbent_frozen_call_keys = {
        key
        for key, score in incumbent_scores.items()
        if score >= INCUMBENT_THRESHOLD
    }
    if (
        incumbent_match["calls"] != DECOUPLED_TARGET_CALLS
        or incumbent_matched_call_keys != incumbent_frozen_call_keys
    ):
        raise ValueError("incumbent matched-call set no longer reproduces")
    evaluated = _evaluate(
        baselines=baselines,
        zooms=zooms,
        actions_by_method={
            "incumbent": incumbent_actions,
            "decoupled": loss_actions,
        },
        scores_by_method={
            "incumbent": incumbent_scores,
            "decoupled": decoupled_scores,
        },
        threshold_by_method={
            "incumbent": float(incumbent_match["threshold"]),
            "decoupled": float(decoupled_match["threshold"]),
        },
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
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
        raise ValueError("incumbent OOF metrics no longer reproduce")

    source_points = evaluated["source_balanced"]
    primary = evaluated["primary_estimand"]
    pass_rule = {
        "utility_margin_at_least_0_00025": float(
            source_points["decoupled"]["utility"]
        )
        >= float(source_points["incumbent"]["utility"]) + 0.00025,
        "paired_ci_low_above_minus_0_0005": float(primary["ci_low"]) > -0.0005,
        "gain_per_call_higher": float(
            source_points["decoupled"]["gain_per_call"]
        )
        > float(source_points["incumbent"]["gain_per_call"]),
        "harm_and_negative_calls_no_greater": float(
            source_points["decoupled"]["induced_harm"]
        )
        <= float(source_points["incumbent"]["induced_harm"])
        and float(source_points["decoupled"]["negative_value_call"])
        <= float(source_points["incumbent"]["negative_value_call"]),
        "proposal_recovery_higher": float(
            source_points["decoupled"]["proposal_helpful_state_recovery"]
        )
        > float(source_points["incumbent"]["proposal_helpful_state_recovery"]),
        "all_audits_passed": True,
    }
    score_report = {
        "scientific_status": (
            "outcome-blind matched-call OOF scores frozen before development "
            "outcome evaluation"
        ),
        "n_sources": len(set(record.source_id for record in baselines.values())),
        "n_decisions": len(keys),
        "feature_mode": feature_mode,
        "fold_source_counts": fold_source_counts,
        "fold_counts": fold_counts,
        "incumbent_match": incumbent_match,
        "decoupled_match": decoupled_match,
        "loss_proposer_scores_used_for_thresholds": False,
        "task_outcomes_used_for_thresholds": False,
        "serialized_outcome_fields": [],
    }
    report = {
        "scientific_status": (
            "opened DocVQA development diagnostic; not independent validation"
        ),
        "decision": (
            "decoupled_loss_proposal_gate_advanced"
            if all(pass_rule.values())
            else "decoupled_loss_proposal_gate_not_advanced"
        ),
        "pass_rule": pass_rule,
        "n_sources": score_report["n_sources"],
        "n_decisions": len(keys),
        "lambda_cost": DECOUPLED_LAMBDA_COST,
        "source_balanced": source_points,
        "question_balanced": evaluated["question_balanced"],
        "primary_estimand": primary,
        "paired_source_bootstrap": evaluated["paired_source_bootstrap"],
        "action_disagreement_rate": evaluated["action_disagreement_rate"],
        "gate_disagreement_rate": evaluated["gate_disagreement_rate"],
        "screenqa_inputs_used": False,
        "protected_role_inputs_used": False,
    }
    return report, score_report, evaluated["score_rows"]
