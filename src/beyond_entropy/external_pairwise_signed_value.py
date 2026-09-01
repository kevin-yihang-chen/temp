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
from .rescue_gate import DecisionKey
from .scaled_action_value import (
    _PreparedDecisions,
    _RankedAction,
    _call_features,
    _fit_call_value_head,
    _fit_pairwise_ranker,
    _predict_call_values,
    _prepare_decisions,
    _serialize_linear,
    _source_folds,
)
from .schema import ActionRecord


EXTERNAL_PAIRWISE_SEED = 20260911
EXTERNAL_PAIRWISE_FOLDS = 5
EXTERNAL_PAIRWISE_RANKER_C = 0.01
EXTERNAL_PAIRWISE_CALL_ALPHA = 100.0
EXTERNAL_PAIRWISE_MAX_ITER = 2000
EXTERNAL_PAIRWISE_TARGET_CALLS = 225
EXTERNAL_PAIRWISE_BOOTSTRAP_RESAMPLES = 20_000
EXTERNAL_PAIRWISE_LAMBDA_COST = 0.05
EXTERNAL_PAIRWISE_FEATURE_MODE = "semantic-context"
EXTERNAL_PAIRWISE_STATE_FEATURE_COUNT = 60
EXTERNAL_PAIRWISE_ACTION_FEATURE_COUNT = 46
EXTERNAL_PAIRWISE_CALL_FEATURE_COUNT = 110


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
                "decoupled", "external_pairwise_signed_value"
            ): _rename_candidate(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rename_candidate(item) for item in value]
    return value


def _rank_actions_small_tie(
    prepared: _PreparedDecisions,
    keys: Sequence[DecisionKey],
    *,
    scaler: Any,
    model: Any,
) -> dict[DecisionKey, _RankedAction]:
    ranked: dict[DecisionKey, _RankedAction] = {}
    for key in keys:
        candidates = prepared.zooms[key]
        features = np.asarray(
            [
                prepared.action_features[(key, candidate.action_id)]
                for candidate in candidates
            ],
            dtype=np.float64,
        )
        scores = np.asarray(
            model.decision_function(scaler.transform(features)), dtype=np.float64
        )
        if scores.shape != (len(candidates),) or not np.isfinite(scores).all():
            raise RuntimeError("external pairwise ranker produced invalid scores")
        selected_index = min(
            range(len(candidates)),
            key=lambda index: (-float(scores[index]), candidates[index].action_id),
        )
        ranked[key] = _RankedAction(
            action_id=candidates[selected_index].action_id,
            action_index=selected_index,
            action_scores=tuple(float(value) for value in scores.tolist()),
        )
    return ranked


def _crossfit_ranker_small_tie(
    prepared: _PreparedDecisions,
    keys: Sequence[DecisionKey],
    *,
    n_folds: int,
    seed: int,
) -> tuple[dict[DecisionKey, _RankedAction], list[dict[str, Any]]]:
    fold_by_key = _source_folds(
        keys, prepared.baselines, n_folds=n_folds, seed=seed
    )
    predictions: dict[DecisionKey, _RankedAction] = {}
    audits: list[dict[str, Any]] = []
    for fold in range(n_folds):
        train_keys = [key for key in keys if fold_by_key[key] != fold]
        test_keys = [key for key in keys if fold_by_key[key] == fold]
        train_sources = {prepared.baselines[key].source_id for key in train_keys}
        test_sources = {prepared.baselines[key].source_id for key in test_keys}
        overlap = train_sources & test_sources
        if overlap:
            raise RuntimeError("external pairwise inner folds leak sources")
        scaler, model, pair_counts = _fit_pairwise_ranker(
            prepared,
            train_keys,
            c_value=EXTERNAL_PAIRWISE_RANKER_C,
            seed=seed + fold,
        )
        if int(model.n_iter_[0]) >= EXTERNAL_PAIRWISE_MAX_ITER:
            raise RuntimeError("external pairwise inner ranker did not converge")
        predictions.update(
            _rank_actions_small_tie(
                prepared, test_keys, scaler=scaler, model=model
            )
        )
        audits.append(
            {
                "fold": fold,
                "train_sources": len(train_sources),
                "test_sources": len(test_sources),
                "source_overlap": 0,
                "source_exclusion_passed": True,
                **pair_counts,
                "iterations": int(model.n_iter_[0]),
            }
        )
    if set(predictions) != set(keys):
        raise RuntimeError("external pairwise inner OOF rankings are incomplete")
    return predictions, audits


def _fit_external_oof(
    prepared: _PreparedDecisions,
    *,
    n_folds: int,
    seed: int,
) -> tuple[
    dict[DecisionKey, _RankedAction],
    dict[DecisionKey, float],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    fold_by_key = _source_folds(
        prepared.keys, prepared.baselines, n_folds=n_folds, seed=seed
    )
    rankings: dict[DecisionKey, _RankedAction] = {}
    predicted_gains: dict[DecisionKey, float] = {}
    serialized_folds: list[dict[str, Any]] = []
    fold_audits: list[dict[str, Any]] = []
    for outer_fold in range(n_folds):
        train_keys = [
            key for key in prepared.keys if fold_by_key[key] != outer_fold
        ]
        test_keys = [key for key in prepared.keys if fold_by_key[key] == outer_fold]
        train_sources = {prepared.baselines[key].source_id for key in train_keys}
        test_sources = {prepared.baselines[key].source_id for key in test_keys}
        overlap = train_sources & test_sources
        if overlap:
            raise RuntimeError("external pairwise outer folds leak sources")
        inner_rankings, inner_audits = _crossfit_ranker_small_tie(
            prepared,
            train_keys,
            n_folds=n_folds,
            seed=seed + 1000 + outer_fold,
        )
        ranker_scaler, ranker_model, pair_counts = _fit_pairwise_ranker(
            prepared,
            train_keys,
            c_value=EXTERNAL_PAIRWISE_RANKER_C,
            seed=seed + 2000 + outer_fold,
        )
        if int(ranker_model.n_iter_[0]) >= EXTERNAL_PAIRWISE_MAX_ITER:
            raise RuntimeError("external pairwise outer ranker did not converge")
        outer_rankings = _rank_actions_small_tie(
            prepared,
            test_keys,
            scaler=ranker_scaler,
            model=ranker_model,
        )
        call_scaler, call_model = _fit_call_value_head(
            prepared,
            inner_rankings,
            train_keys,
            alpha=EXTERNAL_PAIRWISE_CALL_ALPHA,
        )
        outer_predictions = _predict_call_values(
            prepared,
            outer_rankings,
            test_keys,
            scaler=call_scaler,
            model=call_model,
        )
        rankings.update(outer_rankings)
        predicted_gains.update(outer_predictions)
        fold_audits.append(
            {
                "fold": outer_fold,
                "train_decisions": len(train_keys),
                "test_decisions": len(test_keys),
                "train_sources": len(train_sources),
                "test_sources": len(test_sources),
                "source_overlap": len(overlap),
                "source_exclusion_passed": True,
                "inner_source_exclusion_passed": all(
                    audit["source_exclusion_passed"] for audit in inner_audits
                ),
                "inner_selected_actions_cover_outer_train": set(inner_rankings)
                == set(train_keys),
                "ranker_iterations": int(ranker_model.n_iter_[0]),
                "call_training_rows": len(train_keys),
                **pair_counts,
            }
        )
        serialized_folds.append(
            {
                "fold": outer_fold,
                "ranker": _serialize_linear(ranker_scaler, ranker_model),
                "call_value": _serialize_linear(call_scaler, call_model),
                "training_audit": fold_audits[-1],
                "inner_fold_audits": inner_audits,
            }
        )
    if set(rankings) != set(prepared.keys) or set(predicted_gains) != set(
        prepared.keys
    ):
        raise RuntimeError("external pairwise OOF predictions are incomplete")
    if not all(math.isfinite(value) for value in predicted_gains.values()):
        raise RuntimeError("external pairwise predicted gains are non-finite")
    return rankings, predicted_gains, serialized_folds, fold_audits


def _incumbent_index(
    rows: Sequence[Mapping[str, Any]],
    *,
    baselines: Mapping[DecisionKey, ActionRecord],
) -> tuple[
    dict[DecisionKey, str],
    dict[DecisionKey, float],
    dict[DecisionKey, bool],
]:
    actions: dict[DecisionKey, str] = {}
    scores: dict[DecisionKey, float] = {}
    calls: dict[DecisionKey, bool] = {}
    for row in rows:
        if _FORBIDDEN_OUTPUT_FIELDS.intersection(row):
            raise ValueError("external pairwise incumbent rows leak outcomes")
        key = (str(row.get("state_id", "")), str(row.get("replicate_id", "")))
        action_id = str(row.get("incumbent_action_id", ""))
        score = float(row.get("incumbent_score", math.nan))
        called = row.get("incumbent_called")
        if (
            not all(key)
            or key not in baselines
            or key in actions
            or not action_id
            or not math.isfinite(score)
            or not isinstance(called, bool)
            or str(row.get("source_id", "")) != baselines[key].source_id
        ):
            raise ValueError("external pairwise incumbent row is invalid")
        actions[key] = action_id
        scores[key] = score
        calls[key] = called
    if not set(actions) == set(scores) == set(calls) == set(baselines):
        raise ValueError("external pairwise incumbent coverage differs")
    return actions, scores, calls


def evaluate_external_pairwise_signed_value(
    records: Sequence[ActionRecord],
    audited_score_rows: Sequence[Mapping[str, Any]],
    *,
    semantic_decisions: Mapping[DecisionKey, Mapping[str, Any]],
    bound_inputs_verified: bool,
    n_folds: int = EXTERNAL_PAIRWISE_FOLDS,
    bootstrap_resamples: int = EXTERNAL_PAIRWISE_BOOTSTRAP_RESAMPLES,
    seed: int = EXTERNAL_PAIRWISE_SEED,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if not bound_inputs_verified:
        raise ValueError("external pairwise fitting requires bound inputs")
    if n_folds != EXTERNAL_PAIRWISE_FOLDS or seed != EXTERNAL_PAIRWISE_SEED:
        raise ValueError("external pairwise folds and seed are frozen")
    if bootstrap_resamples != EXTERNAL_PAIRWISE_BOOTSTRAP_RESAMPLES:
        raise ValueError("external pairwise bootstrap count is frozen")
    prepared = _prepare_decisions(
        records,
        feature_mode=EXTERNAL_PAIRWISE_FEATURE_MODE,
        semantic_decisions=semantic_decisions,
    )
    if (
        len(prepared.keys) != 13_580
        or len({prepared.baselines[key].source_id for key in prepared.keys}) != 3_500
        or any(len(prepared.zooms[key]) != 4 for key in prepared.keys)
    ):
        raise ValueError("external pairwise population contract changed")
    rankings, predicted_gains, serialized_folds, fold_audits = _fit_external_oof(
        prepared, n_folds=n_folds, seed=seed
    )
    candidate_actions = {
        key: ranking.action_id for key, ranking in rankings.items()
    }
    candidate_scores = {
        key: predicted_gains[key]
        - EXTERNAL_PAIRWISE_LAMBDA_COST
        * prepared.zooms[key][rankings[key].action_index].tool_cost
        for key in prepared.keys
    }
    incumbent_actions, incumbent_scores, audited_incumbent_calls = _incumbent_index(
        audited_score_rows, baselines=prepared.baselines
    )
    incumbent_match = match_call_count_threshold(
        incumbent_scores, target_calls=EXTERNAL_PAIRWISE_TARGET_CALLS
    )
    candidate_match = match_call_count_threshold(
        candidate_scores, target_calls=EXTERNAL_PAIRWISE_TARGET_CALLS
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
        incumbent_match["calls"] != EXTERNAL_PAIRWISE_TARGET_CALLS
        or candidate_match["calls"] != EXTERNAL_PAIRWISE_TARGET_CALLS
        or incumbent_call_keys != audited_call_keys
    ):
        raise ValueError("external pairwise matched-call contract changed")
    baselines = prepared.baselines
    zooms = prepared.zooms
    evaluated = _rename_candidate(
        _evaluate(
            baselines=baselines,
            zooms=zooms,
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
        raise ValueError("external pairwise incumbent metrics do not reproduce")

    first_key = prepared.keys[0]
    state_feature_count = len(prepared.state_features[first_key])
    action_feature_count = len(
        prepared.action_features[(first_key, prepared.zooms[first_key][0].action_id)]
    )
    call_feature_count = len(_call_features(prepared, first_key, rankings[first_key]))
    if (
        state_feature_count != EXTERNAL_PAIRWISE_STATE_FEATURE_COUNT
        or action_feature_count != EXTERNAL_PAIRWISE_ACTION_FEATURE_COUNT
        or call_feature_count != EXTERNAL_PAIRWISE_CALL_FEATURE_COUNT
    ):
        raise ValueError("external pairwise feature dimensions changed")
    full_ranker_scaler, full_ranker_model, full_pair_counts = _fit_pairwise_ranker(
        prepared,
        prepared.keys,
        c_value=EXTERNAL_PAIRWISE_RANKER_C,
        seed=seed + 3000,
    )
    if int(full_ranker_model.n_iter_[0]) >= EXTERNAL_PAIRWISE_MAX_ITER:
        raise RuntimeError("external pairwise full ranker did not converge")
    full_call_scaler, full_call_model = _fit_call_value_head(
        prepared,
        rankings,
        prepared.keys,
        alpha=EXTERNAL_PAIRWISE_CALL_ALPHA,
    )

    source_points = evaluated["source_balanced"]
    incumbent = source_points["incumbent"]
    candidate = source_points["external_pairwise_signed_value"]
    primary = evaluated["primary_estimand"]
    audits = {
        "bound_input_hashes_verified": True,
        "population_exact": True,
        "semantic_action_alignment_exact": True,
        "feature_dimensions_exact": True,
        "outer_source_exclusion": all(
            fold["source_exclusion_passed"] for fold in fold_audits
        ),
        "inner_source_exclusion": all(
            fold["inner_source_exclusion_passed"] for fold in fold_audits
        ),
        "inner_selected_action_coverage": all(
            fold["inner_selected_actions_cover_outer_train"] for fold in fold_audits
        ),
        "pairwise_rankers_converged": all(
            fold["ranker_iterations"] < EXTERNAL_PAIRWISE_MAX_ITER
            for fold in fold_audits
        )
        and int(full_ranker_model.n_iter_[0]) < EXTERNAL_PAIRWISE_MAX_ITER,
        "oof_score_coverage_exact": set(candidate_scores) == set(prepared.keys),
        "full_call_head_uses_oof_actions": set(rankings) == set(prepared.keys),
        "matched_call_counts_exact": incumbent_match["calls"]
        == candidate_match["calls"]
        == EXTERNAL_PAIRWISE_TARGET_CALLS,
        "incumbent_call_set_reproduced": incumbent_call_keys == audited_call_keys,
        "incumbent_pooled_metrics_reproduced": True,
        "serialized_scores_finite": all(
            math.isfinite(value) for value in candidate_scores.values()
        ),
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
        "helpful_call_precision_no_lower": float(
            candidate["helpful_call_precision"]
        )
        >= float(incumbent["helpful_call_precision"]),
        "all_audits_passed": all_audits_passed,
    }
    score_rows: list[dict[str, Any]] = []
    candidate_threshold = float(candidate_match["threshold"])
    outer_fold_by_key = _source_folds(
        prepared.keys,
        prepared.baselines,
        n_folds=n_folds,
        seed=seed,
    )
    for key in prepared.keys:
        ranking = rankings[key]
        ordered_scores = sorted(ranking.action_scores, reverse=True)
        row = {
            "state_id": key[0],
            "replicate_id": key[1],
            "source_id": prepared.baselines[key].source_id,
            "outer_fold": int(outer_fold_by_key[key]),
            "external_pairwise_signed_value_action_id": ranking.action_id,
            "external_pairwise_signed_value_predicted_gain": float(
                predicted_gains[key]
            ),
            "external_pairwise_signed_value_score": float(candidate_scores[key]),
            "external_pairwise_signed_value_called": bool(
                candidate_scores[key] >= candidate_threshold
            ),
            "external_pairwise_ranker_selected_score": float(
                ranking.action_scores[ranking.action_index]
            ),
            "external_pairwise_ranker_top2_gap": float(
                ordered_scores[0] - ordered_scores[1]
            ),
            "incumbent_action_id": incumbent_actions[key],
            "incumbent_score": float(incumbent_scores[key]),
            "incumbent_called": bool(audited_incumbent_calls[key]),
        }
        if _FORBIDDEN_OUTPUT_FIELDS.intersection(row):
            raise RuntimeError("external pairwise serialized rows leak outcomes")
        score_rows.append(row)
    score_report = {
        "scientific_status": (
            "outcome-free externally fixed pairwise signed-value OOF scores "
            "frozen before opened-development outcome evaluation"
        ),
        "n_sources": 3_500,
        "n_decisions": 13_580,
        "n_folds": n_folds,
        "feature_mode": EXTERNAL_PAIRWISE_FEATURE_MODE,
        "state_feature_count": state_feature_count,
        "action_feature_count": action_feature_count,
        "call_feature_count": call_feature_count,
        "ranker_c": EXTERNAL_PAIRWISE_RANKER_C,
        "call_alpha": EXTERNAL_PAIRWISE_CALL_ALPHA,
        "fold_training": fold_audits,
        "incumbent_match": incumbent_match,
        "external_pairwise_signed_value_match": candidate_match,
        "task_outcomes_used_for_thresholds": False,
        "serialized_outcome_fields": [],
        "audits": audits,
    }
    report = {
        "scientific_status": "opened DocVQA development diagnostic; not independent validation",
        "decision": (
            "external_pairwise_signed_value_advanced"
            if all(pass_rule.values())
            else "external_pairwise_signed_value_not_advanced"
        ),
        "pass_rule": pass_rule,
        "n_sources": 3_500,
        "n_decisions": 13_580,
        "lambda_cost": EXTERNAL_PAIRWISE_LAMBDA_COST,
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
        "schema": "external_pairwise_signed_value_v1",
        "feature_mode": EXTERNAL_PAIRWISE_FEATURE_MODE,
        "state_feature_count": state_feature_count,
        "action_feature_count": action_feature_count,
        "call_feature_count": call_feature_count,
        "seed": seed,
        "n_folds": n_folds,
        "lambda_cost": EXTERNAL_PAIRWISE_LAMBDA_COST,
        "ranker_c": EXTERNAL_PAIRWISE_RANKER_C,
        "call_alpha": EXTERNAL_PAIRWISE_CALL_ALPHA,
        "oof_folds": serialized_folds,
        "full_refit": {
            "ranker": _serialize_linear(full_ranker_scaler, full_ranker_model),
            "call_value": _serialize_linear(full_call_scaler, full_call_model),
            "ranker_training": {
                **full_pair_counts,
                "iterations": int(full_ranker_model.n_iter_[0]),
            },
            "call_training_uses_oof_selected_actions": True,
        },
        "screenqa_inputs_used": False,
    }
    return report, score_report, model, score_rows
