from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np  # type: ignore[import-not-found]

from .action_value import _semantic_feature_index, _state_features, _validate_domains
from .decoupled_loss_gate import (
    INCUMBENT_POOLED_CALL_RATE,
    INCUMBENT_POOLED_GAIN,
    INCUMBENT_POOLED_UTILITY,
    _evaluate,
    _prediction_index,
    match_call_count_threshold,
)
from .oof_action_value import _source_folds
from .proposal_conditioned_factorized_gate import (
    FACTORIZED_CONDITIONED_ACTION_FEATURE_COUNT,
    FACTORIZED_CONDITIONED_C,
    FACTORIZED_CONDITIONED_LAMBDA_COST,
    FACTORIZED_CONDITIONED_MAX_ITER,
    FACTORIZED_CONDITIONED_STATE_FEATURE_COUNT,
    _fit_binary_head,
    _serialize_factorized_heads,
    _weight_mass_matches_rows,
)
from .proposal_conditioned_gate import (
    _assert_source_exclusion,
    _audited_incumbent_index,
    _selected_action_features,
    validate_bound_loss_proposer_report,
)
from .rescue_gate import DecisionKey
from .schema import ActionRecord


DUAL_UNION_SEED = 20260908
DUAL_UNION_FOLDS = 5
DUAL_UNION_TARGET_CALLS = 225
DUAL_UNION_BOOTSTRAP_RESAMPLES = 20000
DUAL_UNION_EQUAL_PAIRS = 4875
DUAL_UNION_UNEQUAL_PAIRS = 8705
DUAL_UNION_ROWS = 22285


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


CandidateKey = tuple[DecisionKey, str]


def _union_actions(
    keys: Sequence[DecisionKey],
    *,
    incumbent_actions: Mapping[DecisionKey, str],
    loss_actions: Mapping[DecisionKey, str],
) -> tuple[dict[DecisionKey, tuple[str, ...]], dict[str, int]]:
    union: dict[DecisionKey, tuple[str, ...]] = {}
    equal = 0
    for key in keys:
        actions = tuple(sorted({incumbent_actions[key], loss_actions[key]}))
        if len(actions) not in (1, 2):
            raise RuntimeError("dual-proposer union cardinality is invalid")
        equal += int(len(actions) == 1)
        union[key] = actions
    counts = {
        "equal_proposal_pairs": equal,
        "unequal_proposal_pairs": len(keys) - equal,
        "unique_union_rows": sum(len(actions) for actions in union.values()),
    }
    return union, counts


def _candidate_balanced_weights(
    pairs: Sequence[CandidateKey],
    *,
    domain_by_key: Mapping[DecisionKey, str],
    baselines: Mapping[DecisionKey, ActionRecord],
) -> np.ndarray:
    if not pairs or len(set(pairs)) != len(pairs):
        raise ValueError("candidate-balanced rows must be non-empty and unique")
    domains = {domain_by_key[key] for key, _action_id in pairs}
    sources_by_domain: dict[str, set[str]] = {}
    decisions_by_source: dict[tuple[str, str], set[DecisionKey]] = {}
    candidates_by_decision: dict[tuple[str, DecisionKey], int] = {}
    for key, _action_id in pairs:
        domain = domain_by_key[key]
        source = baselines[key].source_id
        sources_by_domain.setdefault(domain, set()).add(source)
        decisions_by_source.setdefault((domain, source), set()).add(key)
        candidate_key = (domain, key)
        candidates_by_decision[candidate_key] = (
            candidates_by_decision.get(candidate_key, 0) + 1
        )
    raw = []
    for key, _action_id in pairs:
        domain = domain_by_key[key]
        source = baselines[key].source_id
        raw.append(
            1.0
            / (
                len(domains)
                * len(sources_by_domain[domain])
                * len(decisions_by_source[(domain, source)])
                * candidates_by_decision[(domain, key)]
            )
        )
    weights = np.asarray(raw, dtype=np.float64)
    weights *= len(weights) / float(weights.sum())
    if not _weight_mass_matches_rows(float(weights.sum()), len(weights)):
        raise RuntimeError("candidate-balanced weights are not row-normalized")
    return weights


def _fit_candidate_head(
    features: np.ndarray,
    labels: Sequence[int],
    weights: np.ndarray,
    *,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

    if (
        features.ndim != 2
        or features.shape[0] == 0
        or len(labels) != features.shape[0]
        or weights.shape != (features.shape[0],)
    ):
        raise ValueError("dual-union candidate-head rows are not aligned")
    if features.shape[1] != FACTORIZED_CONDITIONED_ACTION_FEATURE_COUNT:
        raise ValueError("dual-union candidate feature count is invalid")
    if not np.isfinite(features).all() or not np.isfinite(weights).all():
        raise ValueError("dual-union candidate-head inputs must be finite")
    label_array = np.asarray(labels, dtype=np.int64)
    if set(label_array.tolist()) != {0, 1}:
        raise ValueError("dual-union candidate head requires both classes")
    scaler = StandardScaler().fit(features)
    model = LogisticRegression(
        C=FACTORIZED_CONDITIONED_C,
        penalty="l2",
        solver="liblinear",
        max_iter=FACTORIZED_CONDITIONED_MAX_ITER,
        random_state=seed,
    ).fit(scaler.transform(features), label_array, sample_weight=weights)
    if int(model.n_iter_[0]) >= FACTORIZED_CONDITIONED_MAX_ITER:
        raise RuntimeError("dual-union candidate logistic head did not converge")
    return (
        {"scaler": scaler, "model": model},
        {
            "train_rows": int(features.shape[0]),
            "feature_count": int(features.shape[1]),
            "negative_rows": int(np.sum(label_array == 0)),
            "positive_rows": int(np.sum(label_array == 1)),
            "weight_mass": float(weights.sum()),
            "iterations": int(model.n_iter_[0]),
            "weighting": "equal_domain_source_decision_candidate",
            "class_balancing": False,
        },
    )


def _fit_union_heads(
    train_keys: Sequence[DecisionKey],
    *,
    union: Mapping[DecisionKey, Sequence[str]],
    state_features_by_key: Mapping[DecisionKey, np.ndarray],
    action_features_by_pair: Mapping[CandidateKey, np.ndarray],
    actions_by_pair: Mapping[CandidateKey, ActionRecord],
    baselines: Mapping[DecisionKey, ActionRecord],
    domain_by_key: Mapping[DecisionKey, str],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    states = np.stack([state_features_by_key[key] for key in train_keys])
    error_labels = [int(baselines[key].correct_before < 0.5) for key in train_keys]
    error_head, _error_weights, error_audit = _fit_binary_head(
        states,
        error_labels,
        [domain_by_key[key] for key in train_keys],
        [baselines[key].source_id for key in train_keys],
        seed=seed,
    )
    rescue_pairs = [
        (key, action_id)
        for key in train_keys
        if baselines[key].correct_before < 0.5
        for action_id in union[key]
    ]
    harm_pairs = [
        (key, action_id)
        for key in train_keys
        if baselines[key].correct_before >= 0.5
        for action_id in union[key]
    ]
    rescue_weights = _candidate_balanced_weights(
        rescue_pairs, domain_by_key=domain_by_key, baselines=baselines
    )
    harm_weights = _candidate_balanced_weights(
        harm_pairs, domain_by_key=domain_by_key, baselines=baselines
    )
    rescue_labels = [
        int(actions_by_pair[pair].delta_success > 0.0) for pair in rescue_pairs
    ]
    harm_labels = [
        int(actions_by_pair[pair].delta_success < 0.0) for pair in harm_pairs
    ]
    rescue_head, rescue_audit = _fit_candidate_head(
        np.stack([action_features_by_pair[pair] for pair in rescue_pairs]),
        rescue_labels,
        rescue_weights,
        seed=seed,
    )
    harm_head, harm_audit = _fit_candidate_head(
        np.stack([action_features_by_pair[pair] for pair in harm_pairs]),
        harm_labels,
        harm_weights,
        seed=seed,
    )
    rescue_positive = np.asarray(rescue_labels, dtype=np.int64) == 1
    harm_positive = np.asarray(harm_labels, dtype=np.int64) == 1
    rescue_magnitude = float(
        np.average(
            np.asarray(
                [actions_by_pair[pair].delta_success for pair in rescue_pairs],
                dtype=np.float64,
            )[rescue_positive],
            weights=rescue_weights[rescue_positive],
        )
    )
    harm_magnitude = float(
        np.average(
            -np.asarray(
                [actions_by_pair[pair].delta_success for pair in harm_pairs],
                dtype=np.float64,
            )[harm_positive],
            weights=harm_weights[harm_positive],
        )
    )
    heads = {
        "error": error_head,
        "rescue": rescue_head,
        "harm": harm_head,
        "rescue_magnitude": rescue_magnitude,
        "harm_magnitude": harm_magnitude,
    }
    audit = {
        "train_decisions": len(train_keys),
        "state_feature_count": int(states.shape[1]),
        "action_feature_count": FACTORIZED_CONDITIONED_ACTION_FEATURE_COUNT,
        "error": error_audit,
        "rescue": rescue_audit,
        "harm": harm_audit,
        "rescue_magnitude": rescue_magnitude,
        "harm_magnitude": harm_magnitude,
    }
    return heads, audit


def _score_union_key(
    heads: Mapping[str, Any],
    key: DecisionKey,
    *,
    union: Mapping[DecisionKey, Sequence[str]],
    state_features_by_key: Mapping[DecisionKey, np.ndarray],
    action_features_by_pair: Mapping[CandidateKey, np.ndarray],
    actions_by_pair: Mapping[CandidateKey, ActionRecord],
) -> tuple[str, float, float, float, float]:
    state = state_features_by_key[key][None, :]
    error_probability = float(
        heads["error"]["model"].predict_proba(
            heads["error"]["scaler"].transform(state)
        )[0, 1]
    )
    candidates: list[tuple[float, str, float, float]] = []
    for action_id in union[key]:
        pair = (key, action_id)
        features = action_features_by_pair[pair][None, :]
        rescue_probability = float(
            heads["rescue"]["model"].predict_proba(
                heads["rescue"]["scaler"].transform(features)
            )[0, 1]
        )
        harm_probability = float(
            heads["harm"]["model"].predict_proba(
                heads["harm"]["scaler"].transform(features)
            )[0, 1]
        )
        score = (
            error_probability
            * rescue_probability
            * float(heads["rescue_magnitude"])
            - (1.0 - error_probability)
            * harm_probability
            * float(heads["harm_magnitude"])
            - FACTORIZED_CONDITIONED_LAMBDA_COST
            * actions_by_pair[pair].tool_cost
        )
        if not all(
            math.isfinite(value)
            for value in (error_probability, rescue_probability, harm_probability, score)
        ):
            raise RuntimeError("dual-union scorer produced non-finite values")
        candidates.append((score, action_id, rescue_probability, harm_probability))
    score, action_id, rescue_probability, harm_probability = min(
        candidates, key=lambda item: (-item[0], item[1])
    )
    return action_id, score, error_probability, rescue_probability, harm_probability


def _rename_candidate(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key).replace("decoupled", "dual_proposer_union"): _rename_candidate(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rename_candidate(item) for item in value]
    return value


def evaluate_dual_proposer_union_factorized_gate(
    records_by_domain: Mapping[str, Sequence[ActionRecord]],
    prediction_rows: Sequence[Mapping[str, Any]],
    audited_score_rows: Sequence[Mapping[str, Any]],
    *,
    semantic_decisions_by_domain: Mapping[
        str, Mapping[DecisionKey, Mapping[str, Any]]
    ],
    bound_loss_proposer_report: Mapping[str, Any],
    bound_inputs_verified: bool,
    feature_mode: str = "hybrid-context-semantic",
    n_folds: int = DUAL_UNION_FOLDS,
    bootstrap_resamples: int = DUAL_UNION_BOOTSTRAP_RESAMPLES,
    seed: int = DUAL_UNION_SEED,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if not bound_inputs_verified:
        raise ValueError("dual-union fitting requires hash-bound inputs")
    if feature_mode != "hybrid-context-semantic":
        raise ValueError("dual-union protocol requires hybrid-context-semantic")
    if n_folds != DUAL_UNION_FOLDS or seed != DUAL_UNION_SEED:
        raise ValueError("dual-union folds and seed are frozen")
    if bootstrap_resamples != DUAL_UNION_BOOTSTRAP_RESAMPLES:
        raise ValueError("dual-union bootstrap count is frozen")
    proposer_audit = validate_bound_loss_proposer_report(bound_loss_proposer_report)

    domain_by_key, baselines, zooms = _validate_domains(records_by_domain)
    if (
        len(set(record.source_id for record in baselines.values())) != 3500
        or len(baselines) != 13580
        or any(len(actions) != 4 for actions in zooms.values())
    ):
        raise ValueError("dual-union population contract no longer matches")
    semantic_by_key = _semantic_feature_index(
        feature_mode=feature_mode,
        records_by_domain=records_by_domain,
        domain_by_key=domain_by_key,
        semantic_decisions_by_domain=semantic_decisions_by_domain,
    )
    keys = sorted(baselines)
    loss_actions, _loss_scores = _prediction_index(prediction_rows, set(keys))
    for row in prediction_rows:
        key = (str(row.get("state_id", "")), str(row.get("replicate_id", "")))
        if str(row.get("source_id", "")) != baselines[key].source_id:
            raise ValueError("dual-union loss-proposer source identity differs")
    incumbent_actions, incumbent_scores, audited_incumbent_calls = _audited_incumbent_index(
        audited_score_rows, baselines=baselines, loss_actions=loss_actions
    )
    union, union_counts = _union_actions(
        keys, incumbent_actions=incumbent_actions, loss_actions=loss_actions
    )
    expected_counts = {
        "equal_proposal_pairs": DUAL_UNION_EQUAL_PAIRS,
        "unequal_proposal_pairs": DUAL_UNION_UNEQUAL_PAIRS,
        "unique_union_rows": DUAL_UNION_ROWS,
    }
    if union_counts != expected_counts:
        raise ValueError(f"dual-union cardinality changed: {union_counts}")

    state_features_by_key: dict[DecisionKey, np.ndarray] = {}
    actions_by_pair: dict[CandidateKey, ActionRecord] = {}
    action_features_by_pair: dict[CandidateKey, np.ndarray] = {}
    for key in keys:
        state = np.asarray(
            _state_features(
                baselines[key], feature_mode=feature_mode, semantic_decision=semantic_by_key.get(key)
            ),
            dtype=np.float64,
        )
        if state.shape != (FACTORIZED_CONDITIONED_STATE_FEATURE_COUNT,) or not np.isfinite(state).all():
            raise ValueError("dual-union state feature schema is invalid")
        state_features_by_key[key] = state
        selected, features = _selected_action_features(
            [key],
            actions={key: action_id for action_id in union[key]},
            baselines=baselines,
            zooms=zooms,
            semantic_by_key=semantic_by_key,
            feature_mode=feature_mode,
        ) if len(union[key]) == 1 else ({}, {})
        if len(union[key]) == 1:
            action_id = union[key][0]
            actions_by_pair[(key, action_id)] = selected[key]
            action_features_by_pair[(key, action_id)] = features[key]
        else:
            for action_id in union[key]:
                selected_one, features_one = _selected_action_features(
                    [key],
                    actions={key: action_id},
                    baselines=baselines,
                    zooms=zooms,
                    semantic_by_key=semantic_by_key,
                    feature_mode=feature_mode,
                )
                actions_by_pair[(key, action_id)] = selected_one[key]
                action_features_by_pair[(key, action_id)] = features_one[key]
    if len(actions_by_pair) != DUAL_UNION_ROWS or len(action_features_by_pair) != DUAL_UNION_ROWS:
        raise RuntimeError("dual-union feature coverage is incomplete")

    fold_by_key, fold_source_counts = _source_folds(
        domain_by_key, baselines, n_folds=n_folds, seed=seed
    )
    candidate_actions: dict[DecisionKey, str] = {}
    candidate_scores: dict[DecisionKey, float] = {}
    probabilities: dict[str, dict[DecisionKey, float]] = {
        "error": {}, "rescue": {}, "harm": {}
    }
    fold_training: list[dict[str, Any]] = []
    serialized_folds: list[dict[str, Any]] = []
    for fold in range(n_folds):
        train_keys = [key for key in keys if fold_by_key[key] != fold]
        test_keys = [key for key in keys if fold_by_key[key] == fold]
        exclusion = _assert_source_exclusion(
            train_keys, test_keys, baselines=baselines, domain_by_key=domain_by_key
        )
        heads, training_audit = _fit_union_heads(
            train_keys,
            union=union,
            state_features_by_key=state_features_by_key,
            action_features_by_pair=action_features_by_pair,
            actions_by_pair=actions_by_pair,
            baselines=baselines,
            domain_by_key=domain_by_key,
            seed=seed + fold,
        )
        for key in test_keys:
            action_id, score, error_p, rescue_p, harm_p = _score_union_key(
                heads,
                key,
                union=union,
                state_features_by_key=state_features_by_key,
                action_features_by_pair=action_features_by_pair,
                actions_by_pair=actions_by_pair,
            )
            if key in candidate_scores:
                raise RuntimeError("dual-union OOF decision scored twice")
            candidate_actions[key] = action_id
            candidate_scores[key] = score
            probabilities["error"][key] = error_p
            probabilities["rescue"][key] = rescue_p
            probabilities["harm"][key] = harm_p
        fold_record = {
            "fold": fold,
            "test_decisions": len(test_keys),
            **exclusion,
            **training_audit,
        }
        fold_training.append(fold_record)
        serialized_folds.append(
            {"fold": fold, "heads": _serialize_factorized_heads(heads, training_audit)}
        )
    if set(candidate_actions) != set(keys) or set(candidate_scores) != set(keys):
        raise RuntimeError("dual-union OOF scoring is incomplete")

    full_heads, full_audit = _fit_union_heads(
        keys,
        union=union,
        state_features_by_key=state_features_by_key,
        action_features_by_pair=action_features_by_pair,
        actions_by_pair=actions_by_pair,
        baselines=baselines,
        domain_by_key=domain_by_key,
        seed=seed,
    )
    incumbent_match = match_call_count_threshold(
        incumbent_scores, target_calls=DUAL_UNION_TARGET_CALLS
    )
    candidate_match = match_call_count_threshold(
        candidate_scores, target_calls=DUAL_UNION_TARGET_CALLS
    )
    incumbent_call_keys = {
        key for key, score in incumbent_scores.items() if score >= float(incumbent_match["threshold"])
    }
    audited_call_keys = {key for key, called in audited_incumbent_calls.items() if called}
    if (
        incumbent_match["calls"] != DUAL_UNION_TARGET_CALLS
        or candidate_match["calls"] != DUAL_UNION_TARGET_CALLS
        or incumbent_call_keys != audited_call_keys
    ):
        raise ValueError("dual-union matched calls do not reproduce incumbent")
    evaluated = _rename_candidate(
        _evaluate(
            baselines=baselines,
            zooms=zooms,
            actions_by_method={"incumbent": incumbent_actions, "decoupled": candidate_actions},
            scores_by_method={"incumbent": incumbent_scores, "decoupled": candidate_scores},
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
        raise ValueError("dual-union incumbent pooled metrics no longer reproduce")

    output_rows: list[dict[str, Any]] = []
    for row in evaluated["score_rows"]:
        key = (str(row["state_id"]), str(row["replicate_id"]))
        row["incumbent_proposal_action_id"] = incumbent_actions[key]
        row["loss_proposal_action_id"] = loss_actions[key]
        row["union_candidate_count"] = len(union[key])
        for name in ("error", "rescue", "harm"):
            row[f"dual_proposer_union_{name}_probability"] = probabilities[name][key]
        if _FORBIDDEN_OUTPUT_FIELDS.intersection(row):
            raise RuntimeError("dual-union output scores leak outcomes")
        output_rows.append(row)

    source_points = evaluated["source_balanced"]
    incumbent = source_points["incumbent"]
    candidate = source_points["dual_proposer_union"]
    primary = evaluated["primary_estimand"]
    weight_audit_passed = all(
        _weight_mass_matches_rows(float(fold[head]["weight_mass"]), int(fold[head]["train_rows"]))
        and fold[head]["class_balancing"] is False
        for fold in fold_training
        for head in ("error", "rescue", "harm")
    )
    audits = {
        "bound_input_hashes_verified": True,
        "loss_proposer_report_validated": True,
        "proposal_inputs_outcome_free": True,
        "union_cardinality_exact": union_counts == expected_counts,
        "state_feature_count_exact": all(fold["state_feature_count"] == FACTORIZED_CONDITIONED_STATE_FEATURE_COUNT for fold in fold_training),
        "action_feature_count_exact": all(fold["action_feature_count"] == FACTORIZED_CONDITIONED_ACTION_FEATURE_COUNT for fold in fold_training),
        "gate_source_exclusion": all(fold["source_exclusion_passed"] for fold in fold_training),
        "candidate_weighting_exact": weight_audit_passed,
        "oof_score_coverage_exact": len(candidate_scores) == len(keys),
        "matched_call_counts_exact": incumbent_match["calls"] == candidate_match["calls"] == DUAL_UNION_TARGET_CALLS,
        "incumbent_call_set_reproduced": incumbent_call_keys == audited_call_keys,
        "incumbent_pooled_metrics_reproduced": True,
        "serialized_scores_finite": all(math.isfinite(value) for value in candidate_scores.values()),
        "serialized_scores_outcome_free": True,
        "screenqa_inputs_used": False,
        "protected_role_inputs_used": False,
    }
    all_audits_passed = all(
        value is True for key, value in audits.items()
        if key not in {"screenqa_inputs_used", "protected_role_inputs_used"}
    ) and not audits["screenqa_inputs_used"] and not audits["protected_role_inputs_used"]
    pass_rule = {
        "utility_margin_at_least_0_00025": float(candidate["utility"]) >= float(incumbent["utility"]) + 0.00025,
        "paired_ci_low_above_minus_0_0005": float(primary["ci_low"]) > -0.0005,
        "gain_per_call_higher": float(candidate["gain_per_call"]) > float(incumbent["gain_per_call"]),
        "harm_and_negative_calls_no_greater": float(candidate["induced_harm"]) <= float(incumbent["induced_harm"])
        and float(candidate["negative_value_call"]) <= float(incumbent["negative_value_call"]),
        "helpful_call_precision_no_lower": float(candidate["helpful_call_precision"]) >= float(incumbent["helpful_call_precision"]),
        "all_audits_passed": all_audits_passed,
    }
    score_report = {
        "scientific_status": "outcome-blind dual-proposer-union OOF scores frozen before opened-development outcome evaluation",
        "n_sources": 3500,
        "n_decisions": len(keys),
        "feature_mode": feature_mode,
        "state_feature_count": FACTORIZED_CONDITIONED_STATE_FEATURE_COUNT,
        "action_feature_count": FACTORIZED_CONDITIONED_ACTION_FEATURE_COUNT,
        "union_counts": union_counts,
        "n_folds": n_folds,
        "fold_source_counts": fold_source_counts,
        "fold_training": fold_training,
        "incumbent_match": incumbent_match,
        "dual_proposer_union_match": candidate_match,
        "task_outcomes_used_for_thresholds": False,
        "serialized_outcome_fields": [],
        "audits": audits,
        "proposer_audit": proposer_audit,
    }
    report = {
        "scientific_status": "opened DocVQA development diagnostic; not independent validation",
        "decision": "dual_proposer_union_factorized_gate_advanced" if all(pass_rule.values()) else "dual_proposer_union_factorized_gate_not_advanced",
        "pass_rule": pass_rule,
        "n_sources": 3500,
        "n_decisions": len(keys),
        "lambda_cost": FACTORIZED_CONDITIONED_LAMBDA_COST,
        "source_balanced": source_points,
        "question_balanced": evaluated["question_balanced"],
        "primary_estimand": primary,
        "paired_source_bootstrap": evaluated["paired_source_bootstrap"],
        "action_disagreement_rate": evaluated["action_disagreement_rate"],
        "gate_disagreement_rate": evaluated["gate_disagreement_rate"],
        "union_counts": union_counts,
        "audits": audits,
        "screenqa_inputs_used": False,
        "protected_role_inputs_used": False,
    }
    model = {
        "schema": "dual_proposer_union_factorized_gate_v1",
        "feature_mode": feature_mode,
        "state_feature_count": FACTORIZED_CONDITIONED_STATE_FEATURE_COUNT,
        "action_feature_count": FACTORIZED_CONDITIONED_ACTION_FEATURE_COUNT,
        "score": "p_error*p_rescue*mu_rescue-(1-p_error)*p_harm*mu_harm-0.05*cost",
        "seed": seed,
        "n_folds": n_folds,
        "C": FACTORIZED_CONDITIONED_C,
        "penalty": "l2",
        "solver": "liblinear",
        "max_iter": FACTORIZED_CONDITIONED_MAX_ITER,
        "union_counts": union_counts,
        "oof_folds": serialized_folds,
        "full_refit": _serialize_factorized_heads(full_heads, full_audit),
        "deployment_composition": "frozen incumbent and loss-only full proposers followed by this union scorer",
        "screenqa_inputs_used": False,
    }
    return report, score_report, model, output_rows
