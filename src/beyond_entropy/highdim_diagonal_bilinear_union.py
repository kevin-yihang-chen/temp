from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np  # type: ignore[import-not-found]

from .action_value import _semantic_feature_index, _validate_domains
from .decoupled_loss_gate import (
    INCUMBENT_POOLED_CALL_RATE,
    INCUMBENT_POOLED_GAIN,
    INCUMBENT_POOLED_UTILITY,
    _evaluate,
    _prediction_index,
    match_call_count_threshold,
)
from .dual_proposer_union_factorized_gate import (
    DUAL_UNION_EQUAL_PAIRS,
    DUAL_UNION_ROWS,
    DUAL_UNION_UNEQUAL_PAIRS,
    CandidateKey,
    _candidate_balanced_weights,
    _union_actions,
)
from .oof_action_value import _domain_source_balanced_weights, _source_folds
from .proposal_conditioned_factorized_gate import _weight_mass_matches_rows
from .proposal_conditioned_gate import (
    _assert_source_exclusion,
    _audited_incumbent_index,
    _selected_action_features,
    validate_bound_loss_proposer_report,
)
from .rescue_gate import DecisionKey, pre_action_context_features
from .schema import ActionRecord


HIGHDIM_SEED = 20260909
HIGHDIM_FOLDS = 5
HIGHDIM_C = 0.01
HIGHDIM_MAX_ITER = 4000
HIGHDIM_EMBEDDING_DIM = 2048
HIGHDIM_STATE_FEATURE_COUNT = 2075
HIGHDIM_ACTION_FEATURE_COUNT = 4142
HIGHDIM_TARGET_CALLS = 225
HIGHDIM_LAMBDA_COST = 0.05
HIGHDIM_BOOTSTRAP_RESAMPLES = 20000


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


def _as_finite_vector(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"highdim {name} must be finite with shape {shape}")
    return result


def _l2_normalize(vector: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"highdim {name} has zero or invalid norm")
    result = vector / norm
    if not math.isclose(float(np.linalg.norm(result)), 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise RuntimeError(f"highdim {name} normalization is unstable")
    return result


def _highdim_features(
    baseline: ActionRecord,
    action: ActionRecord,
    semantic_decision: Mapping[str, Any],
    compact_action_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    action_ids = [str(value) for value in semantic_decision.get("action_ids", [])]
    if len(action_ids) != 4 or len(set(action_ids)) != 4:
        raise ValueError("highdim semantic action IDs must contain four unique crops")
    try:
        action_index = action_ids.index(action.action_id)
    except ValueError as exc:
        raise ValueError("highdim selected action is missing from semantic rows") from exc
    question = _l2_normalize(
        _as_finite_vector(
            semantic_decision.get("question_embedding"),
            (HIGHDIM_EMBEDDING_DIM,),
            "question embedding",
        ),
        "question embedding",
    )
    global_visual = _l2_normalize(
        _as_finite_vector(
            semantic_decision.get("global_visual_embedding"),
            (HIGHDIM_EMBEDDING_DIM,),
            "global visual embedding",
        ),
        "global visual embedding",
    )
    regions = _as_finite_vector(
        semantic_decision.get("region_embeddings"),
        (4, HIGHDIM_EMBEDDING_DIM),
        "region embeddings",
    )
    region = _l2_normalize(regions[action_index], "region embedding")
    context = np.asarray(pre_action_context_features(baseline), dtype=np.float64)
    if context.shape != (27,) or compact_action_features.shape != (46,):
        raise ValueError("highdim compact feature schemas are invalid")
    state = np.concatenate((context, question * global_visual))
    action_features = np.concatenate(
        (compact_action_features, question * region, global_visual * region)
    )
    if (
        state.shape != (HIGHDIM_STATE_FEATURE_COUNT,)
        or action_features.shape != (HIGHDIM_ACTION_FEATURE_COUNT,)
        or not np.isfinite(state).all()
        or not np.isfinite(action_features).all()
    ):
        raise RuntimeError("highdim diagonal-bilinear feature construction failed")
    return state, action_features


def _fit_highdim_head(
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
        raise ValueError("highdim head rows are not aligned")
    if not np.isfinite(features).all() or not np.isfinite(weights).all():
        raise ValueError("highdim head inputs must be finite")
    if not _weight_mass_matches_rows(float(weights.sum()), len(weights)):
        raise ValueError("highdim head weight mass is not row-normalized")
    label_array = np.asarray(labels, dtype=np.int64)
    if set(label_array.tolist()) != {0, 1}:
        raise ValueError("highdim head requires both classes")
    scaler = StandardScaler().fit(features)
    model = LogisticRegression(
        C=HIGHDIM_C,
        penalty="l2",
        solver="liblinear",
        max_iter=HIGHDIM_MAX_ITER,
        random_state=seed,
    ).fit(scaler.transform(features), label_array, sample_weight=weights)
    if int(model.n_iter_[0]) >= HIGHDIM_MAX_ITER:
        raise RuntimeError("highdim logistic head did not converge")
    return (
        {"scaler": scaler, "model": model},
        {
            "train_rows": int(features.shape[0]),
            "feature_count": int(features.shape[1]),
            "negative_rows": int(np.sum(label_array == 0)),
            "positive_rows": int(np.sum(label_array == 1)),
            "weight_mass": float(weights.sum()),
            "iterations": int(model.n_iter_[0]),
            "C": HIGHDIM_C,
            "class_balancing": False,
        },
    )


def _fit_highdim_heads(
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
    error_weights = np.asarray(
        _domain_source_balanced_weights(
            [domain_by_key[key] for key in train_keys],
            [baselines[key].source_id for key in train_keys],
        ),
        dtype=np.float64,
    )
    error_head, error_audit = _fit_highdim_head(
        states, error_labels, error_weights, seed=seed
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
    rescue_head, rescue_audit = _fit_highdim_head(
        np.stack([action_features_by_pair[pair] for pair in rescue_pairs]),
        rescue_labels,
        rescue_weights,
        seed=seed,
    )
    harm_head, harm_audit = _fit_highdim_head(
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
                [actions_by_pair[pair].delta_success for pair in rescue_pairs]
            )[rescue_positive],
            weights=rescue_weights[rescue_positive],
        )
    )
    harm_magnitude = float(
        np.average(
            -np.asarray(
                [actions_by_pair[pair].delta_success for pair in harm_pairs]
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
        "action_feature_count": HIGHDIM_ACTION_FEATURE_COUNT,
        "error": error_audit,
        "rescue": rescue_audit,
        "harm": harm_audit,
        "rescue_magnitude": rescue_magnitude,
        "harm_magnitude": harm_magnitude,
    }
    return heads, audit


def _score_highdim_key(
    heads: Mapping[str, Any],
    key: DecisionKey,
    *,
    union: Mapping[DecisionKey, Sequence[str]],
    state_features_by_key: Mapping[DecisionKey, np.ndarray],
    action_features_by_pair: Mapping[CandidateKey, np.ndarray],
    actions_by_pair: Mapping[CandidateKey, ActionRecord],
) -> tuple[str, float, float, float, float]:
    error_p = float(
        heads["error"]["model"].predict_proba(
            heads["error"]["scaler"].transform(state_features_by_key[key][None, :])
        )[0, 1]
    )
    candidates = []
    for action_id in union[key]:
        pair = (key, action_id)
        features = action_features_by_pair[pair][None, :]
        rescue_p = float(
            heads["rescue"]["model"].predict_proba(
                heads["rescue"]["scaler"].transform(features)
            )[0, 1]
        )
        harm_p = float(
            heads["harm"]["model"].predict_proba(
                heads["harm"]["scaler"].transform(features)
            )[0, 1]
        )
        score = (
            error_p * rescue_p * float(heads["rescue_magnitude"])
            - (1.0 - error_p) * harm_p * float(heads["harm_magnitude"])
            - HIGHDIM_LAMBDA_COST * actions_by_pair[pair].tool_cost
        )
        if not all(math.isfinite(value) for value in (error_p, rescue_p, harm_p, score)):
            raise RuntimeError("highdim scorer produced non-finite values")
        candidates.append((score, action_id, rescue_p, harm_p))
    score, action_id, rescue_p, harm_p = min(
        candidates, key=lambda item: (-item[0], item[1])
    )
    return action_id, score, error_p, rescue_p, harm_p


def _serialize_heads(heads: Mapping[str, Any], audit: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "training_audit": dict(audit),
        "rescue_magnitude": float(heads["rescue_magnitude"]),
        "harm_magnitude": float(heads["harm_magnitude"]),
    }
    for name in ("error", "rescue", "harm"):
        scaler = heads[name]["scaler"]
        model = heads[name]["model"]
        result[name] = {
            "scaler_mean": [float(value) for value in scaler.mean_.tolist()],
            "scaler_scale": [float(value) for value in scaler.scale_.tolist()],
            "coefficient": [float(value) for value in model.coef_[0].tolist()],
            "intercept": float(model.intercept_[0]),
            "classes": [int(value) for value in model.classes_.tolist()],
        }
    return result


def _rename_candidate(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key).replace("decoupled", "highdim_diagonal_bilinear"): _rename_candidate(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rename_candidate(item) for item in value]
    return value


def evaluate_highdim_diagonal_bilinear_union(
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
    n_folds: int = HIGHDIM_FOLDS,
    bootstrap_resamples: int = HIGHDIM_BOOTSTRAP_RESAMPLES,
    seed: int = HIGHDIM_SEED,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if not bound_inputs_verified:
        raise ValueError("highdim fitting requires hash-bound inputs")
    if feature_mode != "hybrid-context-semantic":
        raise ValueError("highdim protocol requires hybrid-context-semantic")
    if n_folds != HIGHDIM_FOLDS or seed != HIGHDIM_SEED:
        raise ValueError("highdim folds and seed are frozen")
    if bootstrap_resamples != HIGHDIM_BOOTSTRAP_RESAMPLES:
        raise ValueError("highdim bootstrap count is frozen")
    proposer_audit = validate_bound_loss_proposer_report(bound_loss_proposer_report)
    domain_by_key, baselines, zooms = _validate_domains(records_by_domain)
    if (
        len(set(record.source_id for record in baselines.values())) != 3500
        or len(baselines) != 13580
        or any(len(actions) != 4 for actions in zooms.values())
    ):
        raise ValueError("highdim population contract no longer matches")
    semantic_by_key = _semantic_feature_index(
        feature_mode=feature_mode,
        records_by_domain=records_by_domain,
        domain_by_key=domain_by_key,
        semantic_decisions_by_domain=semantic_decisions_by_domain,
    )
    keys = sorted(baselines)
    loss_actions, _loss_scores = _prediction_index(prediction_rows, set(keys))
    incumbent_actions, incumbent_scores, audited_incumbent_calls = _audited_incumbent_index(
        audited_score_rows, baselines=baselines, loss_actions=loss_actions
    )
    union, union_counts = _union_actions(
        keys, incumbent_actions=incumbent_actions, loss_actions=loss_actions
    )
    if union_counts != {
        "equal_proposal_pairs": DUAL_UNION_EQUAL_PAIRS,
        "unequal_proposal_pairs": DUAL_UNION_UNEQUAL_PAIRS,
        "unique_union_rows": DUAL_UNION_ROWS,
    }:
        raise ValueError("highdim union cardinality no longer matches")

    state_features_by_key: dict[DecisionKey, np.ndarray] = {}
    action_features_by_pair: dict[CandidateKey, np.ndarray] = {}
    actions_by_pair: dict[CandidateKey, ActionRecord] = {}
    for key in keys:
        for action_id in union[key]:
            selected, compact = _selected_action_features(
                [key],
                actions={key: action_id},
                baselines=baselines,
                zooms=zooms,
                semantic_by_key=semantic_by_key,
                feature_mode=feature_mode,
            )
            state, action_features = _highdim_features(
                baselines[key], selected[key], semantic_by_key[key], compact[key]
            )
            prior_state = state_features_by_key.setdefault(key, state)
            if not np.array_equal(prior_state, state):
                raise RuntimeError("highdim state features differ across candidates")
            pair = (key, action_id)
            actions_by_pair[pair] = selected[key]
            action_features_by_pair[pair] = action_features
    if len(actions_by_pair) != DUAL_UNION_ROWS:
        raise RuntimeError("highdim union feature coverage is incomplete")

    fold_by_key, fold_source_counts = _source_folds(
        domain_by_key, baselines, n_folds=n_folds, seed=seed
    )
    candidate_actions: dict[DecisionKey, str] = {}
    candidate_scores: dict[DecisionKey, float] = {}
    probabilities: dict[str, dict[DecisionKey, float]] = {
        "error": {}, "rescue": {}, "harm": {}
    }
    fold_training = []
    serialized_folds = []
    for fold in range(n_folds):
        train_keys = [key for key in keys if fold_by_key[key] != fold]
        test_keys = [key for key in keys if fold_by_key[key] == fold]
        exclusion = _assert_source_exclusion(
            train_keys, test_keys, baselines=baselines, domain_by_key=domain_by_key
        )
        heads, training_audit = _fit_highdim_heads(
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
            action_id, score, error_p, rescue_p, harm_p = _score_highdim_key(
                heads,
                key,
                union=union,
                state_features_by_key=state_features_by_key,
                action_features_by_pair=action_features_by_pair,
                actions_by_pair=actions_by_pair,
            )
            candidate_actions[key] = action_id
            candidate_scores[key] = score
            probabilities["error"][key] = error_p
            probabilities["rescue"][key] = rescue_p
            probabilities["harm"][key] = harm_p
        fold_record = {"fold": fold, "test_decisions": len(test_keys), **exclusion, **training_audit}
        fold_training.append(fold_record)
        serialized_folds.append({"fold": fold, "heads": _serialize_heads(heads, training_audit)})
    if set(candidate_scores) != set(keys) or set(candidate_actions) != set(keys):
        raise RuntimeError("highdim OOF scoring is incomplete")
    full_heads, full_audit = _fit_highdim_heads(
        keys,
        union=union,
        state_features_by_key=state_features_by_key,
        action_features_by_pair=action_features_by_pair,
        actions_by_pair=actions_by_pair,
        baselines=baselines,
        domain_by_key=domain_by_key,
        seed=seed,
    )

    incumbent_match = match_call_count_threshold(incumbent_scores, target_calls=HIGHDIM_TARGET_CALLS)
    candidate_match = match_call_count_threshold(candidate_scores, target_calls=HIGHDIM_TARGET_CALLS)
    incumbent_call_keys = {key for key, score in incumbent_scores.items() if score >= float(incumbent_match["threshold"])}
    audited_call_keys = {key for key, called in audited_incumbent_calls.items() if called}
    if (
        incumbent_match["calls"] != HIGHDIM_TARGET_CALLS
        or candidate_match["calls"] != HIGHDIM_TARGET_CALLS
        or incumbent_call_keys != audited_call_keys
    ):
        raise ValueError("highdim matched calls do not reproduce incumbent")
    evaluated = _rename_candidate(
        _evaluate(
            baselines=baselines,
            zooms=zooms,
            actions_by_method={"incumbent": incumbent_actions, "decoupled": candidate_actions},
            scores_by_method={"incumbent": incumbent_scores, "decoupled": candidate_scores},
            threshold_by_method={"incumbent": float(incumbent_match["threshold"]), "decoupled": float(candidate_match["threshold"])},
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
        raise ValueError("highdim incumbent pooled metrics no longer reproduce")
    output_rows = []
    for row in evaluated["score_rows"]:
        key = (str(row["state_id"]), str(row["replicate_id"]))
        row["incumbent_proposal_action_id"] = incumbent_actions[key]
        row["loss_proposal_action_id"] = loss_actions[key]
        row["union_candidate_count"] = len(union[key])
        for name in ("error", "rescue", "harm"):
            row[f"highdim_diagonal_bilinear_{name}_probability"] = probabilities[name][key]
        if _FORBIDDEN_OUTPUT_FIELDS.intersection(row):
            raise RuntimeError("highdim output scores leak outcomes")
        output_rows.append(row)

    source_points = evaluated["source_balanced"]
    incumbent = source_points["incumbent"]
    candidate = source_points["highdim_diagonal_bilinear"]
    primary = evaluated["primary_estimand"]
    audits = {
        "bound_input_hashes_verified": True,
        "loss_proposer_report_validated": True,
        "proposal_inputs_outcome_free": True,
        "embedding_dimensions_exact": True,
        "embedding_action_alignment_exact": True,
        "union_cardinality_exact": True,
        "state_feature_count_exact": all(fold["state_feature_count"] == HIGHDIM_STATE_FEATURE_COUNT for fold in fold_training),
        "action_feature_count_exact": all(fold["action_feature_count"] == HIGHDIM_ACTION_FEATURE_COUNT for fold in fold_training),
        "gate_source_exclusion": all(fold["source_exclusion_passed"] for fold in fold_training),
        "source_candidate_weighting_exact": all(
            _weight_mass_matches_rows(float(fold[head]["weight_mass"]), int(fold[head]["train_rows"]))
            and fold[head]["class_balancing"] is False
            for fold in fold_training for head in ("error", "rescue", "harm")
        ),
        "oof_score_coverage_exact": len(candidate_scores) == len(keys),
        "matched_call_counts_exact": incumbent_match["calls"] == candidate_match["calls"] == HIGHDIM_TARGET_CALLS,
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
        "scientific_status": "outcome-blind highdim diagonal-bilinear OOF scores frozen before opened-development outcome evaluation",
        "n_sources": 3500,
        "n_decisions": len(keys),
        "embedding_dim": HIGHDIM_EMBEDDING_DIM,
        "state_feature_count": HIGHDIM_STATE_FEATURE_COUNT,
        "action_feature_count": HIGHDIM_ACTION_FEATURE_COUNT,
        "union_counts": union_counts,
        "n_folds": n_folds,
        "fold_source_counts": fold_source_counts,
        "fold_training": fold_training,
        "incumbent_match": incumbent_match,
        "highdim_diagonal_bilinear_match": candidate_match,
        "task_outcomes_used_for_thresholds": False,
        "serialized_outcome_fields": [],
        "audits": audits,
        "proposer_audit": proposer_audit,
    }
    report = {
        "scientific_status": "opened DocVQA development diagnostic; not independent validation",
        "decision": "highdim_diagonal_bilinear_union_advanced" if all(pass_rule.values()) else "highdim_diagonal_bilinear_union_not_advanced",
        "pass_rule": pass_rule,
        "n_sources": 3500,
        "n_decisions": len(keys),
        "lambda_cost": HIGHDIM_LAMBDA_COST,
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
        "schema": "highdim_diagonal_bilinear_union_v1",
        "embedding_dim": HIGHDIM_EMBEDDING_DIM,
        "state_feature_count": HIGHDIM_STATE_FEATURE_COUNT,
        "action_feature_count": HIGHDIM_ACTION_FEATURE_COUNT,
        "seed": seed,
        "n_folds": n_folds,
        "C": HIGHDIM_C,
        "penalty": "l2",
        "solver": "liblinear",
        "max_iter": HIGHDIM_MAX_ITER,
        "union_counts": union_counts,
        "oof_folds": serialized_folds,
        "full_refit": _serialize_heads(full_heads, full_audit),
        "deployment_composition": "frozen incumbent/loss proposers plus highdim diagonal-bilinear union scorer",
        "screenqa_inputs_used": False,
    }
    return report, score_report, model, output_rows
