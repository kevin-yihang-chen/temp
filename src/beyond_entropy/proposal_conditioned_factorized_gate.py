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
from .oof_action_value import _domain_source_balanced_weights, _source_folds
from .proposal_conditioned_gate import (
    PROPOSAL_CONDITIONED_FEATURE_COUNT,
    _assert_source_exclusion,
    _audited_incumbent_index,
    _selected_action_features,
    validate_bound_loss_proposer_report,
)
from .rescue_gate import DecisionKey
from .schema import ActionRecord


FACTORIZED_CONDITIONED_SEED = 20260907
FACTORIZED_CONDITIONED_FOLDS = 5
FACTORIZED_CONDITIONED_C = 1.0
FACTORIZED_CONDITIONED_MAX_ITER = 2000
FACTORIZED_CONDITIONED_STATE_FEATURE_COUNT = 27
FACTORIZED_CONDITIONED_ACTION_FEATURE_COUNT = PROPOSAL_CONDITIONED_FEATURE_COUNT
FACTORIZED_CONDITIONED_TARGET_CALLS = 225
FACTORIZED_CONDITIONED_LAMBDA_COST = 0.05
FACTORIZED_CONDITIONED_BOOTSTRAP_RESAMPLES = 20000


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


def _fit_binary_head(
    features: np.ndarray,
    labels: Sequence[int],
    domains: Sequence[str],
    sources: Sequence[str],
    *,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("factorized conditioned head requires a non-empty matrix")
    n_rows = features.shape[0]
    if len(labels) != n_rows or len(domains) != n_rows or len(sources) != n_rows:
        raise ValueError("factorized conditioned head rows are not aligned")
    if not np.isfinite(features).all():
        raise ValueError("factorized conditioned head features must be finite")
    label_array = np.asarray(labels, dtype=np.int64)
    if set(label_array.tolist()) != {0, 1}:
        raise ValueError("factorized conditioned head requires both classes")
    weights = np.asarray(
        _domain_source_balanced_weights(domains, sources), dtype=np.float64
    )
    if not math.isclose(float(weights.sum()), n_rows, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError("source-balanced head weights are not row-normalized")
    scaler = StandardScaler().fit(features)
    model = LogisticRegression(
        C=FACTORIZED_CONDITIONED_C,
        penalty="l2",
        solver="liblinear",
        max_iter=FACTORIZED_CONDITIONED_MAX_ITER,
        random_state=seed,
    ).fit(
        scaler.transform(features),
        label_array,
        sample_weight=weights,
    )
    if int(model.n_iter_[0]) >= FACTORIZED_CONDITIONED_MAX_ITER:
        raise RuntimeError("factorized conditioned logistic head did not converge")
    return (
        {"scaler": scaler, "model": model},
        weights,
        {
            "train_rows": n_rows,
            "feature_count": int(features.shape[1]),
            "negative_rows": int(np.sum(label_array == 0)),
            "positive_rows": int(np.sum(label_array == 1)),
            "weight_mass": float(weights.sum()),
            "iterations": int(model.n_iter_[0]),
            "weighting": "equal_domain_then_source_then_row",
            "class_balancing": False,
        },
    )


def _fit_factorized_conditioned_heads(
    state_features: np.ndarray,
    action_features: np.ndarray,
    error_labels: Sequence[int],
    deltas: Sequence[float],
    domains: Sequence[str],
    sources: Sequence[str],
    *,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    n_rows = state_features.shape[0]
    if (
        state_features.shape != (n_rows, FACTORIZED_CONDITIONED_STATE_FEATURE_COUNT)
        or action_features.shape
        != (n_rows, FACTORIZED_CONDITIONED_ACTION_FEATURE_COUNT)
        or len(error_labels) != n_rows
        or len(deltas) != n_rows
        or len(domains) != n_rows
        or len(sources) != n_rows
    ):
        raise ValueError("factorized conditioned training arrays are not aligned")
    if not np.isfinite(state_features).all() or not np.isfinite(action_features).all():
        raise ValueError("factorized conditioned features must be finite")
    delta_array = np.asarray(deltas, dtype=np.float64)
    if not np.isfinite(delta_array).all():
        raise ValueError("factorized conditioned training deltas must be finite")
    error_array = np.asarray(error_labels, dtype=np.int64)
    if set(error_array.tolist()) != {0, 1}:
        raise ValueError("factorized conditioned error head requires both classes")
    rescue_mask = error_array == 1
    harm_mask = error_array == 0
    rescue_labels = (delta_array[rescue_mask] > 0.0).astype(np.int64)
    harm_labels = (delta_array[harm_mask] < 0.0).astype(np.int64)

    error_head, _error_weights, error_audit = _fit_binary_head(
        state_features,
        error_array.tolist(),
        domains,
        sources,
        seed=seed,
    )
    rescue_head, rescue_weights, rescue_audit = _fit_binary_head(
        action_features[rescue_mask],
        rescue_labels.tolist(),
        [domains[index] for index in np.flatnonzero(rescue_mask)],
        [sources[index] for index in np.flatnonzero(rescue_mask)],
        seed=seed,
    )
    harm_head, harm_weights, harm_audit = _fit_binary_head(
        action_features[harm_mask],
        harm_labels.tolist(),
        [domains[index] for index in np.flatnonzero(harm_mask)],
        [sources[index] for index in np.flatnonzero(harm_mask)],
        seed=seed,
    )
    rescue_positive = rescue_labels == 1
    harm_positive = harm_labels == 1
    rescue_magnitude = float(
        np.average(
            delta_array[rescue_mask][rescue_positive],
            weights=rescue_weights[rescue_positive],
        )
    )
    harm_magnitude = float(
        np.average(
            -delta_array[harm_mask][harm_positive],
            weights=harm_weights[harm_positive],
        )
    )
    if not (
        math.isfinite(rescue_magnitude)
        and rescue_magnitude > 0.0
        and math.isfinite(harm_magnitude)
        and harm_magnitude > 0.0
    ):
        raise RuntimeError("factorized conditioned magnitudes are invalid")
    heads = {
        "error": error_head,
        "rescue": rescue_head,
        "harm": harm_head,
        "rescue_magnitude": rescue_magnitude,
        "harm_magnitude": harm_magnitude,
    }
    audit = {
        "train_rows": n_rows,
        "state_feature_count": int(state_features.shape[1]),
        "action_feature_count": int(action_features.shape[1]),
        "error": error_audit,
        "rescue": rescue_audit,
        "harm": harm_audit,
        "rescue_magnitude": rescue_magnitude,
        "harm_magnitude": harm_magnitude,
    }
    return heads, audit


def _score_factorized_conditioned_heads(
    heads: Mapping[str, Any],
    state_features: np.ndarray,
    action_features: np.ndarray,
    tool_costs: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_rows = state_features.shape[0]
    if (
        state_features.shape != (n_rows, FACTORIZED_CONDITIONED_STATE_FEATURE_COUNT)
        or action_features.shape
        != (n_rows, FACTORIZED_CONDITIONED_ACTION_FEATURE_COUNT)
        or len(tool_costs) != n_rows
    ):
        raise ValueError("factorized conditioned scoring arrays are not aligned")
    probabilities: dict[str, np.ndarray] = {}
    for name, features in (
        ("error", state_features),
        ("rescue", action_features),
        ("harm", action_features),
    ):
        head = heads[name]
        probabilities[name] = np.asarray(
            head["model"].predict_proba(head["scaler"].transform(features))[:, 1],
            dtype=np.float64,
        )
    costs = np.asarray(tool_costs, dtype=np.float64)
    scores = (
        probabilities["error"]
        * probabilities["rescue"]
        * float(heads["rescue_magnitude"])
        - (1.0 - probabilities["error"])
        * probabilities["harm"]
        * float(heads["harm_magnitude"])
        - FACTORIZED_CONDITIONED_LAMBDA_COST * costs
    )
    if not all(np.isfinite(values).all() for values in (*probabilities.values(), scores)):
        raise RuntimeError("factorized conditioned scoring produced non-finite values")
    return probabilities["error"], probabilities["rescue"], probabilities["harm"], scores


def _serialize_factorized_heads(
    heads: Mapping[str, Any], audit: Mapping[str, Any]
) -> dict[str, Any]:
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
            str(key).replace("decoupled", "proposal_conditioned_factorized"): (
                _rename_candidate(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rename_candidate(item) for item in value]
    return value


def evaluate_proposal_conditioned_factorized_gate(
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
    n_folds: int = FACTORIZED_CONDITIONED_FOLDS,
    bootstrap_resamples: int = FACTORIZED_CONDITIONED_BOOTSTRAP_RESAMPLES,
    seed: int = FACTORIZED_CONDITIONED_SEED,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if not bound_inputs_verified:
        raise ValueError("factorized conditioned fitting requires hash-bound inputs")
    if feature_mode != "hybrid-context-semantic":
        raise ValueError("factorized conditioned protocol requires hybrid features")
    if n_folds != FACTORIZED_CONDITIONED_FOLDS or seed != FACTORIZED_CONDITIONED_SEED:
        raise ValueError("factorized conditioned folds and seed are frozen")
    if bootstrap_resamples != FACTORIZED_CONDITIONED_BOOTSTRAP_RESAMPLES:
        raise ValueError("factorized conditioned bootstrap count is frozen")
    proposer_audit = validate_bound_loss_proposer_report(bound_loss_proposer_report)

    domain_by_key, baselines, zooms = _validate_domains(records_by_domain)
    if (
        len(set(record.source_id for record in baselines.values())) != 3500
        or len(baselines) != 13580
        or any(len(actions) != 4 for actions in zooms.values())
    ):
        raise ValueError("factorized conditioned population contract no longer matches")
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
            raise ValueError("loss-proposer source identity differs from rollouts")
    incumbent_actions, incumbent_scores, audited_incumbent_calls = (
        _audited_incumbent_index(
            audited_score_rows,
            baselines=baselines,
            loss_actions=loss_actions,
        )
    )
    selected_actions, action_features_by_key = _selected_action_features(
        keys,
        actions=loss_actions,
        baselines=baselines,
        zooms=zooms,
        semantic_by_key=semantic_by_key,
        feature_mode=feature_mode,
    )
    state_features_by_key: dict[DecisionKey, np.ndarray] = {}
    for key in keys:
        features = np.asarray(
            _state_features(
                baselines[key],
                feature_mode=feature_mode,
                semantic_decision=semantic_by_key.get(key),
            ),
            dtype=np.float64,
        )
        if features.shape != (FACTORIZED_CONDITIONED_STATE_FEATURE_COUNT,) or not np.isfinite(
            features
        ).all():
            raise ValueError("factorized conditioned state feature schema is invalid")
        state_features_by_key[key] = features

    fold_by_key, fold_source_counts = _source_folds(
        domain_by_key, baselines, n_folds=n_folds, seed=seed
    )
    candidate_scores: dict[DecisionKey, float] = {}
    probabilities: dict[str, dict[DecisionKey, float]] = {
        "error": {},
        "rescue": {},
        "harm": {},
    }
    fold_training: list[dict[str, Any]] = []
    serialized_folds: list[dict[str, Any]] = []
    for fold in range(n_folds):
        train_keys = [key for key in keys if fold_by_key[key] != fold]
        test_keys = [key for key in keys if fold_by_key[key] == fold]
        exclusion = _assert_source_exclusion(
            train_keys,
            test_keys,
            baselines=baselines,
            domain_by_key=domain_by_key,
        )
        train_states = np.stack([state_features_by_key[key] for key in train_keys])
        train_actions = np.stack([action_features_by_key[key] for key in train_keys])
        heads, training_audit = _fit_factorized_conditioned_heads(
            train_states,
            train_actions,
            [int(baselines[key].correct_before < 0.5) for key in train_keys],
            [selected_actions[key].delta_success for key in train_keys],
            [domain_by_key[key] for key in train_keys],
            [baselines[key].source_id for key in train_keys],
            seed=seed + fold,
        )
        test_states = np.stack([state_features_by_key[key] for key in test_keys])
        test_actions = np.stack([action_features_by_key[key] for key in test_keys])
        error_p, rescue_p, harm_p, scores = _score_factorized_conditioned_heads(
            heads,
            test_states,
            test_actions,
            [selected_actions[key].tool_cost for key in test_keys],
        )
        for index, key in enumerate(test_keys):
            if key in candidate_scores:
                raise RuntimeError("factorized conditioned OOF decision scored twice")
            candidate_scores[key] = float(scores[index])
            probabilities["error"][key] = float(error_p[index])
            probabilities["rescue"][key] = float(rescue_p[index])
            probabilities["harm"][key] = float(harm_p[index])
        fold_record = {
            "fold": fold,
            "train_decisions": len(train_keys),
            "test_decisions": len(test_keys),
            **exclusion,
            **training_audit,
        }
        fold_training.append(fold_record)
        serialized_folds.append(
            {
                "fold": fold,
                "heads": _serialize_factorized_heads(heads, training_audit),
            }
        )
    if set(candidate_scores) != set(keys) or any(
        set(values) != set(keys) for values in probabilities.values()
    ):
        raise RuntimeError("factorized conditioned OOF scoring is incomplete")

    all_states = np.stack([state_features_by_key[key] for key in keys])
    all_actions = np.stack([action_features_by_key[key] for key in keys])
    full_heads, full_audit = _fit_factorized_conditioned_heads(
        all_states,
        all_actions,
        [int(baselines[key].correct_before < 0.5) for key in keys],
        [selected_actions[key].delta_success for key in keys],
        [domain_by_key[key] for key in keys],
        [baselines[key].source_id for key in keys],
        seed=seed,
    )

    incumbent_match = match_call_count_threshold(
        incumbent_scores, target_calls=FACTORIZED_CONDITIONED_TARGET_CALLS
    )
    candidate_match = match_call_count_threshold(
        candidate_scores, target_calls=FACTORIZED_CONDITIONED_TARGET_CALLS
    )
    incumbent_call_keys = {
        key
        for key, score in incumbent_scores.items()
        if score >= float(incumbent_match["threshold"])
    }
    audited_call_keys = {key for key, called in audited_incumbent_calls.items() if called}
    if (
        incumbent_match["calls"] != FACTORIZED_CONDITIONED_TARGET_CALLS
        or candidate_match["calls"] != FACTORIZED_CONDITIONED_TARGET_CALLS
        or incumbent_call_keys != audited_call_keys
    ):
        raise ValueError("factorized matched calls do not reproduce incumbent")
    evaluated = _rename_candidate(
        _evaluate(
            baselines=baselines,
            zooms=zooms,
            actions_by_method={"incumbent": incumbent_actions, "decoupled": loss_actions},
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
        raise ValueError("factorized incumbent pooled metrics no longer reproduce")

    output_rows: list[dict[str, Any]] = []
    for row in evaluated["score_rows"]:
        key = (str(row["state_id"]), str(row["replicate_id"]))
        for name in ("error", "rescue", "harm"):
            row[f"proposal_conditioned_factorized_{name}_probability"] = probabilities[name][key]
        if _FORBIDDEN_OUTPUT_FIELDS.intersection(row):
            raise RuntimeError("factorized conditioned output scores leak outcomes")
        output_rows.append(row)

    source_points = evaluated["source_balanced"]
    incumbent = source_points["incumbent"]
    candidate = source_points["proposal_conditioned_factorized"]
    primary = evaluated["primary_estimand"]
    weight_audit_passed = all(
        math.isclose(
            float(fold[head]["weight_mass"]),
            int(fold[head]["train_rows"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        and fold[head]["class_balancing"] is False
        for fold in fold_training
        for head in ("error", "rescue", "harm")
    )
    audits = {
        "bound_input_hashes_verified": True,
        "loss_proposer_report_validated": True,
        "loss_proposer_predictions_outcome_free": True,
        "audited_incumbent_scores_outcome_free": True,
        "state_feature_count_exact": all(
            fold["state_feature_count"] == FACTORIZED_CONDITIONED_STATE_FEATURE_COUNT
            for fold in fold_training
        ),
        "action_feature_count_exact": all(
            fold["action_feature_count"] == FACTORIZED_CONDITIONED_ACTION_FEATURE_COUNT
            for fold in fold_training
        ),
        "gate_source_exclusion": all(fold["source_exclusion_passed"] for fold in fold_training),
        "oof_score_coverage_exact": len(candidate_scores) == len(keys),
        "source_weighting_exact": weight_audit_passed,
        "matched_call_counts_exact": incumbent_match["calls"] == candidate_match["calls"] == FACTORIZED_CONDITIONED_TARGET_CALLS,
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
    score_report = {
        "scientific_status": (
            "outcome-blind matched-call factorized proposal-conditioned OOF scores "
            "frozen before opened-development outcome evaluation"
        ),
        "n_sources": 3500,
        "n_decisions": len(keys),
        "feature_mode": feature_mode,
        "state_feature_count": FACTORIZED_CONDITIONED_STATE_FEATURE_COUNT,
        "action_feature_count": FACTORIZED_CONDITIONED_ACTION_FEATURE_COUNT,
        "n_folds": n_folds,
        "fold_source_counts": fold_source_counts,
        "fold_training": fold_training,
        "incumbent_match": incumbent_match,
        "proposal_conditioned_factorized_match": candidate_match,
        "task_outcomes_used_for_thresholds": False,
        "serialized_outcome_fields": [],
        "audits": audits,
        "proposer_audit": proposer_audit,
    }
    report = {
        "scientific_status": "opened DocVQA development diagnostic; not independent validation",
        "decision": (
            "proposal_conditioned_factorized_gate_advanced"
            if all(pass_rule.values())
            else "proposal_conditioned_factorized_gate_not_advanced"
        ),
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
        "audits": audits,
        "screenqa_inputs_used": False,
        "protected_role_inputs_used": False,
    }
    model = {
        "schema": "proposal_conditioned_factorized_gate_v1",
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
        "oof_folds": serialized_folds,
        "full_refit": _serialize_factorized_heads(full_heads, full_audit),
        "deployment_composition": "frozen full loss-only proposer followed by this full factorized gate refit",
        "screenqa_inputs_used": False,
    }
    return report, score_report, model, output_rows
