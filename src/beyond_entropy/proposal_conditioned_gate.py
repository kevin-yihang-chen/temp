from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np  # type: ignore[import-not-found]

from .action_value import (
    _action_features,
    _semantic_feature_index,
    _validate_domains,
)
from .decoupled_loss_gate import (
    INCUMBENT_POOLED_CALL_RATE,
    INCUMBENT_POOLED_GAIN,
    INCUMBENT_POOLED_UTILITY,
    _evaluate,
    _prediction_index,
    match_call_count_threshold,
)
from .oof_action_value import _domain_source_balanced_weights, _source_folds
from .rescue_gate import DecisionKey
from .schema import ActionRecord


PROPOSAL_CONDITIONED_SEED = 20260906
PROPOSAL_CONDITIONED_FOLDS = 5
PROPOSAL_CONDITIONED_C = 1.0
PROPOSAL_CONDITIONED_MAX_ITER = 2000
PROPOSAL_CONDITIONED_FEATURE_COUNT = 46
PROPOSAL_CONDITIONED_TARGET_CALLS = 225
PROPOSAL_CONDITIONED_LAMBDA_COST = 0.05
PROPOSAL_CONDITIONED_BOOTSTRAP_RESAMPLES = 20000
PROPOSAL_CONDITIONED_CONFIDENCE = 0.95


_FORBIDDEN_SCORE_FIELDS = {
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


def _class_balanced_source_weights(
    domains: Sequence[str],
    sources: Sequence[str],
    labels: Sequence[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Normalize equal-domain/source row mass to one half per binary class."""

    if len(labels) != len(domains) or len(labels) != len(sources) or not labels:
        raise ValueError("class-balanced source weights require aligned rows")
    label_array = np.asarray(labels, dtype=np.int64)
    if set(label_array.tolist()) != {0, 1}:
        raise ValueError("proposal-conditioned heads require both binary classes")
    base = np.asarray(
        _domain_source_balanced_weights(domains, sources), dtype=np.float64
    )
    weights = np.zeros_like(base)
    for label in (0, 1):
        selected = label_array == label
        mass = float(base[selected].sum())
        if not math.isfinite(mass) or mass <= 0.0:
            raise ValueError("class-balanced source mass must be positive")
        weights[selected] = 0.5 * base[selected] / mass
    negative_mass = float(weights[label_array == 0].sum())
    positive_mass = float(weights[label_array == 1].sum())
    if not (
        math.isclose(negative_mass, 0.5, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(positive_mass, 0.5, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise RuntimeError("binary class balancing did not produce equal mass")
    return weights, {
        "negative_rows": int(np.sum(label_array == 0)),
        "positive_rows": int(np.sum(label_array == 1)),
        "negative_weight_mass": negative_mass,
        "positive_weight_mass": positive_mass,
        "total_weight_mass": float(weights.sum()),
        "base_weighting": "equal_domain_then_source_then_row",
        "class_balance": "one_half_mass_per_class",
    }


def _fit_conditioned_heads(
    features: np.ndarray,
    rescue_labels: Sequence[int],
    harm_labels: Sequence[int],
    domains: Sequence[str],
    sources: Sequence[str],
    *,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("proposal-conditioned features must be a non-empty matrix")
    n_rows = features.shape[0]
    if (
        len(rescue_labels) != n_rows
        or len(harm_labels) != n_rows
        or len(domains) != n_rows
        or len(sources) != n_rows
    ):
        raise ValueError("proposal-conditioned training rows are not aligned")
    if features.shape[1] != PROPOSAL_CONDITIONED_FEATURE_COUNT:
        raise ValueError(
            "proposal-conditioned gate requires exactly "
            f"{PROPOSAL_CONDITIONED_FEATURE_COUNT} features"
        )
    if not np.isfinite(features).all():
        raise ValueError("proposal-conditioned features must be finite")

    rescue_weights, rescue_balance = _class_balanced_source_weights(
        domains, sources, rescue_labels
    )
    harm_weights, harm_balance = _class_balanced_source_weights(
        domains, sources, harm_labels
    )
    model_kwargs = {
        "C": PROPOSAL_CONDITIONED_C,
        "penalty": "l2",
        "solver": "liblinear",
        "max_iter": PROPOSAL_CONDITIONED_MAX_ITER,
        "random_state": seed,
    }
    rescue_scaler = StandardScaler().fit(features)
    harm_scaler = StandardScaler().fit(features)
    rescue_model = LogisticRegression(**model_kwargs).fit(
        rescue_scaler.transform(features),
        np.asarray(rescue_labels, dtype=np.int64),
        sample_weight=rescue_weights,
    )
    harm_model = LogisticRegression(**model_kwargs).fit(
        harm_scaler.transform(features),
        np.asarray(harm_labels, dtype=np.int64),
        sample_weight=harm_weights,
    )
    if not (bool(rescue_model.n_iter_[0] < PROPOSAL_CONDITIONED_MAX_ITER) and bool(
        harm_model.n_iter_[0] < PROPOSAL_CONDITIONED_MAX_ITER
    )):
        raise RuntimeError("proposal-conditioned logistic regression did not converge")
    heads = {
        "rescue_scaler": rescue_scaler,
        "rescue_model": rescue_model,
        "harm_scaler": harm_scaler,
        "harm_model": harm_model,
    }
    audit = {
        "train_rows": n_rows,
        "feature_count": int(features.shape[1]),
        "rescue_balance": rescue_balance,
        "harm_balance": harm_balance,
        "rescue_iterations": int(rescue_model.n_iter_[0]),
        "harm_iterations": int(harm_model.n_iter_[0]),
    }
    return heads, audit


def _score_conditioned_heads(
    heads: Mapping[str, Any], features: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if features.ndim != 2 or features.shape[1] != PROPOSAL_CONDITIONED_FEATURE_COUNT:
        raise ValueError("proposal-conditioned scoring feature schema is invalid")
    rescue = np.asarray(
        heads["rescue_model"].predict_proba(
            heads["rescue_scaler"].transform(features)
        )[:, 1],
        dtype=np.float64,
    )
    harm = np.asarray(
        heads["harm_model"].predict_proba(
            heads["harm_scaler"].transform(features)
        )[:, 1],
        dtype=np.float64,
    )
    scores = rescue - harm - PROPOSAL_CONDITIONED_LAMBDA_COST
    if not (
        np.isfinite(rescue).all()
        and np.isfinite(harm).all()
        and np.isfinite(scores).all()
        and np.all((rescue >= 0.0) & (rescue <= 1.0))
        and np.all((harm >= 0.0) & (harm <= 1.0))
    ):
        raise RuntimeError("proposal-conditioned heads produced invalid probabilities")
    return rescue, harm, scores


def _serialize_conditioned_heads(
    heads: Mapping[str, Any], training_audit: Mapping[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {"training_audit": dict(training_audit)}
    for name in ("rescue", "harm"):
        scaler = heads[f"{name}_scaler"]
        model = heads[f"{name}_model"]
        result[name] = {
            "scaler_mean": [float(value) for value in scaler.mean_.tolist()],
            "scaler_scale": [float(value) for value in scaler.scale_.tolist()],
            "coefficient": [float(value) for value in model.coef_[0].tolist()],
            "intercept": float(model.intercept_[0]),
            "classes": [int(value) for value in model.classes_.tolist()],
        }
    return result


def _selected_action_features(
    keys: Sequence[DecisionKey],
    *,
    actions: Mapping[DecisionKey, str],
    baselines: Mapping[DecisionKey, ActionRecord],
    zooms: Mapping[DecisionKey, Sequence[ActionRecord]],
    semantic_by_key: Mapping[DecisionKey, Mapping[str, Any]],
    feature_mode: str,
) -> tuple[dict[DecisionKey, ActionRecord], dict[DecisionKey, np.ndarray]]:
    selected: dict[DecisionKey, ActionRecord] = {}
    features: dict[DecisionKey, np.ndarray] = {}
    for key in keys:
        matches = [
            action for action in zooms[key] if action.action_id == actions[key]
        ]
        if len(matches) != 1:
            raise ValueError(f"loss-proposed action is invalid for {key!r}")
        action = matches[0]
        vector = np.asarray(
            _action_features(
                baselines[key],
                action,
                feature_mode=feature_mode,
                semantic_decision=semantic_by_key.get(key),
            ),
            dtype=np.float64,
        )
        if vector.shape != (PROPOSAL_CONDITIONED_FEATURE_COUNT,) or not np.isfinite(
            vector
        ).all():
            raise ValueError("loss-proposed action feature schema is invalid")
        selected[key] = action
        features[key] = vector
    return selected, features


def _audited_incumbent_index(
    rows: Sequence[Mapping[str, Any]],
    *,
    baselines: Mapping[DecisionKey, ActionRecord],
    loss_actions: Mapping[DecisionKey, str],
) -> tuple[
    dict[DecisionKey, str],
    dict[DecisionKey, float],
    dict[DecisionKey, bool],
]:
    incumbent_actions: dict[DecisionKey, str] = {}
    incumbent_scores: dict[DecisionKey, float] = {}
    incumbent_calls: dict[DecisionKey, bool] = {}
    for row in rows:
        if _FORBIDDEN_SCORE_FIELDS.intersection(row):
            raise ValueError("audited score rows contain forbidden outcomes")
        key = (str(row.get("state_id", "")), str(row.get("replicate_id", "")))
        if not all(key) or key not in baselines or key in incumbent_actions:
            raise ValueError("audited score row identities are invalid")
        action_id = str(row.get("incumbent_action_id", ""))
        score = float(row.get("incumbent_score", math.nan))
        called = row.get("incumbent_called")
        if (
            not action_id
            or not math.isfinite(score)
            or not isinstance(called, bool)
            or str(row.get("source_id", "")) != baselines[key].source_id
            or str(row.get("decoupled_action_id", "")) != loss_actions[key]
        ):
            raise ValueError("audited incumbent row differs from bound inputs")
        incumbent_actions[key] = action_id
        incumbent_scores[key] = score
        incumbent_calls[key] = called
    expected = set(baselines)
    if not (
        set(incumbent_actions)
        == set(incumbent_scores)
        == set(incumbent_calls)
        == expected
    ):
        raise ValueError("audited incumbent rows do not exactly cover decisions")
    return incumbent_actions, incumbent_scores, incumbent_calls


def _assert_source_exclusion(
    train_keys: Sequence[DecisionKey],
    test_keys: Sequence[DecisionKey],
    *,
    baselines: Mapping[DecisionKey, ActionRecord],
    domain_by_key: Mapping[DecisionKey, str],
) -> dict[str, int | bool]:
    train_sources = {
        (domain_by_key[key], baselines[key].source_id) for key in train_keys
    }
    test_sources = {
        (domain_by_key[key], baselines[key].source_id) for key in test_keys
    }
    overlap = train_sources & test_sources
    if overlap:
        raise ValueError("proposal-conditioned fold leaks whole sources")
    return {
        "train_sources": len(train_sources),
        "test_sources": len(test_sources),
        "source_overlap": len(overlap),
        "source_exclusion_passed": True,
    }


def _rename_decoupled_candidate(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key).replace("decoupled", "proposal_conditioned"): (
                _rename_decoupled_candidate(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rename_decoupled_candidate(item) for item in value]
    return value


def validate_bound_loss_proposer_report(report: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "decision": "joint_auxiliary_proposer_not_advanced",
        "n_sources": 3500,
        "n_decisions": 13580,
        "n_folds": 5,
        "feature_count": PROPOSAL_CONDITIONED_FEATURE_COUNT,
        "feature_mode": "hybrid-context-semantic",
        "docvqa_calibration_formal_reserve_inputs_used": False,
    }
    mismatches = {
        key: {"expected": expected, "observed": report.get(key)}
        for key, expected in required.items()
        if report.get(key) != expected
    }
    fold_counts = report.get("fold_source_counts")
    if fold_counts != {"docvqa": [700, 700, 700, 700, 700]}:
        mismatches["fold_source_counts"] = {
            "expected": {"docvqa": [700, 700, 700, 700, 700]},
            "observed": fold_counts,
        }
    if mismatches:
        raise ValueError(f"bound loss proposer report audit failed: {mismatches}")
    return {
        "loss_proposer_report_validated": True,
        "loss_proposer_oof_source_folds": 5,
        "loss_proposer_oof_sources": 3500,
        "loss_proposer_oof_decisions": 13580,
    }


def evaluate_proposal_conditioned_gate(
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
    n_folds: int = PROPOSAL_CONDITIONED_FOLDS,
    bootstrap_resamples: int = PROPOSAL_CONDITIONED_BOOTSTRAP_RESAMPLES,
    seed: int = PROPOSAL_CONDITIONED_SEED,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if not bound_inputs_verified:
        raise ValueError("proposal-conditioned fitting requires hash-bound inputs")
    if feature_mode != "hybrid-context-semantic":
        raise ValueError("proposal-conditioned protocol requires hybrid-context-semantic")
    if n_folds != PROPOSAL_CONDITIONED_FOLDS or seed != PROPOSAL_CONDITIONED_SEED:
        raise ValueError("proposal-conditioned folds and seed are frozen")
    if bootstrap_resamples != PROPOSAL_CONDITIONED_BOOTSTRAP_RESAMPLES:
        raise ValueError("proposal-conditioned bootstrap count is frozen")
    proposer_audit = validate_bound_loss_proposer_report(bound_loss_proposer_report)

    domain_by_key, baselines, zooms = _validate_domains(records_by_domain)
    if (
        len(set(record.source_id for record in baselines.values())) != 3500
        or len(baselines) != 13580
        or any(len(actions) != 4 for actions in zooms.values())
    ):
        raise ValueError("proposal-conditioned population contract no longer matches")
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
    selected_actions, features_by_key = _selected_action_features(
        keys,
        actions=loss_actions,
        baselines=baselines,
        zooms=zooms,
        semantic_by_key=semantic_by_key,
        feature_mode=feature_mode,
    )

    fold_by_key, fold_source_counts = _source_folds(
        domain_by_key,
        baselines,
        n_folds=n_folds,
        seed=seed,
    )
    candidate_scores: dict[DecisionKey, float] = {}
    candidate_rescue: dict[DecisionKey, float] = {}
    candidate_harm: dict[DecisionKey, float] = {}
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
        train_features = np.stack([features_by_key[key] for key in train_keys])
        rescue_labels = [
            int(selected_actions[key].delta_success > 0.0) for key in train_keys
        ]
        harm_labels = [
            int(selected_actions[key].delta_success < 0.0) for key in train_keys
        ]
        domains = [domain_by_key[key] for key in train_keys]
        sources = [baselines[key].source_id for key in train_keys]
        heads, training_audit = _fit_conditioned_heads(
            train_features,
            rescue_labels,
            harm_labels,
            domains,
            sources,
            seed=seed + fold,
        )
        test_features = np.stack([features_by_key[key] for key in test_keys])
        rescue_probability, harm_probability, scores = _score_conditioned_heads(
            heads, test_features
        )
        for index, key in enumerate(test_keys):
            if key in candidate_scores:
                raise RuntimeError("proposal-conditioned OOF decision scored twice")
            candidate_scores[key] = float(scores[index])
            candidate_rescue[key] = float(rescue_probability[index])
            candidate_harm[key] = float(harm_probability[index])
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
                "heads": _serialize_conditioned_heads(heads, training_audit),
            }
        )
    if not (
        set(candidate_scores)
        == set(candidate_rescue)
        == set(candidate_harm)
        == set(keys)
    ):
        raise RuntimeError("proposal-conditioned OOF scoring is incomplete")

    all_features = np.stack([features_by_key[key] for key in keys])
    all_domains = [domain_by_key[key] for key in keys]
    all_sources = [baselines[key].source_id for key in keys]
    full_heads, full_training_audit = _fit_conditioned_heads(
        all_features,
        [int(selected_actions[key].delta_success > 0.0) for key in keys],
        [int(selected_actions[key].delta_success < 0.0) for key in keys],
        all_domains,
        all_sources,
        seed=seed,
    )

    incumbent_match = match_call_count_threshold(
        incumbent_scores, target_calls=PROPOSAL_CONDITIONED_TARGET_CALLS
    )
    candidate_match = match_call_count_threshold(
        candidate_scores, target_calls=PROPOSAL_CONDITIONED_TARGET_CALLS
    )
    incumbent_call_keys = {
        key
        for key, value in incumbent_scores.items()
        if value >= float(incumbent_match["threshold"])
    }
    audited_call_keys = {key for key, called in audited_incumbent_calls.items() if called}
    if (
        incumbent_match["calls"] != PROPOSAL_CONDITIONED_TARGET_CALLS
        or candidate_match["calls"] != PROPOSAL_CONDITIONED_TARGET_CALLS
        or incumbent_call_keys != audited_call_keys
    ):
        raise ValueError("matched-call comparison does not reproduce the incumbent")

    evaluated_alias = _evaluate(
        baselines=baselines,
        zooms=zooms,
        actions_by_method={
            "incumbent": incumbent_actions,
            "decoupled": loss_actions,
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
    evaluated = _rename_decoupled_candidate(evaluated_alias)
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
        raise ValueError("audited incumbent pooled metrics no longer reproduce")

    output_rows: list[dict[str, Any]] = []
    for row in evaluated["score_rows"]:
        key = (str(row["state_id"]), str(row["replicate_id"]))
        row["proposal_conditioned_rescue_probability"] = candidate_rescue[key]
        row["proposal_conditioned_harm_probability"] = candidate_harm[key]
        if _FORBIDDEN_SCORE_FIELDS.intersection(row):
            raise RuntimeError("proposal-conditioned output scores leak outcomes")
        output_rows.append(row)

    source_points = evaluated["source_balanced"]
    incumbent = source_points["incumbent"]
    candidate = source_points["proposal_conditioned"]
    primary = evaluated["primary_estimand"]
    class_balance_passed = all(
        math.isclose(
            float(fold[head]["positive_weight_mass"]),
            0.5,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(fold[head]["negative_weight_mass"]),
            0.5,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for fold in fold_training
        for head in ("rescue_balance", "harm_balance")
    )
    audits = {
        "bound_input_hashes_verified": True,
        "loss_proposer_report_validated": True,
        "loss_proposer_predictions_outcome_free": True,
        "audited_incumbent_scores_outcome_free": True,
        "semantic_feature_schema_exact": True,
        "feature_count_exact": all(
            fold["feature_count"] == PROPOSAL_CONDITIONED_FEATURE_COUNT
            for fold in fold_training
        ),
        "gate_source_exclusion": all(
            fold["source_exclusion_passed"] for fold in fold_training
        ),
        "oof_score_coverage_exact": len(candidate_scores) == len(keys),
        "class_balance_exact": class_balance_passed,
        "matched_call_counts_exact": (
            incumbent_match["calls"]
            == candidate_match["calls"]
            == PROPOSAL_CONDITIONED_TARGET_CALLS
        ),
        "incumbent_call_set_reproduced": incumbent_call_keys == audited_call_keys,
        "incumbent_pooled_metrics_reproduced": True,
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
            "outcome-blind matched-call proposal-conditioned OOF scores frozen "
            "before opened-development outcome evaluation"
        ),
        "n_sources": 3500,
        "n_decisions": len(keys),
        "feature_mode": feature_mode,
        "feature_count": PROPOSAL_CONDITIONED_FEATURE_COUNT,
        "n_folds": n_folds,
        "fold_source_counts": fold_source_counts,
        "fold_training": fold_training,
        "incumbent_match": incumbent_match,
        "proposal_conditioned_match": candidate_match,
        "task_outcomes_used_for_thresholds": False,
        "serialized_outcome_fields": [],
        "audits": audits,
        "proposer_audit": proposer_audit,
    }
    report = {
        "scientific_status": (
            "opened DocVQA development diagnostic; not independent validation"
        ),
        "decision": (
            "proposal_conditioned_gate_advanced"
            if all(pass_rule.values())
            else "proposal_conditioned_gate_not_advanced"
        ),
        "pass_rule": pass_rule,
        "n_sources": 3500,
        "n_decisions": len(keys),
        "lambda_cost": PROPOSAL_CONDITIONED_LAMBDA_COST,
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
        "schema": "proposal_conditioned_rescue_harm_gate_v1",
        "feature_mode": feature_mode,
        "feature_count": PROPOSAL_CONDITIONED_FEATURE_COUNT,
        "score": "p_rescue_minus_p_harm_minus_0.05",
        "seed": seed,
        "n_folds": n_folds,
        "C": PROPOSAL_CONDITIONED_C,
        "penalty": "l2",
        "solver": "liblinear",
        "max_iter": PROPOSAL_CONDITIONED_MAX_ITER,
        "oof_folds": serialized_folds,
        "full_refit": _serialize_conditioned_heads(
            full_heads, full_training_audit
        ),
        "deployment_composition": (
            "frozen full loss-only proposer followed by this full rescue/harm refit"
        ),
        "screenqa_inputs_used": False,
    }
    return report, score_report, model, output_rows
