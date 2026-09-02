from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np  # type: ignore[import-not-found]

from .dataset import DecisionKey
from .infographicvqa_attention_stop_diagnostic import (
    _positive_net_keys,
    _selection_diagnostic,
)
from .infographicvqa_attention_where import assemble_attention_where_features
from .infographicvqa_attention_where_evaluation import _paired_utility_differences
from .infographicvqa_decar import DECAR_ACTION_IDS
from .infographicvqa_decar_evaluation import (
    DECAR_BOOTSTRAP_RESAMPLES,
    DECAR_CALL_RATES,
    DECAR_LAMBDA_COST,
    _aggregate_policy,
    _bootstrap_all_policies,
    _complete_tie_exact_match,
    _policy_metrics,
    _source_concentration,
    build_decar_outcomes,
)
from .rescue_gate import compact_action_features, compact_rescue_features
from .scaled_action_value import _serialize_linear
from .schema import ActionRecord


ATTENTION_SIGNED_STOP_SCHEMA = "infographicvqa_attention_signed_stop_oof_v1"
ATTENTION_SIGNED_STOP_SEED = 20260918
ATTENTION_SIGNED_STOP_FOLDS = 5
ATTENTION_SIGNED_STOP_C = 0.01
ATTENTION_SIGNED_STOP_MAX_ITER = 2_000
ATTENTION_SIGNED_STOP_FEATURE_COUNT = 80
ATTENTION_SIGNED_STOP_PRIMARY_RATE = 0.02
ATTENTION_SIGNED_STOP_PRIMARY_CALLS = 479


@dataclass(frozen=True)
class PreparedAttentionSignedStop:
    keys: tuple[DecisionKey, ...]
    outcomes: Mapping[DecisionKey, Any]
    action_by_key: Mapping[DecisionKey, str]
    features: np.ndarray
    labels: np.ndarray
    utilities: np.ndarray
    source_ids: tuple[str, ...]
    image_ids: tuple[str, ...]
    fold_by_key: Mapping[DecisionKey, int]
    feature_audit: Mapping[str, Any]


def _source_folds(
    keys: Sequence[DecisionKey],
    source_by_key: Mapping[DecisionKey, str],
    *,
    n_folds: int,
    seed: int,
) -> dict[DecisionKey, int]:
    if n_folds < 2:
        raise ValueError("attention signed stop requires at least two folds")
    sources = {source_by_key[key] for key in keys}
    if len(sources) < n_folds:
        raise ValueError("attention signed stop has fewer sources than folds")
    ordered = sorted(
        sources,
        key=lambda source: (
            hashlib.sha256(
                f"infographicvqa-attention-signed-stop-fold-v1\0{seed}\0{source}".encode()
            ).digest(),
            source,
        ),
    )
    source_fold = {source: index % n_folds for index, source in enumerate(ordered)}
    return {key: source_fold[source_by_key[key]] for key in keys}


def _source_utility_weights(
    utilities: Sequence[float], source_ids: Sequence[str]
) -> np.ndarray:
    if not utilities or len(utilities) != len(source_ids):
        raise ValueError("attention signed stop weights require aligned rows")
    values = np.asarray(utilities, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values == 0.0):
        raise ValueError("attention signed stop utilities must be finite and nonzero")
    totals: dict[str, float] = {}
    for source_id, utility in zip(source_ids, values.tolist(), strict=True):
        totals[source_id] = totals.get(source_id, 0.0) + abs(float(utility))
    if not totals or any(total <= 0.0 for total in totals.values()):
        raise ValueError("attention signed stop source utility mass is invalid")
    weights = np.asarray(
        [
            abs(float(utility)) / totals[source_id]
            for source_id, utility in zip(source_ids, values.tolist(), strict=True)
        ],
        dtype=np.float64,
    )
    weights *= len(weights) / float(weights.sum())
    return weights


def prepare_attention_signed_stop(
    records: Sequence[ActionRecord],
    attention_feature_payload: Mapping[str, Any],
    *,
    expected_attention_code_revision: str,
    expected_model_revision: str,
    expected_source_features_sha256: str,
    expected_rollouts_sha256: str,
    expected_decisions: int = 23_946,
    expected_sources: int = 2_204,
    expected_positive_net_states: int = 1_023,
    n_folds: int = ATTENTION_SIGNED_STOP_FOLDS,
    seed: int = ATTENTION_SIGNED_STOP_SEED,
) -> PreparedAttentionSignedStop:
    if n_folds != ATTENTION_SIGNED_STOP_FOLDS or seed != ATTENTION_SIGNED_STOP_SEED:
        raise ValueError("attention signed stop folds or seed changed")
    outcomes = build_decar_outcomes(
        records,
        expected_decisions=expected_decisions,
        expected_sources=expected_sources,
    )
    attention, feature_audit = assemble_attention_where_features(
        records,
        attention_feature_payload,
        expected_code_revision=expected_attention_code_revision,
        expected_model_revision=expected_model_revision,
        expected_source_features_sha256=expected_source_features_sha256,
        expected_rollouts_sha256=expected_rollouts_sha256,
    )
    keys = tuple(sorted(outcomes))
    attention_keys = tuple(
        zip(attention.state_ids, attention.replicate_ids, strict=True)
    )
    if attention_keys != keys:
        raise ValueError("attention signed stop feature ordering changed")
    decisions = attention_feature_payload.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("attention signed stop decisions are missing")
    ordered_decisions = sorted(
        decisions,
        key=lambda decision: (
            str(decision["state_id"]),
            str(decision["replicate_id"]),
        ),
    )
    action_by_key: dict[DecisionKey, str] = {}
    feature_rows: list[list[float]] = []
    labels: list[int] = []
    utilities: list[float] = []
    source_ids: list[str] = []
    image_ids: list[str] = []
    for index, (key, decision) in enumerate(
        zip(keys, ordered_decisions, strict=True)
    ):
        selected_index = int(attention.selected_indices[index])
        action_id = DECAR_ACTION_IDS[selected_index]
        outcome = outcomes[key]
        crop_by_id = {crop.action_id: crop for crop in outcome.crops}
        crop = crop_by_id[action_id]
        net_utility = float(
            crop.delta_success - DECAR_LAMBDA_COST * crop.tool_cost
        )
        mass = float(decision.get("question_image_attention_mass", math.nan))
        if not math.isfinite(mass) or mass <= 0.0:
            raise ValueError("attention signed stop image attention mass is invalid")
        features = [
            *compact_rescue_features(decision, outcome.baseline),
            *compact_action_features(decision, selected_index),
            math.log(mass),
        ]
        if len(features) != ATTENTION_SIGNED_STOP_FEATURE_COUNT or not all(
            math.isfinite(value) for value in features
        ):
            raise ValueError("attention signed stop feature contract changed")
        action_by_key[key] = action_id
        feature_rows.append(features)
        labels.append(int(net_utility > 0.0))
        utilities.append(net_utility)
        source_ids.append(outcome.source_id)
        image_ids.append(outcome.image_id)
    feature_array = np.asarray(feature_rows, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    utility_array = np.asarray(utilities, dtype=np.float64)
    if (
        feature_array.shape != (expected_decisions, ATTENTION_SIGNED_STOP_FEATURE_COUNT)
        or not np.isfinite(feature_array).all()
        or not np.isfinite(utility_array).all()
        or np.any(utility_array == 0.0)
        or set(label_array.tolist()) != {0, 1}
        or int(label_array.sum()) != expected_positive_net_states
    ):
        raise ValueError("attention signed stop prepared population changed")
    source_by_key = {key: source_ids[index] for index, key in enumerate(keys)}
    fold_by_key = _source_folds(
        keys,
        source_by_key,
        n_folds=n_folds,
        seed=seed,
    )
    return PreparedAttentionSignedStop(
        keys=keys,
        outcomes=outcomes,
        action_by_key=action_by_key,
        features=feature_array,
        labels=label_array,
        utilities=utility_array,
        source_ids=tuple(source_ids),
        image_ids=tuple(image_ids),
        fold_by_key=fold_by_key,
        feature_audit=feature_audit,
    )


def smoke_attention_signed_stop(
    records: Sequence[ActionRecord],
    attention_feature_payload: Mapping[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    prepared = prepare_attention_signed_stop(
        records, attention_feature_payload, **kwargs
    )
    folds: list[dict[str, Any]] = []
    for fold in range(ATTENTION_SIGNED_STOP_FOLDS):
        train = [
            index
            for index, key in enumerate(prepared.keys)
            if prepared.fold_by_key[key] != fold
        ]
        held_out = [
            index
            for index, key in enumerate(prepared.keys)
            if prepared.fold_by_key[key] == fold
        ]
        train_sources = {prepared.source_ids[index] for index in train}
        held_out_sources = {prepared.source_ids[index] for index in held_out}
        if train_sources.intersection(held_out_sources):
            raise RuntimeError("attention signed stop smoke found source leakage")
        if set(prepared.labels[train].tolist()) != {0, 1}:
            raise RuntimeError("attention signed stop smoke fold lacks a class")
        folds.append(
            {
                "fold": fold,
                "train_decisions": len(train),
                "held_out_decisions": len(held_out),
                "train_sources": len(train_sources),
                "held_out_sources": len(held_out_sources),
                "train_positive_net_states": int(prepared.labels[train].sum()),
                "held_out_positive_net_states": int(
                    prepared.labels[held_out].sum()
                ),
                "source_overlap": 0,
            }
        )
    return {
        "schema": "infographicvqa_attention_signed_stop_smoke_v1",
        "passed": True,
        "fit_performed": False,
        "policy_metrics_computed": False,
        "decisions": len(prepared.keys),
        "sources": len(set(prepared.source_ids)),
        "feature_count": int(prepared.features.shape[1]),
        "positive_net_states": int(prepared.labels.sum()),
        "negative_net_states": int((prepared.labels == 0).sum()),
        "finite_features": bool(np.isfinite(prepared.features).all()),
        "folds": folds,
        "validation_or_test_inputs_used": False,
    }


def _fit_oof(
    prepared: PreparedAttentionSignedStop,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

    scores = np.full(len(prepared.keys), np.nan, dtype=np.float64)
    fold_models: list[dict[str, Any]] = []
    fold_audits: list[dict[str, Any]] = []
    for fold in range(ATTENTION_SIGNED_STOP_FOLDS):
        train_indices = np.asarray(
            [
                index
                for index, key in enumerate(prepared.keys)
                if prepared.fold_by_key[key] != fold
            ],
            dtype=np.int64,
        )
        held_out_indices = np.asarray(
            [
                index
                for index, key in enumerate(prepared.keys)
                if prepared.fold_by_key[key] == fold
            ],
            dtype=np.int64,
        )
        train_sources = {prepared.source_ids[index] for index in train_indices}
        held_out_sources = {
            prepared.source_ids[index] for index in held_out_indices
        }
        if train_sources.intersection(held_out_sources):
            raise RuntimeError("attention signed stop OOF folds leak sources")
        weights = _source_utility_weights(
            prepared.utilities[train_indices].tolist(),
            [prepared.source_ids[index] for index in train_indices],
        )
        scaler = StandardScaler().fit(prepared.features[train_indices])
        model = LogisticRegression(
            C=ATTENTION_SIGNED_STOP_C,
            penalty="l2",
            solver="liblinear",
            max_iter=ATTENTION_SIGNED_STOP_MAX_ITER,
            class_weight=None,
            random_state=ATTENTION_SIGNED_STOP_SEED + fold,
        ).fit(
            scaler.transform(prepared.features[train_indices]),
            prepared.labels[train_indices],
            sample_weight=weights,
        )
        if int(model.n_iter_[0]) >= ATTENTION_SIGNED_STOP_MAX_ITER:
            raise RuntimeError("attention signed stop head did not converge")
        held_out_scores = np.asarray(
            model.decision_function(
                scaler.transform(prepared.features[held_out_indices])
            ),
            dtype=np.float64,
        )
        if (
            held_out_scores.shape != (len(held_out_indices),)
            or not np.isfinite(held_out_scores).all()
        ):
            raise RuntimeError("attention signed stop held-out scores are invalid")
        scores[held_out_indices] = held_out_scores
        source_mass: dict[str, float] = {}
        for index, weight in zip(train_indices, weights.tolist(), strict=True):
            source_id = prepared.source_ids[index]
            source_mass[source_id] = source_mass.get(source_id, 0.0) + weight
        expected_source_mass = len(train_indices) / len(train_sources)
        if any(
            not math.isclose(
                mass, expected_source_mass, rel_tol=0.0, abs_tol=1e-8
            )
            for mass in source_mass.values()
        ):
            raise RuntimeError("attention signed stop source weight masses differ")
        audit = {
            "fold": fold,
            "train_decisions": len(train_indices),
            "held_out_decisions": len(held_out_indices),
            "train_sources": len(train_sources),
            "held_out_sources": len(held_out_sources),
            "source_overlap": 0,
            "source_exclusion_passed": True,
            "train_positive_net_states": int(
                prepared.labels[train_indices].sum()
            ),
            "held_out_positive_net_states": int(
                prepared.labels[held_out_indices].sum()
            ),
            "feature_count": int(prepared.features.shape[1]),
            "weighting": "equal_source_then_absolute_fixed_action_net_utility",
            "weight_mass": float(weights.sum()),
            "expected_weight_mass": len(train_indices),
            "source_mass_min": float(min(source_mass.values())),
            "source_mass_max": float(max(source_mass.values())),
            "expected_source_mass": expected_source_mass,
            "iterations": int(model.n_iter_[0]),
            "converged": True,
        }
        fold_audits.append(audit)
        fold_models.append(
            {
                "fold": fold,
                "linear_head": _serialize_linear(scaler, model),
                "training_audit": audit,
            }
        )
    if not np.isfinite(scores).all():
        raise RuntimeError("attention signed stop OOF coverage is incomplete")
    return scores, fold_models, fold_audits


def _rank_exact(
    scores: Mapping[DecisionKey, float], *, target_calls: int
) -> tuple[set[DecisionKey], dict[str, Any]]:
    if not scores or not 0 <= target_calls <= len(scores):
        raise ValueError("attention signed stop target calls are invalid")
    ordered = sorted(scores, key=lambda key: (-float(scores[key]), key))
    called = set(ordered[:target_calls])
    return called, {
        "target_calls": target_calls,
        "actual_calls": len(called),
        "matched_call_count": len(called) == target_calls,
        "tie_break": "score_desc_state_id_replicate_id",
        "selection_uses_outcomes": False,
    }


def evaluate_attention_signed_stop(
    records: Sequence[ActionRecord],
    attention_feature_payload: Mapping[str, Any],
    *,
    expected_attention_code_revision: str,
    expected_model_revision: str,
    expected_source_features_sha256: str,
    expected_rollouts_sha256: str,
    bootstrap_indices: Any,
    expected_decisions: int = 23_946,
    expected_sources: int = 2_204,
    expected_positive_net_states: int = 1_023,
    expected_bootstrap_resamples: int = DECAR_BOOTSTRAP_RESAMPLES,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    prepared = prepare_attention_signed_stop(
        records,
        attention_feature_payload,
        expected_attention_code_revision=expected_attention_code_revision,
        expected_model_revision=expected_model_revision,
        expected_source_features_sha256=expected_source_features_sha256,
        expected_rollouts_sha256=expected_rollouts_sha256,
        expected_decisions=expected_decisions,
        expected_sources=expected_sources,
        expected_positive_net_states=expected_positive_net_states,
    )
    sources = sorted(set(prepared.source_ids))
    if (
        tuple(bootstrap_indices.shape)
        != (expected_bootstrap_resamples, expected_sources)
        or str(bootstrap_indices.dtype) != "int32"
    ):
        raise ValueError("attention signed stop bootstrap contract changed")
    scores, fold_models, fold_audits = _fit_oof(prepared)
    score_by_key = {
        key: float(scores[index]) for index, key in enumerate(prepared.keys)
    }
    entropy_by_key = {
        key: float(prepared.outcomes[key].baseline.entropy_before)
        for key in prepared.keys
    }
    positive_net, _ = _positive_net_keys(
        prepared.outcomes, prepared.action_by_key
    )
    policy_values: dict[str, dict[DecisionKey, dict[str, float]]] = {}
    selections: list[dict[str, Any]] = []
    for rate in DECAR_CALL_RATES:
        name = f"rate-{rate:.3f}"
        target_calls = min(
            len(prepared.keys), max(1, math.ceil(rate * len(prepared.keys)))
        )
        learned_called, learned_audit = _rank_exact(
            score_by_key, target_calls=target_calls
        )
        entropy_called, entropy_audit = _complete_tie_exact_match(
            entropy_by_key, target_calls=target_calls
        )
        if (
            len(learned_called) != target_calls
            or len(entropy_called) != target_calls
        ):
            raise RuntimeError("attention signed stop matched-call contract failed")
        call_sets = {
            "signed_value_stop": learned_called,
            "entropy_stop": entropy_called,
        }
        for policy_name, called in call_sets.items():
            policy_values[f"{name}/{policy_name}"] = _policy_metrics(
                prepared.outcomes,
                called_keys=called,
                action_by_key=prepared.action_by_key,
            )
        selections.append(
            {
                "name": name,
                "nominal_question_call_rate": rate,
                "target_calls": target_calls,
                "selection_audits": {
                    "signed_value_stop": learned_audit,
                    "entropy_stop": entropy_audit,
                },
                "selection_diagnostics": {
                    policy_name: _selection_diagnostic(
                        called, positive_net, prepared.outcomes
                    )
                    for policy_name, called in call_sets.items()
                },
            }
        )
    aggregates = {
        name: _aggregate_policy(values, prepared.outcomes, sources)
        for name, values in policy_values.items()
    }
    public = {
        name: {key: value for key, value in aggregate.items() if key != "source_values"}
        for name, aggregate in aggregates.items()
    }
    bootstrap, _ = _bootstrap_all_policies(
        aggregates, sources, bootstrap_indices
    )
    operating_points: list[dict[str, Any]] = []
    for selection in selections:
        name = str(selection["name"])
        learned_name = f"{name}/signed_value_stop"
        entropy_name = f"{name}/entropy_stop"
        difference = _paired_utility_differences(
            primary=aggregates[learned_name],
            comparators={"entropy_stop": aggregates[entropy_name]},
            sources=sources,
            bootstrap_indices=bootstrap_indices,
        )["entropy_stop"]
        operating_points.append(
            {
                **selection,
                "policies": {
                    "signed_value_stop": public[learned_name],
                    "entropy_stop": public[entropy_name],
                },
                "source_bootstrap": {
                    "signed_value_stop": bootstrap["policies"][learned_name],
                    "entropy_stop": bootstrap["policies"][entropy_name],
                },
                "paired_source_utility_difference_from_entropy_stop": difference,
                "source_concentration": {
                    "signed_value_stop": _source_concentration(
                        policy_values[learned_name], prepared.outcomes
                    ),
                    "entropy_stop": _source_concentration(
                        policy_values[entropy_name], prepared.outcomes
                    ),
                },
            }
        )
    primary = next(
        point
        for point in operating_points
        if math.isclose(
            float(point["nominal_question_call_rate"]),
            ATTENTION_SIGNED_STOP_PRIMARY_RATE,
        )
    )
    candidate_interval = primary["source_bootstrap"]["signed_value_stop"][
        "additive"
    ]["utility"]
    paired_interval = primary[
        "paired_source_utility_difference_from_entropy_stop"
    ]
    candidate_precision = primary["selection_diagnostics"][
        "signed_value_stop"
    ]["positive_net_precision"]
    entropy_precision = primary["selection_diagnostics"]["entropy_stop"][
        "positive_net_precision"
    ]
    clauses = {
        "audits_passed": all(
            fold["source_exclusion_passed"] and fold["converged"]
            for fold in fold_audits
        ),
        "primary_calls_equal_479": primary["target_calls"]
        == ATTENTION_SIGNED_STOP_PRIMARY_CALLS,
        "primary_utility_ci_low_above_zero": float(
            candidate_interval["ci_low"]
        )
        > 0.0,
        "primary_paired_utility_ci_low_above_zero": float(
            paired_interval["ci_low"]
        )
        > 0.0,
        "primary_positive_net_precision_above_entropy": (
            candidate_precision is not None
            and entropy_precision is not None
            and float(candidate_precision) > float(entropy_precision)
        ),
    }
    advanced = all(clauses.values())
    score_rows = [
        {
            "schema": "infographicvqa_attention_signed_stop_oof_score_v1",
            "state_id": key[0],
            "replicate_id": key[1],
            "source_id": prepared.source_ids[index],
            "image_id": prepared.image_ids[index],
            "action_id": prepared.action_by_key[key],
            "outer_fold": int(prepared.fold_by_key[key]),
            "score": float(scores[index]),
        }
        for index, key in enumerate(prepared.keys)
    ]
    report = {
        "schema": ATTENTION_SIGNED_STOP_SCHEMA,
        "decision": (
            "fixed_action_signed_stop_train_supported"
            if advanced
            else "fixed_action_signed_stop_train_not_supported"
        ),
        "scientific_status": (
            "opened-train whole-source OOF candidate; not a formal, calibration, "
            "validation, test, or deployable result"
        ),
        "population": {
            "decisions": len(prepared.keys),
            "sources": len(sources),
            "images": len(set(prepared.image_ids)),
            "positive_net_states": int(prepared.labels.sum()),
            "negative_net_states": int((prepared.labels == 0).sum()),
        },
        "candidate": {
            "action": "argmax_question_region_attention_fixed",
            "target": "fixed_action_delta_success_minus_0.05_gt_zero",
            "feature_count": ATTENTION_SIGNED_STOP_FEATURE_COUNT,
            "model": "standardized_l2_logistic_regression",
            "c": ATTENTION_SIGNED_STOP_C,
            "class_weight": None,
            "sample_weight": "equal_source_then_absolute_fixed_action_net_utility",
            "n_folds": ATTENTION_SIGNED_STOP_FOLDS,
            "seed": ATTENTION_SIGNED_STOP_SEED,
        },
        "feature_audit": prepared.feature_audit,
        "fold_audits": fold_audits,
        "registered_call_rates": list(DECAR_CALL_RATES),
        "primary_rate": ATTENTION_SIGNED_STOP_PRIMARY_RATE,
        "primary_calls": ATTENTION_SIGNED_STOP_PRIMARY_CALLS,
        "decision_clauses": clauses,
        "operating_points": operating_points,
        "bootstrap": bootstrap["metadata"],
        "validation_or_test_inputs_used": False,
        "protected_role_inputs_used": False,
        "valid_for_formal_claim": False,
    }
    model = {
        "schema": "infographicvqa_attention_signed_stop_oof_model_v1",
        "feature_count": ATTENTION_SIGNED_STOP_FEATURE_COUNT,
        "feature_specification": [
            "compact_rescue_features_with_baseline_context",
            "compact_action_features_for_raw_attention_argmax",
            "log_question_image_attention_mass",
        ],
        "c": ATTENTION_SIGNED_STOP_C,
        "class_weight": None,
        "seed": ATTENTION_SIGNED_STOP_SEED,
        "n_folds": ATTENTION_SIGNED_STOP_FOLDS,
        "folds": fold_models,
        "validation_or_test_inputs_used": False,
    }
    return report, model, score_rows
