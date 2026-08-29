from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass
from statistics import mean
from typing import Any, Mapping, Sequence

from .action_value import _action_features, _state_features
from .dataset import group_by_decision
from .metrics import bootstrap_policy_evaluation, evaluate_policy
from .rescue_gate import DecisionKey, PrecomputedActionGatePolicy
from .risk_control import AcquisitionCalibrationRow, threshold_grid_from_training_scores
from .schema import ActionRecord


@dataclass(frozen=True)
class ScaledActionValuePrediction:
    state_id: str
    replicate_id: str
    source_id: str
    action_id: str
    predicted_gain: float
    score: float
    tool_cost: float


@dataclass(frozen=True)
class _PreparedDecisions:
    keys: tuple[DecisionKey, ...]
    baselines: Mapping[DecisionKey, ActionRecord]
    zooms: Mapping[DecisionKey, tuple[ActionRecord, ...]]
    state_features: Mapping[DecisionKey, tuple[float, ...]]
    action_features: Mapping[tuple[DecisionKey, str], tuple[float, ...]]


@dataclass(frozen=True)
class _RankedAction:
    action_id: str
    action_index: int
    action_scores: tuple[float, ...]


def _prepare_decisions(
    records: Sequence[ActionRecord],
    *,
    feature_mode: str,
    semantic_decisions: Mapping[DecisionKey, Mapping[str, Any]] | None,
) -> _PreparedDecisions:
    grouped = group_by_decision(records)
    if not grouped:
        raise ValueError("scaled action value requires non-empty sibling records")
    semantic_modes = {"semantic-context", "hybrid-context-semantic"}
    if feature_mode in semantic_modes and semantic_decisions is None:
        raise ValueError("semantic feature mode requires semantic decisions")
    baselines: dict[DecisionKey, ActionRecord] = {}
    zooms: dict[DecisionKey, tuple[ActionRecord, ...]] = {}
    state_features: dict[DecisionKey, tuple[float, ...]] = {}
    action_features: dict[tuple[DecisionKey, str], tuple[float, ...]] = {}
    for key, siblings in grouped.items():
        answers = [record for record in siblings if record.action_type == "ANSWER"]
        candidates = tuple(
            sorted(
                (record for record in siblings if record.action_type == "ZOOM"),
                key=lambda record: record.action_id,
            )
        )
        if len(answers) != 1 or len(candidates) < 2:
            raise ValueError(f"decision {key!r} requires one ANSWER and at least two ZOOMs")
        semantic = semantic_decisions.get(key) if semantic_decisions is not None else None
        if feature_mode in semantic_modes:
            assert semantic is not None
            expected_action_ids = [str(value) for value in semantic.get("action_ids", [])]
            if expected_action_ids != [candidate.action_id for candidate in candidates]:
                raise ValueError(f"semantic action IDs differ for decision {key!r}")
        baseline = answers[0]
        baselines[key] = baseline
        zooms[key] = candidates
        state_features[key] = tuple(
            _state_features(
                baseline,
                feature_mode=feature_mode,
                semantic_decision=semantic,
            )
        )
        for candidate in candidates:
            action_features[(key, candidate.action_id)] = tuple(
                _action_features(
                    baseline,
                    candidate,
                    feature_mode=feature_mode,
                    semantic_decision=semantic,
                )
            )
    keys = tuple(sorted(grouped))
    state_sizes = {len(state_features[key]) for key in keys}
    action_sizes = {
        len(action_features[(key, candidate.action_id)])
        for key in keys
        for candidate in zooms[key]
    }
    if len(state_sizes) != 1 or len(action_sizes) != 1:
        raise ValueError("scaled action-value feature dimensions must be constant")
    return _PreparedDecisions(
        keys=keys,
        baselines=baselines,
        zooms=zooms,
        state_features=state_features,
        action_features=action_features,
    )


def _source_folds(
    keys: Sequence[DecisionKey],
    baselines: Mapping[DecisionKey, ActionRecord],
    *,
    n_folds: int,
    seed: int,
) -> dict[DecisionKey, int]:
    if n_folds < 2:
        raise ValueError("source cross-fitting requires at least two folds")
    sources = {baselines[key].source_id for key in keys}
    if len(sources) < n_folds:
        raise ValueError("source cross-fitting has fewer sources than folds")
    ordered = sorted(
        sources,
        key=lambda source: (
            hashlib.sha256(
                f"scaled-action-value-fold-v1\0{seed}\0{source}".encode()
            ).digest(),
            source,
        ),
    )
    source_fold = {source: index % n_folds for index, source in enumerate(ordered)}
    return {key: source_fold[baselines[key].source_id] for key in keys}


def _source_balanced_weights(source_ids: Sequence[str]) -> list[float]:
    counts: dict[str, int] = {}
    for source_id in source_ids:
        counts[source_id] = counts.get(source_id, 0) + 1
    raw = [1.0 / counts[source_id] for source_id in source_ids]
    scale = len(raw) / sum(raw)
    return [value * scale for value in raw]


def _fit_pairwise_ranker(
    prepared: _PreparedDecisions,
    keys: Sequence[DecisionKey],
    *,
    c_value: float,
    seed: int,
) -> tuple[Any, Any, dict[str, int]]:
    import numpy as np  # type: ignore[import-not-found]
    from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

    rows: list[list[float]] = []
    labels: list[int] = []
    source_ids: list[str] = []
    tied_pairs = 0
    for key in keys:
        candidates = prepared.zooms[key]
        for left, right in itertools.combinations(candidates, 2):
            if math.isclose(left.delta_success, right.delta_success, abs_tol=1e-12):
                tied_pairs += 1
                continue
            left_features = prepared.action_features[(key, left.action_id)]
            right_features = prepared.action_features[(key, right.action_id)]
            difference = [
                left_value - right_value
                for left_value, right_value in zip(left_features, right_features)
            ]
            label = int(left.delta_success > right.delta_success)
            rows.extend((difference, [-value for value in difference]))
            labels.extend((label, 1 - label))
            source_ids.extend((prepared.baselines[key].source_id,) * 2)
    if not rows:
        raise ValueError("pairwise ranker has no unequal-outcome action pairs")
    array = np.asarray(rows, dtype=np.float64)
    scaler = StandardScaler().fit(array)
    model = LogisticRegression(
        C=float(c_value),
        solver="liblinear",
        max_iter=2000,
        random_state=seed,
    ).fit(
        scaler.transform(array),
        np.asarray(labels, dtype=np.int64),
        sample_weight=np.asarray(_source_balanced_weights(source_ids)),
    )
    return scaler, model, {
        "training_rows": len(rows),
        "tied_unordered_pairs": tied_pairs,
    }


def _rank_actions(
    prepared: _PreparedDecisions,
    keys: Sequence[DecisionKey],
    *,
    scaler: Any,
    model: Any,
) -> dict[DecisionKey, _RankedAction]:
    import numpy as np  # type: ignore[import-not-found]

    ranked: dict[DecisionKey, _RankedAction] = {}
    for key in keys:
        candidates = prepared.zooms[key]
        features = np.asarray(
            [prepared.action_features[(key, candidate.action_id)] for candidate in candidates],
            dtype=np.float64,
        )
        scores = model.decision_function(scaler.transform(features)).tolist()
        selected_index = max(
            range(len(candidates)),
            key=lambda index: (float(scores[index]), candidates[index].action_id),
        )
        ranked[key] = _RankedAction(
            action_id=candidates[selected_index].action_id,
            action_index=selected_index,
            action_scores=tuple(float(value) for value in scores),
        )
    return ranked


def _crossfit_ranker(
    prepared: _PreparedDecisions,
    keys: Sequence[DecisionKey],
    *,
    n_folds: int,
    c_value: float,
    seed: int,
) -> dict[DecisionKey, _RankedAction]:
    fold_by_key = _source_folds(keys, prepared.baselines, n_folds=n_folds, seed=seed)
    predictions: dict[DecisionKey, _RankedAction] = {}
    for fold in range(n_folds):
        train_keys = [key for key in keys if fold_by_key[key] != fold]
        test_keys = [key for key in keys if fold_by_key[key] == fold]
        scaler, model, _ = _fit_pairwise_ranker(
            prepared,
            train_keys,
            c_value=c_value,
            seed=seed + fold,
        )
        predictions.update(
            _rank_actions(prepared, test_keys, scaler=scaler, model=model)
        )
    if set(predictions) != set(keys):
        raise RuntimeError("cross-fitted ranker predictions are incomplete")
    return predictions


def _selected_action(
    prepared: _PreparedDecisions,
    key: DecisionKey,
    ranking: _RankedAction,
) -> ActionRecord:
    candidate = prepared.zooms[key][ranking.action_index]
    if candidate.action_id != ranking.action_id:
        raise RuntimeError("ranked action index and ID disagree")
    return candidate


def _call_features(
    prepared: _PreparedDecisions,
    key: DecisionKey,
    ranking: _RankedAction,
) -> list[float]:
    selected = _selected_action(prepared, key, ranking)
    ordered_scores = sorted(ranking.action_scores, reverse=True)
    gap = ordered_scores[0] - ordered_scores[1]
    score_mean = mean(ranking.action_scores)
    score_variance = mean(
        (value - score_mean) ** 2 for value in ranking.action_scores
    )
    return [
        *prepared.state_features[key],
        *prepared.action_features[(key, selected.action_id)],
        ranking.action_scores[ranking.action_index],
        gap,
        score_mean,
        math.sqrt(score_variance),
    ]


def _fit_call_value_head(
    prepared: _PreparedDecisions,
    rankings: Mapping[DecisionKey, _RankedAction],
    keys: Sequence[DecisionKey],
    *,
    alpha: float,
) -> tuple[Any, Any]:
    import numpy as np  # type: ignore[import-not-found]
    from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

    features = np.asarray(
        [_call_features(prepared, key, rankings[key]) for key in keys],
        dtype=np.float64,
    )
    targets = np.asarray(
        [_selected_action(prepared, key, rankings[key]).delta_success for key in keys],
        dtype=np.float64,
    )
    source_ids = [prepared.baselines[key].source_id for key in keys]
    scaler = StandardScaler().fit(features)
    model = Ridge(alpha=float(alpha)).fit(
        scaler.transform(features),
        targets,
        sample_weight=np.asarray(_source_balanced_weights(source_ids)),
    )
    return scaler, model


def _predict_call_values(
    prepared: _PreparedDecisions,
    rankings: Mapping[DecisionKey, _RankedAction],
    keys: Sequence[DecisionKey],
    *,
    scaler: Any,
    model: Any,
) -> dict[DecisionKey, float]:
    import numpy as np  # type: ignore[import-not-found]

    features = np.asarray(
        [_call_features(prepared, key, rankings[key]) for key in keys],
        dtype=np.float64,
    )
    values = model.predict(scaler.transform(features)).tolist()
    return {key: float(value) for key, value in zip(keys, values)}


def _ranking_summary(
    prepared: _PreparedDecisions,
    rankings: Mapping[DecisionKey, _RankedAction],
) -> dict[str, float]:
    selected_gains: dict[str, list[float]] = {}
    random_gains: dict[str, list[float]] = {}
    helpful_rescues: list[float] = []
    random_rescues: list[float] = []
    for key in prepared.keys:
        source_id = prepared.baselines[key].source_id
        selected = _selected_action(prepared, key, rankings[key])
        candidates = prepared.zooms[key]
        selected_gains.setdefault(source_id, []).append(selected.delta_success)
        random_gains.setdefault(source_id, []).append(
            mean(candidate.delta_success for candidate in candidates)
        )
        if any(candidate.delta_success > 0.0 for candidate in candidates):
            helpful_rescues.append(float(selected.delta_success > 0.0))
            random_rescues.append(
                mean(candidate.delta_success > 0.0 for candidate in candidates)
            )
    return {
        "source_balanced_selected_gain": mean(
            mean(values) for values in selected_gains.values()
        ),
        "source_balanced_random_gain": mean(
            mean(values) for values in random_gains.values()
        ),
        "top1_rescue_rate_within_helpful_states": mean(helpful_rescues),
        "random_rescue_rate_within_helpful_states": mean(random_rescues),
    }


def _source_balanced_mse(
    prepared: _PreparedDecisions,
    rankings: Mapping[DecisionKey, _RankedAction],
    predictions: Mapping[DecisionKey, float],
) -> float:
    losses: dict[str, list[float]] = {}
    for key, predicted in predictions.items():
        target = _selected_action(prepared, key, rankings[key]).delta_success
        losses.setdefault(prepared.baselines[key].source_id, []).append(
            (predicted - target) ** 2
        )
    return mean(mean(values) for values in losses.values())


def _serialize_linear(scaler: Any, model: Any) -> dict[str, Any]:
    coefficient = model.coef_
    if getattr(coefficient, "ndim", 1) == 2:
        coefficient = coefficient[0]
    intercept = model.intercept_
    if hasattr(intercept, "tolist"):
        intercept = intercept.tolist()
    if isinstance(intercept, list):
        intercept = intercept[0]
    return {
        "scaler_mean": [float(value) for value in scaler.mean_.tolist()],
        "scaler_scale": [float(value) for value in scaler.scale_.tolist()],
        "coefficient": [float(value) for value in coefficient.tolist()],
        "intercept": float(intercept),
    }


def _serialized_linear_predict(payload: Mapping[str, Any], features: Sequence[float]) -> float:
    means = [float(value) for value in payload["scaler_mean"]]
    scales = [float(value) for value in payload["scaler_scale"]]
    coefficients = [float(value) for value in payload["coefficient"]]
    if not len(features) == len(means) == len(scales) == len(coefficients):
        raise ValueError("serialized linear feature dimension mismatch")
    return float(payload["intercept"]) + sum(
        coefficient * ((float(value) - center) / scale)
        for value, center, scale, coefficient in zip(
            features, means, scales, coefficients
        )
    )


def fit_scaled_pairwise_action_value_model(
    records: Sequence[ActionRecord],
    *,
    feature_mode: str = "context-geometry",
    semantic_decisions: Mapping[DecisionKey, Mapping[str, Any]] | None = None,
    n_folds: int = 5,
    lambda_cost: float = 0.05,
    ranker_c_values: Sequence[float] = (0.01, 0.1, 1.0),
    call_alpha_values: Sequence[float] = (1.0, 10.0, 100.0),
    max_thresholds: int = 128,
    bootstrap_resamples: int = 2000,
    seed: int = 20260828,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit a source-cross-fitted pairwise ranker and selected-action value head."""

    if lambda_cost < 0.0 or not math.isfinite(lambda_cost):
        raise ValueError("lambda_cost must be finite and non-negative")
    if not ranker_c_values or any(value <= 0.0 for value in ranker_c_values):
        raise ValueError("ranker C values must be positive")
    if not call_alpha_values or any(value <= 0.0 for value in call_alpha_values):
        raise ValueError("call alpha values must be positive")
    prepared = _prepare_decisions(
        records,
        feature_mode=feature_mode,
        semantic_decisions=semantic_decisions,
    )
    fold_by_key = _source_folds(
        prepared.keys,
        prepared.baselines,
        n_folds=n_folds,
        seed=seed,
    )

    ranker_oof_by_c: dict[float, dict[DecisionKey, _RankedAction]] = {}
    ranker_reports: list[dict[str, Any]] = []
    for c_value in ranker_c_values:
        rankings = _crossfit_ranker(
            prepared,
            prepared.keys,
            n_folds=n_folds,
            c_value=float(c_value),
            seed=seed,
        )
        ranker_oof_by_c[float(c_value)] = rankings
        ranker_reports.append(
            {"c_value": float(c_value), **_ranking_summary(prepared, rankings)}
        )
    selected_ranker = max(
        ranker_reports,
        key=lambda report: (
            report["source_balanced_selected_gain"],
            report["top1_rescue_rate_within_helpful_states"],
            -report["c_value"],
        ),
    )
    selected_c = float(selected_ranker["c_value"])

    nested_rankings: dict[DecisionKey, _RankedAction] = {}
    call_predictions_by_alpha: dict[float, dict[DecisionKey, float]] = {
        float(alpha): {} for alpha in call_alpha_values
    }
    for outer_fold in range(n_folds):
        outer_train_keys = [
            key for key in prepared.keys if fold_by_key[key] != outer_fold
        ]
        outer_test_keys = [
            key for key in prepared.keys if fold_by_key[key] == outer_fold
        ]
        inner_rankings = _crossfit_ranker(
            prepared,
            outer_train_keys,
            n_folds=n_folds,
            c_value=selected_c,
            seed=seed + 1000 + outer_fold,
        )
        outer_scaler, outer_ranker, _ = _fit_pairwise_ranker(
            prepared,
            outer_train_keys,
            c_value=selected_c,
            seed=seed + 2000 + outer_fold,
        )
        outer_rankings = _rank_actions(
            prepared,
            outer_test_keys,
            scaler=outer_scaler,
            model=outer_ranker,
        )
        nested_rankings.update(outer_rankings)
        for alpha in call_alpha_values:
            call_scaler, call_model = _fit_call_value_head(
                prepared,
                inner_rankings,
                outer_train_keys,
                alpha=float(alpha),
            )
            call_predictions_by_alpha[float(alpha)].update(
                _predict_call_values(
                    prepared,
                    outer_rankings,
                    outer_test_keys,
                    scaler=call_scaler,
                    model=call_model,
                )
            )
    if set(nested_rankings) != set(prepared.keys):
        raise RuntimeError("nested OOF rankings are incomplete")
    call_reports = []
    for alpha, predictions in call_predictions_by_alpha.items():
        if set(predictions) != set(prepared.keys):
            raise RuntimeError("nested OOF call-value predictions are incomplete")
        call_reports.append(
            {
                "alpha": alpha,
                "source_balanced_mse": _source_balanced_mse(
                    prepared,
                    nested_rankings,
                    predictions,
                ),
            }
        )
    selected_call = min(
        call_reports,
        key=lambda report: (report["source_balanced_mse"], -report["alpha"]),
    )
    selected_alpha = float(selected_call["alpha"])
    oof_predicted_gains = call_predictions_by_alpha[selected_alpha]
    oof_scores = {
        key: oof_predicted_gains[key]
        - lambda_cost * _selected_action(prepared, key, nested_rankings[key]).tool_cost
        for key in prepared.keys
    }
    thresholds = threshold_grid_from_training_scores(
        list(oof_scores.values()),
        max_thresholds=max_thresholds,
    )
    oof_actions = {
        key: (
            nested_rankings[key].action_id if oof_scores[key] >= 0.0 else None
        )
        for key in prepared.keys
    }
    oof_policy = PrecomputedActionGatePolicy(
        oof_actions,
        name="nested_oof_pairwise_ranker_call_value_at_zero",
    )
    oof_policy_result: dict[str, Any] = dict(
        evaluate_policy(records, oof_policy, lambda_cost=lambda_cost)
    )
    oof_policy_result["source_bootstrap"] = bootstrap_policy_evaluation(
        records,
        oof_policy,
        lambda_cost=lambda_cost,
        n_resamples=bootstrap_resamples,
        seed=seed,
        cluster_by="source_id",
    )

    final_ranker_scaler, final_ranker, pair_counts = _fit_pairwise_ranker(
        prepared,
        prepared.keys,
        c_value=selected_c,
        seed=seed,
    )
    final_call_training_rankings = ranker_oof_by_c[selected_c]
    final_call_scaler, final_call_model = _fit_call_value_head(
        prepared,
        final_call_training_rankings,
        prepared.keys,
        alpha=selected_alpha,
    )
    model_payload = {
        "model_type": "source_crossfit_pairwise_ranker_call_value_v1",
        "feature_mode": feature_mode,
        "lambda_cost": lambda_cost,
        "seed": seed,
        "n_folds": n_folds,
        "selected_ranker_c": selected_c,
        "selected_call_alpha": selected_alpha,
        "state_feature_count": len(prepared.state_features[prepared.keys[0]]),
        "action_feature_count": len(
            prepared.action_features[
                (prepared.keys[0], prepared.zooms[prepared.keys[0]][0].action_id)
            ]
        ),
        "call_feature_count": len(
            _call_features(
                prepared,
                prepared.keys[0],
                final_call_training_rankings[prepared.keys[0]],
            )
        ),
        "ranker": _serialize_linear(final_ranker_scaler, final_ranker),
        "call_value": _serialize_linear(final_call_scaler, final_call_model),
        "threshold_grid": thresholds,
        "calibrated_threshold": None,
    }
    report = {
        "scientific_status": (
            "source-nested-crossfit ranker development; independent calibration "
            "outcomes are not used"
        ),
        "feature_mode": feature_mode,
        "lambda_cost": lambda_cost,
        "seed": seed,
        "n_folds": n_folds,
        "n_sources": len(
            {prepared.baselines[key].source_id for key in prepared.keys}
        ),
        "n_decisions": len(prepared.keys),
        "ranker_candidates": ranker_reports,
        "selected_ranker": selected_ranker,
        "call_value_candidates": call_reports,
        "selected_call_value": selected_call,
        "nested_ranking": _ranking_summary(prepared, nested_rankings),
        "oof_zero_threshold_policy": oof_policy_result,
        "threshold_grid": thresholds,
        "pairwise_training": pair_counts,
        "calibration_outcomes_used": False,
        "formal_outcomes_used": False,
    }
    return report, model_payload


def predict_scaled_action_value(
    model: Mapping[str, Any],
    records: Sequence[ActionRecord],
    *,
    semantic_decisions: Mapping[DecisionKey, Mapping[str, Any]] | None = None,
) -> list[ScaledActionValuePrediction]:
    """Apply a frozen model using only its registered pre-action feature path."""

    if model.get("model_type") != "source_crossfit_pairwise_ranker_call_value_v1":
        raise ValueError("unsupported scaled action-value model type")
    feature_mode = str(model["feature_mode"])
    prepared = _prepare_decisions(
        records,
        feature_mode=feature_mode,
        semantic_decisions=semantic_decisions,
    )
    ranker_payload = model["ranker"]
    call_payload = model["call_value"]
    predictions: list[ScaledActionValuePrediction] = []
    for key in prepared.keys:
        candidates = prepared.zooms[key]
        action_scores = tuple(
            _serialized_linear_predict(
                ranker_payload,
                prepared.action_features[(key, candidate.action_id)],
            )
            for candidate in candidates
        )
        selected_index = max(
            range(len(candidates)),
            key=lambda index: (action_scores[index], candidates[index].action_id),
        )
        ranking = _RankedAction(
            action_id=candidates[selected_index].action_id,
            action_index=selected_index,
            action_scores=action_scores,
        )
        predicted_gain = _serialized_linear_predict(
            call_payload,
            _call_features(prepared, key, ranking),
        )
        selected = candidates[selected_index]
        predictions.append(
            ScaledActionValuePrediction(
                state_id=key[0],
                replicate_id=key[1],
                source_id=prepared.baselines[key].source_id,
                action_id=selected.action_id,
                predicted_gain=predicted_gain,
                score=predicted_gain
                - float(model["lambda_cost"]) * selected.tool_cost,
                tool_cost=selected.tool_cost,
            )
        )
    return predictions


def acquisition_calibration_rows(
    predictions: Sequence[ScaledActionValuePrediction],
    records: Sequence[ActionRecord],
) -> list[AcquisitionCalibrationRow]:
    """Join frozen pre-action predictions to outcomes only for calibration."""

    grouped = group_by_decision(records)
    rows: list[AcquisitionCalibrationRow] = []
    seen: set[DecisionKey] = set()
    for prediction in predictions:
        key = (prediction.state_id, prediction.replicate_id)
        if key in seen or key not in grouped:
            raise ValueError("calibration predictions must uniquely cover record decisions")
        matches = [
            record
            for record in grouped[key]
            if record.action_id == prediction.action_id and record.action_type == "ZOOM"
        ]
        if len(matches) != 1:
            raise ValueError(f"calibration action is absent for decision {key!r}")
        action = matches[0]
        if prediction.source_id != action.source_id:
            raise ValueError(f"calibration source differs for decision {key!r}")
        if not math.isclose(
            prediction.tool_cost,
            action.tool_cost,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"calibration tool cost differs for decision {key!r}")
        rows.append(
            AcquisitionCalibrationRow(
                source_id=prediction.source_id,
                score=prediction.score,
                gain=action.delta_success,
                tool_cost=action.tool_cost,
            )
        )
        seen.add(key)
    if seen != set(grouped):
        raise ValueError("calibration predictions do not cover every decision")
    return rows
