from __future__ import annotations

import math
import random
import re
from statistics import mean, median
from typing import Any, Mapping, Sequence

from .dataset import group_by_decision, split_by_group
from .metrics import bootstrap_policy_evaluation, evaluate_policy
from .policies import AnswerNowPolicy, ExpectedRandomZoomPolicy, PolicyDecision
from .schema import ActionRecord


DecisionKey = tuple[str, str]


class PrecomputedRescueGatePolicy:
    """Execute one uniform random crop when a pre-action state score passes a gate."""

    name = "compact_rescue_gate_uniform_random_expectation"

    def __init__(
        self,
        scores: Mapping[DecisionKey, float],
        *,
        threshold: float,
        name: str = "compact_rescue_gate_uniform_random_expectation",
    ) -> None:
        self.scores = scores
        self.threshold = threshold
        self.name = name

    def select(self, siblings: Sequence[ActionRecord]) -> PolicyDecision:
        exemplar = siblings[0]
        key = (exemplar.state_id, exemplar.replicate_id)
        if key not in self.scores:
            raise ValueError(f"missing rescue-gate score for {key!r}")
        if self.scores[key] < self.threshold:
            return AnswerNowPolicy().select(siblings)
        return ExpectedRandomZoomPolicy().select(siblings)


class PrecomputedActionGatePolicy:
    """Execute a precomputed concrete crop, or stop, for each decision."""

    def __init__(
        self,
        selected_action_ids: Mapping[DecisionKey, str | None],
        *,
        name: str = "precomputed_action_gate",
    ) -> None:
        self.selected_action_ids = selected_action_ids
        self.name = name

    def select(self, siblings: Sequence[ActionRecord]) -> PolicyDecision:
        exemplar = siblings[0]
        key = (exemplar.state_id, exemplar.replicate_id)
        if key not in self.selected_action_ids:
            raise ValueError(f"missing action-gate decision for {key!r}")
        action_id = self.selected_action_ids[key]
        if action_id is None:
            return AnswerNowPolicy().select(siblings)
        matches = [record for record in siblings if record.action_id == action_id]
        if len(matches) != 1 or matches[0].action_type != "ZOOM":
            raise ValueError(f"invalid selected ZOOM action {action_id!r} for {key!r}")
        selected = matches[0]
        return PolicyDecision(selected, tool_calls=1, visual_cost=selected.tool_cost)


def tune_rescue_gate_threshold(
    scores: Sequence[float],
    action_utilities: Sequence[float],
) -> tuple[float, float, float]:
    """Tune a score gate on validation utility, breaking ties toward fewer calls."""

    if len(scores) != len(action_utilities) or not scores:
        raise ValueError("threshold tuning requires paired non-empty scores and utilities")
    unique = sorted(set(float(value) for value in scores))
    thresholds = [unique[-1] + 1e-9]
    thresholds.extend(
        (left + right) / 2.0 for left, right in zip(reversed(unique[:-1]), reversed(unique[1:]))
    )
    thresholds.append(unique[0] - 1e-9)
    best_threshold = thresholds[0]
    best_score = (float("-inf"), float("-inf"))
    best_tool_rate = 0.0
    for threshold in thresholds:
        selected = [score >= threshold for score in scores]
        utility = mean(
            action_utility if use_action else 0.0
            for use_action, action_utility in zip(selected, action_utilities)
        )
        tool_rate = mean(selected)
        score = (utility, -tool_rate)
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_tool_rate = tool_rate
    return best_threshold, best_score[0], best_tool_rate


def pre_action_context_features(record: ActionRecord) -> list[float]:
    """Extract low-capacity text/confidence features available before tool use."""

    question = record.question.lower().strip()
    answer = record.answer_before.lower().strip()
    backend = record.metadata.get("baseline_backend", {})
    raw_entropies = backend.get("normalized_token_entropies", []) if isinstance(
        backend, Mapping
    ) else []
    token_entropies = []
    if isinstance(raw_entropies, (list, tuple)):
        for value in raw_entropies:
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                token_entropies.append(float(value))
    entropy_mean = mean(token_entropies) if token_entropies else record.entropy_before
    entropy_variance = (
        mean((value - entropy_mean) ** 2 for value in token_entropies)
        if token_entropies
        else 0.0
    )
    numeric_answer = bool(
        re.fullmatch(r"\s*[$€£]?\s*[+-]?(?:\d+(?:[,.]\d+)*|\.\d+)\s*%?\s*", answer)
    )
    keyword_patterns = (
        r"\bhow many\b",
        r"\b(?:percentage|percent)\b",
        r"\b(?:difference|differ)\b",
        r"\b(?:sum|total)\b",
        r"\b(?:average|mean)\b",
        r"\b(?:ratio|proportion)\b",
        r"\b(?:highest|most|largest|maximum|max)\b",
        r"\b(?:lowest|least|smallest|minimum|min)\b",
        r"\b(?:increase|decrease|change|growth|decline)\b",
        r"\bwhich\b",
        r"\bwhat\b",
        r"\b(?:when|year)\b",
        r"\b(?:yes|no)\b",
    )
    return [
        float(len(question)),
        float(len(question.split())),
        float(len(answer)),
        float(len(answer.split())),
        float(any(character.isdigit() for character in answer)),
        float(numeric_answer),
        float("%" in answer or "percent" in answer),
        float(answer in {"yes", "no"}),
        float(len(token_entropies)),
        float(entropy_mean),
        float(max(token_entropies, default=record.entropy_before)),
        float(math.sqrt(entropy_variance)),
        float(token_entropies[0] if token_entropies else record.entropy_before),
        float(token_entropies[-1] if token_entropies else record.entropy_before),
        *(float(bool(re.search(pattern, question))) for pattern in keyword_patterns),
    ]


def compact_rescue_features(
    decision: Mapping[str, Any],
    baseline: ActionRecord | None = None,
) -> list[float]:
    """Build low-capacity state features from frozen pre-action representations."""

    import torch  # type: ignore[import-not-found]
    import torch.nn.functional as functional  # type: ignore[import-not-found]

    question = functional.normalize(decision["question_embedding"].float(), dim=0)
    global_visual = functional.normalize(
        decision["global_visual_embedding"].float(),
        dim=0,
    )
    regions = functional.normalize(decision["region_embeddings"].float(), dim=1)
    features = torch.cat(
        (
            decision["state_signals"].float(),
            torch.dot(question, global_visual).reshape(1),
            regions @ question,
            regions @ global_visual,
            decision["bboxes"].float().reshape(-1),
        )
    )
    result = [float(value) for value in features.tolist()]
    if baseline is not None:
        result.extend(pre_action_context_features(baseline))
    return result


def compact_action_features(
    decision: Mapping[str, Any],
    action_index: int,
) -> list[float]:
    """Build low-capacity pre-action features for ranking one candidate crop."""

    import torch  # type: ignore[import-not-found]
    import torch.nn.functional as functional  # type: ignore[import-not-found]

    question = functional.normalize(decision["question_embedding"].float(), dim=0)
    global_visual = functional.normalize(
        decision["global_visual_embedding"].float(),
        dim=0,
    )
    regions = functional.normalize(decision["region_embeddings"].float(), dim=1)
    bboxes = decision["bboxes"].float()
    if not 0 <= action_index < regions.shape[0] or bboxes.shape[0] != regions.shape[0]:
        raise ValueError("action index or bbox count does not match region embeddings")
    question_region = regions @ question
    global_region = regions @ global_visual
    bbox = bboxes[action_index]
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    geometry = torch.stack(
        (
            *bbox,
            width,
            height,
            width * height,
            (bbox[0] + bbox[2]) / 2.0,
            (bbox[1] + bbox[3]) / 2.0,
        )
    )
    features = torch.cat(
        (
            decision["state_signals"].float(),
            torch.dot(question, global_visual).reshape(1),
            question_region[action_index].reshape(1),
            global_region[action_index].reshape(1),
            (question_region[action_index] - question_region.mean()).reshape(1),
            (global_region[action_index] - global_region.mean()).reshape(1),
            geometry,
        )
    )
    return [float(value) for value in features.tolist()]


def _decision_baselines(records: Sequence[ActionRecord]) -> dict[DecisionKey, ActionRecord]:
    baselines: dict[DecisionKey, ActionRecord] = {}
    for key, siblings in group_by_decision(records).items():
        answers = [record for record in siblings if record.action_type == "ANSWER"]
        if len(answers) != 1:
            raise ValueError(f"decision {key!r} must contain exactly one ANSWER")
        baselines[key] = answers[0]
    return baselines


def _rescue_feature_map(
    keys: Sequence[DecisionKey],
    decision_by_key: Mapping[DecisionKey, Mapping[str, Any]],
    baselines: Mapping[DecisionKey, ActionRecord],
    *,
    feature_mode: str,
) -> dict[DecisionKey, list[float]]:
    if feature_mode not in ("semantic", "context", "semantic-context"):
        raise ValueError(f"unsupported rescue feature mode: {feature_mode}")
    if feature_mode == "context":
        return {key: pre_action_context_features(baselines[key]) for key in keys}
    include_context = feature_mode == "semantic-context"
    return {
        key: compact_rescue_features(
            decision_by_key[key],
            baselines[key] if include_context else None,
        )
        for key in keys
    }


def _decision_outcomes(
    records: Sequence[ActionRecord],
    *,
    lambda_cost: float,
) -> dict[DecisionKey, dict[str, float | bool]]:
    outcomes: dict[DecisionKey, dict[str, float | bool]] = {}
    for key, siblings in group_by_decision(records).items():
        zooms = [record for record in siblings if record.action_type == "ZOOM"]
        expected_gain = mean(record.delta_success for record in zooms)
        expected_cost = mean(record.tool_cost for record in zooms)
        outcomes[key] = {
            "helpful": any(record.delta_success > 0.0 for record in zooms),
            "expected_gain": expected_gain,
            "expected_utility": expected_gain - lambda_cost * expected_cost,
        }
    return outcomes


def _keys(records: Sequence[ActionRecord]) -> list[DecisionKey]:
    return sorted(group_by_decision(records))


def fit_rescue_gate_split(
    records: Sequence[ActionRecord],
    decision_by_key: Mapping[DecisionKey, Mapping[str, Any]],
    *,
    split_group: str = "image_id",
    train_fraction: float = 0.7,
    validation_fraction: float = 0.2,
    lambda_cost: float = 0.05,
    feature_mode: str = "semantic",
    c_values: Sequence[float] = (0.001, 0.01, 0.1, 1.0, 10.0),
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 0,
    seed: int = 17,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit and evaluate one leakage-safe compact rescuability gate split."""

    import numpy as np  # type: ignore[import-not-found]
    from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
    from sklearn.metrics import (  # type: ignore[import-untyped]
        average_precision_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

    if not c_values or any(value <= 0.0 for value in c_values):
        raise ValueError("c_values must contain positive regularization values")
    outer_train, test = split_by_group(
        records,
        group=split_group,  # type: ignore[arg-type]
        train_fraction=train_fraction,
        seed=seed,
    )
    model_train, validation = split_by_group(
        outer_train,
        group=split_group,  # type: ignore[arg-type]
        train_fraction=1.0 - validation_fraction,
        seed=seed + 1,
    )
    split_keys = {
        "model_train": _keys(model_train),
        "validation": _keys(validation),
        "test": _keys(test),
    }
    missing = set().union(*map(set, split_keys.values())) - set(decision_by_key)
    if missing:
        raise ValueError(f"rescue features are missing decisions: {sorted(missing)[:5]}")
    outcomes = _decision_outcomes(records, lambda_cost=lambda_cost)
    baselines = _decision_baselines(records)
    feature_by_key = _rescue_feature_map(
        [key for keys in split_keys.values() for key in keys],
        decision_by_key,
        baselines,
        feature_mode=feature_mode,
    )
    features = {
        name: np.asarray(
            [feature_by_key[key] for key in keys],
            dtype=np.float64,
        )
        for name, keys in split_keys.items()
    }
    labels = {
        name: np.asarray([bool(outcomes[key]["helpful"]) for key in keys], dtype=np.int64)
        for name, keys in split_keys.items()
    }
    scaler = StandardScaler().fit(features["model_train"])
    transformed = {
        name: scaler.transform(values) for name, values in features.items()
    }
    candidates: list[tuple[float, float, float, float, Any]] = []
    validation_utilities = [
        float(outcomes[key]["expected_utility"]) for key in split_keys["validation"]
    ]
    for c_value in c_values:
        model = LogisticRegression(
            C=float(c_value),
            class_weight="balanced",
            solver="liblinear",
            max_iter=2000,
            random_state=seed,
        ).fit(transformed["model_train"], labels["model_train"])
        validation_scores = model.decision_function(transformed["validation"])
        threshold, utility, tool_rate = tune_rescue_gate_threshold(
            validation_scores.tolist(),
            validation_utilities,
        )
        candidates.append((utility, -tool_rate, -float(c_value), threshold, model))
    validation_utility, negative_tool_rate, negative_c, threshold, model = max(
        candidates,
        key=lambda value: value[:3],
    )
    test_scores_array = model.decision_function(transformed["test"])
    test_scores = {
        key: float(score)
        for key, score in zip(split_keys["test"], test_scores_array.tolist())
    }
    policy = PrecomputedRescueGatePolicy(test_scores, threshold=threshold)
    policy_result: dict[str, Any] = dict(
        evaluate_policy(test, policy, lambda_cost=lambda_cost)
    )
    policy_result["bootstrap"] = bootstrap_policy_evaluation(
        test,
        policy,
        lambda_cost=lambda_cost,
        n_resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    test_labels = labels["test"]
    report = {
        "seed": seed,
        "feature_mode": feature_mode,
        "split_group": split_group,
        "train_fraction": train_fraction,
        "validation_fraction_within_train": validation_fraction,
        "model_train_decisions": len(split_keys["model_train"]),
        "validation_decisions": len(split_keys["validation"]),
        "test_decisions": len(split_keys["test"]),
        "feature_count": int(features["model_train"].shape[1]),
        "selected_c": -negative_c,
        "validation_threshold": threshold,
        "validation_utility": validation_utility,
        "validation_tool_rate": -negative_tool_rate,
        "test_helpful_rate": float(test_labels.mean()),
        "test_helpful_roc_auc": float(roc_auc_score(test_labels, test_scores_array)),
        "test_helpful_average_precision": float(
            average_precision_score(test_labels, test_scores_array)
        ),
        "policy_result": policy_result,
    }
    model_payload = {
        "model_type": "compact_rescue_gate_logistic",
        "feature_mode": feature_mode,
        "seed": seed,
        "selected_c": -negative_c,
        "threshold": threshold,
        "scaler_mean": [float(value) for value in scaler.mean_.tolist()],
        "scaler_scale": [float(value) for value in scaler.scale_.tolist()],
        "coefficient": [float(value) for value in model.coef_[0].tolist()],
        "intercept": float(model.intercept_[0]),
    }
    return report, model_payload


def _grouped_crossfit_records(
    records: Sequence[ActionRecord],
    *,
    split_group: str,
    n_folds: int,
    seed: int,
) -> list[tuple[list[ActionRecord], list[ActionRecord]]]:
    if split_group not in ("source_id", "image_id", "state_id"):
        raise ValueError(f"unsupported split group: {split_group}")
    group_ids = sorted({str(getattr(record, split_group)) for record in records})
    if n_folds < 2 or n_folds > len(group_ids):
        raise ValueError("n_folds must be between 2 and the number of groups")
    random.Random(seed).shuffle(group_ids)
    fold_ids = [set(group_ids[index::n_folds]) for index in range(n_folds)]
    folds = []
    for validation_ids in fold_ids:
        training = [
            record
            for record in records
            if str(getattr(record, split_group)) not in validation_ids
        ]
        validation = [
            record
            for record in records
            if str(getattr(record, split_group)) in validation_ids
        ]
        folds.append((training, validation))
    return folds


def fit_crossfit_rescue_gate_split(
    records: Sequence[ActionRecord],
    decision_by_key: Mapping[DecisionKey, Mapping[str, Any]],
    *,
    split_group: str = "image_id",
    train_fraction: float = 0.7,
    n_folds: int = 5,
    lambda_cost: float = 0.05,
    feature_mode: str = "semantic",
    c_values: Sequence[float] = (0.001, 0.01, 0.1, 1.0, 10.0),
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 0,
    seed: int = 17,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Tune on grouped OOF scores and deploy an ensemble of the fold models."""

    import numpy as np  # type: ignore[import-not-found]
    from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
    from sklearn.metrics import (  # type: ignore[import-untyped]
        average_precision_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

    if not c_values or any(value <= 0.0 for value in c_values):
        raise ValueError("c_values must contain positive regularization values")
    outer_train, test = split_by_group(
        records,
        group=split_group,  # type: ignore[arg-type]
        train_fraction=train_fraction,
        seed=seed,
    )
    outer_keys = _keys(outer_train)
    test_keys = _keys(test)
    missing = set(outer_keys + test_keys) - set(decision_by_key)
    if missing:
        raise ValueError(f"semantic features are missing decisions: {sorted(missing)[:5]}")
    outcomes = _decision_outcomes(records, lambda_cost=lambda_cost)
    raw_features = _rescue_feature_map(
        outer_keys + test_keys,
        decision_by_key,
        _decision_baselines(records),
        feature_mode=feature_mode,
    )
    labels = {key: int(bool(outcomes[key]["helpful"])) for key in outer_keys + test_keys}
    folds = _grouped_crossfit_records(
        outer_train,
        split_group=split_group,
        n_folds=n_folds,
        seed=seed + 1,
    )
    candidate_models: list[
        tuple[float, float, float, float, list[tuple[Any, Any]], dict[DecisionKey, float]]
    ] = []
    for c_value in c_values:
        oof_scores: dict[DecisionKey, float] = {}
        ensemble: list[tuple[Any, Any]] = []
        test_fold_scores: list[Any] = []
        for fold_train, fold_validation in folds:
            train_keys = _keys(fold_train)
            validation_keys = _keys(fold_validation)
            scaler = StandardScaler().fit(
                np.asarray([raw_features[key] for key in train_keys], dtype=np.float64)
            )
            model = LogisticRegression(
                C=float(c_value),
                class_weight="balanced",
                solver="liblinear",
                max_iter=2000,
                random_state=seed,
            ).fit(
                scaler.transform(
                    np.asarray([raw_features[key] for key in train_keys], dtype=np.float64)
                ),
                np.asarray([labels[key] for key in train_keys], dtype=np.int64),
            )
            validation_scores = model.decision_function(
                scaler.transform(
                    np.asarray(
                        [raw_features[key] for key in validation_keys],
                        dtype=np.float64,
                    )
                )
            )
            for key, score in zip(validation_keys, validation_scores.tolist()):
                if key in oof_scores:
                    raise RuntimeError(f"duplicate rescue-gate OOF score for {key!r}")
                oof_scores[key] = float(score)
            test_fold_scores.append(
                model.decision_function(
                    scaler.transform(
                        np.asarray([raw_features[key] for key in test_keys], dtype=np.float64)
                    )
                )
            )
            ensemble.append((scaler, model))
        if set(oof_scores) != set(outer_keys):
            raise RuntimeError("rescue-gate OOF scores do not cover every outer-train decision")
        threshold, utility, tool_rate = tune_rescue_gate_threshold(
            [oof_scores[key] for key in outer_keys],
            [float(outcomes[key]["expected_utility"]) for key in outer_keys],
        )
        test_scores_array = np.stack(test_fold_scores).mean(axis=0)
        test_scores = {
            key: float(score) for key, score in zip(test_keys, test_scores_array.tolist())
        }
        candidate_models.append(
            (utility, -tool_rate, -float(c_value), threshold, ensemble, test_scores)
        )
    oof_utility, negative_tool_rate, negative_c, threshold, ensemble, test_scores = max(
        candidate_models,
        key=lambda value: value[:3],
    )
    policy = PrecomputedRescueGatePolicy(test_scores, threshold=threshold)
    policy_result: dict[str, Any] = dict(
        evaluate_policy(test, policy, lambda_cost=lambda_cost)
    )
    policy_result["bootstrap"] = bootstrap_policy_evaluation(
        test,
        policy,
        lambda_cost=lambda_cost,
        n_resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    test_scores_array = np.asarray([test_scores[key] for key in test_keys])
    test_labels = np.asarray([labels[key] for key in test_keys], dtype=np.int64)
    report = {
        "seed": seed,
        "selection_mode": "grouped_oof_fold_ensemble",
        "feature_mode": feature_mode,
        "split_group": split_group,
        "train_fraction": train_fraction,
        "outer_train_decisions": len(outer_keys),
        "test_decisions": len(test_keys),
        "n_folds": n_folds,
        "feature_count": len(raw_features[outer_keys[0]]),
        "selected_c": -negative_c,
        "oof_threshold": threshold,
        "oof_utility": oof_utility,
        "oof_tool_rate": -negative_tool_rate,
        "test_helpful_rate": float(test_labels.mean()),
        "test_helpful_roc_auc": float(roc_auc_score(test_labels, test_scores_array)),
        "test_helpful_average_precision": float(
            average_precision_score(test_labels, test_scores_array)
        ),
        "policy_result": policy_result,
    }
    ensemble_payload = []
    for scaler, model in ensemble:
        ensemble_payload.append(
            {
                "scaler_mean": [float(value) for value in scaler.mean_.tolist()],
                "scaler_scale": [float(value) for value in scaler.scale_.tolist()],
                "coefficient": [float(value) for value in model.coef_[0].tolist()],
                "intercept": float(model.intercept_[0]),
            }
        )
    model_payload = {
        "model_type": "compact_rescue_gate_grouped_fold_ensemble",
        "feature_mode": feature_mode,
        "seed": seed,
        "selected_c": -negative_c,
        "threshold": threshold,
        "n_folds": n_folds,
        "fold_models": ensemble_payload,
    }
    return report, model_payload


def fit_expected_gain_rescue_gate_split(
    records: Sequence[ActionRecord],
    decision_by_key: Mapping[DecisionKey, Mapping[str, Any]],
    *,
    split_group: str = "image_id",
    train_fraction: float = 0.7,
    validation_fraction: float = 0.2,
    lambda_cost: float = 0.05,
    feature_mode: str = "semantic",
    alpha_values: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 0,
    seed: int = 17,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Regress expected one-crop gain, then tune a cost gate on validation only."""

    import numpy as np  # type: ignore[import-not-found]
    from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
    from sklearn.metrics import (  # type: ignore[import-untyped]
        average_precision_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

    if not alpha_values or any(value <= 0.0 for value in alpha_values):
        raise ValueError("alpha_values must contain positive regularization values")
    outer_train, test = split_by_group(
        records,
        group=split_group,  # type: ignore[arg-type]
        train_fraction=train_fraction,
        seed=seed,
    )
    model_train, validation = split_by_group(
        outer_train,
        group=split_group,  # type: ignore[arg-type]
        train_fraction=1.0 - validation_fraction,
        seed=seed + 1,
    )
    split_keys = {
        "model_train": _keys(model_train),
        "validation": _keys(validation),
        "test": _keys(test),
    }
    missing = set().union(*map(set, split_keys.values())) - set(decision_by_key)
    if missing:
        raise ValueError(f"rescue features are missing decisions: {sorted(missing)[:5]}")
    outcomes = _decision_outcomes(records, lambda_cost=lambda_cost)
    baselines = _decision_baselines(records)
    feature_by_key = _rescue_feature_map(
        [key for keys in split_keys.values() for key in keys],
        decision_by_key,
        baselines,
        feature_mode=feature_mode,
    )
    features = {
        name: np.asarray(
            [feature_by_key[key] for key in keys],
            dtype=np.float64,
        )
        for name, keys in split_keys.items()
    }
    scaler = StandardScaler().fit(features["model_train"])
    transformed = {
        name: scaler.transform(values) for name, values in features.items()
    }
    train_targets = np.asarray(
        [float(outcomes[key]["expected_gain"]) for key in split_keys["model_train"]],
        dtype=np.float64,
    )
    validation_utilities = [
        float(outcomes[key]["expected_utility"]) for key in split_keys["validation"]
    ]
    candidates: list[tuple[float, float, float, float, Any]] = []
    for alpha in alpha_values:
        model = Ridge(alpha=float(alpha)).fit(transformed["model_train"], train_targets)
        validation_scores = model.predict(transformed["validation"])
        threshold, utility, tool_rate = tune_rescue_gate_threshold(
            validation_scores.tolist(),
            validation_utilities,
        )
        candidates.append((utility, -tool_rate, -float(alpha), threshold, model))
    validation_utility, negative_tool_rate, negative_alpha, threshold, model = max(
        candidates,
        key=lambda value: value[:3],
    )
    test_scores_array = model.predict(transformed["test"])
    test_scores = {
        key: float(score)
        for key, score in zip(split_keys["test"], test_scores_array.tolist())
    }
    policy = PrecomputedRescueGatePolicy(
        test_scores,
        threshold=threshold,
        name="compact_expected_gain_gate_uniform_random_expectation",
    )
    policy_result: dict[str, Any] = dict(
        evaluate_policy(test, policy, lambda_cost=lambda_cost)
    )
    policy_result["bootstrap"] = bootstrap_policy_evaluation(
        test,
        policy,
        lambda_cost=lambda_cost,
        n_resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    test_labels = np.asarray(
        [bool(outcomes[key]["helpful"]) for key in split_keys["test"]],
        dtype=np.int64,
    )
    report = {
        "seed": seed,
        "selection_mode": "inner_validation_expected_gain_ridge",
        "feature_mode": feature_mode,
        "split_group": split_group,
        "train_fraction": train_fraction,
        "validation_fraction_within_train": validation_fraction,
        "model_train_decisions": len(split_keys["model_train"]),
        "validation_decisions": len(split_keys["validation"]),
        "test_decisions": len(split_keys["test"]),
        "feature_count": int(features["model_train"].shape[1]),
        "selected_alpha": -negative_alpha,
        "validation_threshold": threshold,
        "validation_utility": validation_utility,
        "validation_tool_rate": -negative_tool_rate,
        "test_helpful_rate": float(test_labels.mean()),
        "test_helpful_roc_auc": float(roc_auc_score(test_labels, test_scores_array)),
        "test_helpful_average_precision": float(
            average_precision_score(test_labels, test_scores_array)
        ),
        "policy_result": policy_result,
    }
    model_payload = {
        "model_type": "compact_expected_random_gain_ridge",
        "feature_mode": feature_mode,
        "seed": seed,
        "selected_alpha": -negative_alpha,
        "threshold": threshold,
        "scaler_mean": [float(value) for value in scaler.mean_.tolist()],
        "scaler_scale": [float(value) for value in scaler.scale_.tolist()],
        "coefficient": [float(value) for value in model.coef_.tolist()],
        "intercept": float(model.intercept_),
    }
    return report, model_payload


def fit_nested_oof_rescue_gate(
    records: Sequence[ActionRecord],
    decision_by_key: Mapping[DecisionKey, Mapping[str, Any]],
    *,
    split_group: str = "image_id",
    n_outer_folds: int = 5,
    validation_fraction: float = 0.2,
    lambda_cost: float = 0.05,
    feature_mode: str = "semantic-context",
    c_values: Sequence[float] = (0.001, 0.01, 0.1, 1.0, 10.0),
    bootstrap_resamples: int = 5000,
    bootstrap_seed: int = 0,
    seed: int = 17,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate a logistic rescue gate with disjoint nested outer test folds."""

    import numpy as np  # type: ignore[import-not-found]
    from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
    from sklearn.metrics import (  # type: ignore[import-untyped]
        average_precision_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

    if not c_values or any(value <= 0.0 for value in c_values):
        raise ValueError("c_values must contain positive regularization values")
    all_keys = _keys(records)
    missing = set(all_keys) - set(decision_by_key)
    if missing:
        raise ValueError(f"rescue features are missing decisions: {sorted(missing)[:5]}")
    outcomes = _decision_outcomes(records, lambda_cost=lambda_cost)
    raw_features = _rescue_feature_map(
        all_keys,
        decision_by_key,
        _decision_baselines(records),
        feature_mode=feature_mode,
    )
    labels = {key: int(bool(outcomes[key]["helpful"])) for key in all_keys}
    outer_folds = _grouped_crossfit_records(
        records,
        split_group=split_group,
        n_folds=n_outer_folds,
        seed=seed,
    )
    pooled_actions: dict[DecisionKey, float] = {}
    pooled_margins: dict[DecisionKey, float] = {}
    fold_reports: list[dict[str, Any]] = []
    fold_models: list[dict[str, Any]] = []
    for fold_index, (outer_train, outer_test) in enumerate(outer_folds):
        model_train, validation = split_by_group(
            outer_train,
            group=split_group,  # type: ignore[arg-type]
            train_fraction=1.0 - validation_fraction,
            seed=seed + 101 + fold_index,
        )
        train_keys = _keys(model_train)
        validation_keys = _keys(validation)
        test_keys = _keys(outer_test)
        scaler = StandardScaler().fit(
            np.asarray([raw_features[key] for key in train_keys], dtype=np.float64)
        )
        train_features = scaler.transform(
            np.asarray([raw_features[key] for key in train_keys], dtype=np.float64)
        )
        validation_features = scaler.transform(
            np.asarray([raw_features[key] for key in validation_keys], dtype=np.float64)
        )
        test_features = scaler.transform(
            np.asarray([raw_features[key] for key in test_keys], dtype=np.float64)
        )
        train_labels = np.asarray([labels[key] for key in train_keys], dtype=np.int64)
        validation_utilities = [
            float(outcomes[key]["expected_utility"]) for key in validation_keys
        ]
        candidates: list[tuple[float, float, float, float, Any]] = []
        for c_value in c_values:
            model = LogisticRegression(
                C=float(c_value),
                class_weight="balanced",
                solver="liblinear",
                max_iter=2000,
                random_state=seed + fold_index,
            ).fit(train_features, train_labels)
            validation_scores = model.decision_function(validation_features)
            threshold, utility, tool_rate = tune_rescue_gate_threshold(
                validation_scores.tolist(),
                validation_utilities,
            )
            candidates.append((utility, -tool_rate, -float(c_value), threshold, model))
        validation_utility, negative_tool_rate, negative_c, threshold, model = max(
            candidates,
            key=lambda value: value[:3],
        )
        test_scores_array = model.decision_function(test_features)
        test_scores = {
            key: float(score)
            for key, score in zip(test_keys, test_scores_array.tolist())
        }
        overlap = set(test_scores) & set(pooled_actions)
        if overlap:
            raise RuntimeError(f"nested OOF test decisions overlap: {sorted(overlap)[:5]}")
        for key, score in test_scores.items():
            pooled_actions[key] = float(score >= threshold)
            pooled_margins[key] = score - threshold
        fold_policy = PrecomputedRescueGatePolicy(
            test_scores,
            threshold=threshold,
            name=f"nested_oof_{feature_mode}_uniform_random_expectation",
        )
        fold_result = dict(evaluate_policy(outer_test, fold_policy, lambda_cost=lambda_cost))
        test_labels = np.asarray([labels[key] for key in test_keys], dtype=np.int64)
        has_both_test_classes = len(set(test_labels.tolist())) == 2
        fold_reports.append(
            {
                "fold": fold_index,
                "model_train_decisions": len(train_keys),
                "validation_decisions": len(validation_keys),
                "test_decisions": len(test_keys),
                "selected_c": -negative_c,
                "validation_threshold": threshold,
                "validation_utility": validation_utility,
                "validation_tool_rate": -negative_tool_rate,
                "test_helpful_roc_auc": (
                    float(roc_auc_score(test_labels, test_scores_array))
                    if has_both_test_classes
                    else None
                ),
                "test_helpful_average_precision": (
                    float(average_precision_score(test_labels, test_scores_array))
                    if has_both_test_classes
                    else None
                ),
                "policy_result": fold_result,
            }
        )
        fold_models.append(
            {
                "fold": fold_index,
                "selected_c": -negative_c,
                "threshold": threshold,
                "scaler_mean": [float(value) for value in scaler.mean_.tolist()],
                "scaler_scale": [float(value) for value in scaler.scale_.tolist()],
                "coefficient": [float(value) for value in model.coef_[0].tolist()],
                "intercept": float(model.intercept_[0]),
            }
        )
    if set(pooled_actions) != set(all_keys):
        missing_oof = sorted(set(all_keys) - set(pooled_actions))
        raise RuntimeError(f"nested OOF predictions are incomplete: {missing_oof[:5]}")
    pooled_policy = PrecomputedRescueGatePolicy(
        pooled_actions,
        threshold=0.5,
        name=f"nested_oof_{feature_mode}_uniform_random_expectation",
    )
    pooled_result: dict[str, Any] = dict(
        evaluate_policy(records, pooled_policy, lambda_cost=lambda_cost)
    )
    pooled_result["bootstrap"] = bootstrap_policy_evaluation(
        records,
        pooled_policy,
        lambda_cost=lambda_cost,
        n_resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    pooled_labels = np.asarray([labels[key] for key in all_keys], dtype=np.int64)
    pooled_margin_array = np.asarray([pooled_margins[key] for key in all_keys])
    report = {
        "scientific_status": (
            "nested grouped OOF diagnostic; each decision is evaluated once by a model "
            "that excluded its image group"
        ),
        "seed": seed,
        "feature_mode": feature_mode,
        "feature_count": len(raw_features[all_keys[0]]),
        "split_group": split_group,
        "n_outer_folds": n_outer_folds,
        "validation_fraction_within_outer_train": validation_fraction,
        "n_decisions": len(all_keys),
        "pooled_helpful_roc_auc_of_fold_margin": float(
            roc_auc_score(pooled_labels, pooled_margin_array)
        ),
        "pooled_helpful_average_precision_of_fold_margin": float(
            average_precision_score(pooled_labels, pooled_margin_array)
        ),
        "folds": fold_reports,
        "policy_result": pooled_result,
    }
    model_payload = {
        "model_type": "compact_rescue_gate_nested_grouped_oof",
        "seed": seed,
        "feature_mode": feature_mode,
        "n_outer_folds": n_outer_folds,
        "fold_models": fold_models,
    }
    return report, model_payload


def fit_nested_oof_entropy_gate(
    records: Sequence[ActionRecord],
    *,
    split_group: str = "image_id",
    n_outer_folds: int = 5,
    lambda_cost: float = 0.05,
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 0,
    seed: int = 17,
) -> dict[str, Any]:
    """Evaluate entropy stopping with thresholds tuned outside each outer fold."""

    all_keys = _keys(records)
    baselines = _decision_baselines(records)
    outcomes = _decision_outcomes(records, lambda_cost=lambda_cost)
    outer_folds = _grouped_crossfit_records(
        records,
        split_group=split_group,
        n_folds=n_outer_folds,
        seed=seed,
    )
    pooled_actions: dict[DecisionKey, float] = {}
    fold_reports = []
    for fold_index, (outer_train, outer_test) in enumerate(outer_folds):
        train_keys = _keys(outer_train)
        test_keys = _keys(outer_test)
        threshold, train_utility, train_tool_rate = tune_rescue_gate_threshold(
            [baselines[key].entropy_before for key in train_keys],
            [float(outcomes[key]["expected_utility"]) for key in train_keys],
        )
        test_scores = {key: baselines[key].entropy_before for key in test_keys}
        overlap = set(test_scores) & set(pooled_actions)
        if overlap:
            raise RuntimeError(f"nested OOF test decisions overlap: {sorted(overlap)[:5]}")
        pooled_actions.update(
            {key: float(score >= threshold) for key, score in test_scores.items()}
        )
        fold_policy = PrecomputedRescueGatePolicy(
            test_scores,
            threshold=threshold,
            name="nested_oof_entropy_uniform_random_expectation",
        )
        fold_reports.append(
            {
                "fold": fold_index,
                "train_decisions": len(train_keys),
                "test_decisions": len(test_keys),
                "train_threshold": threshold,
                "train_utility": train_utility,
                "train_tool_rate": train_tool_rate,
                "policy_result": dict(
                    evaluate_policy(outer_test, fold_policy, lambda_cost=lambda_cost)
                ),
            }
        )
    if set(pooled_actions) != set(all_keys):
        missing_oof = sorted(set(all_keys) - set(pooled_actions))
        raise RuntimeError(f"nested OOF predictions are incomplete: {missing_oof[:5]}")
    policy = PrecomputedRescueGatePolicy(
        pooled_actions,
        threshold=0.5,
        name="nested_oof_entropy_uniform_random_expectation",
    )
    policy_result: dict[str, Any] = dict(
        evaluate_policy(records, policy, lambda_cost=lambda_cost)
    )
    policy_result["bootstrap"] = bootstrap_policy_evaluation(
        records,
        policy,
        lambda_cost=lambda_cost,
        n_resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    return {
        "scientific_status": (
            "nested grouped OOF scalar baseline; each entropy threshold excludes its "
            "outer test image groups"
        ),
        "seed": seed,
        "split_group": split_group,
        "n_outer_folds": n_outer_folds,
        "n_decisions": len(all_keys),
        "folds": fold_reports,
        "policy_result": policy_result,
    }


def fit_nested_oof_factorized_rescue_gate(
    records: Sequence[ActionRecord],
    decision_by_key: Mapping[DecisionKey, Mapping[str, Any]],
    *,
    split_group: str = "image_id",
    n_outer_folds: int = 5,
    validation_fraction: float = 0.2,
    lambda_cost: float = 0.05,
    error_feature_mode: str = "context",
    rescue_feature_mode: str = "semantic-context",
    c_values: Sequence[float] = (0.001, 0.01, 0.1, 1.0, 10.0),
    bootstrap_resamples: int = 5000,
    bootstrap_seed: int = 0,
    seed: int = 17,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Factor helpfulness into baseline error and conditional rescuability."""

    import numpy as np  # type: ignore[import-not-found]
    from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
    from sklearn.metrics import (  # type: ignore[import-untyped]
        average_precision_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

    if not c_values or any(value <= 0.0 for value in c_values):
        raise ValueError("c_values must contain positive regularization values")
    all_keys = _keys(records)
    missing = set(all_keys) - set(decision_by_key)
    if missing:
        raise ValueError(f"rescue features are missing decisions: {sorted(missing)[:5]}")
    baselines = _decision_baselines(records)
    outcomes = _decision_outcomes(records, lambda_cost=lambda_cost)
    error_features = _rescue_feature_map(
        all_keys,
        decision_by_key,
        baselines,
        feature_mode=error_feature_mode,
    )
    rescue_features = _rescue_feature_map(
        all_keys,
        decision_by_key,
        baselines,
        feature_mode=rescue_feature_mode,
    )
    error_labels = {
        key: int(baselines[key].correct_before < 0.5) for key in all_keys
    }
    rescue_labels = {key: int(bool(outcomes[key]["helpful"])) for key in all_keys}
    outer_folds = _grouped_crossfit_records(
        records,
        split_group=split_group,
        n_folds=n_outer_folds,
        seed=seed,
    )
    pooled_actions: dict[DecisionKey, float] = {}
    pooled_scores: dict[DecisionKey, float] = {}
    pooled_error_probabilities: dict[DecisionKey, float] = {}
    pooled_rescue_probabilities: dict[DecisionKey, float] = {}
    fold_reports: list[dict[str, Any]] = []
    fold_models: list[dict[str, Any]] = []
    for fold_index, (outer_train, outer_test) in enumerate(outer_folds):
        model_train, validation = split_by_group(
            outer_train,
            group=split_group,  # type: ignore[arg-type]
            train_fraction=1.0 - validation_fraction,
            seed=seed + 307 + fold_index,
        )
        train_keys = _keys(model_train)
        validation_keys = _keys(validation)
        test_keys = _keys(outer_test)
        wrong_train_keys = [key for key in train_keys if error_labels[key]]
        error_scaler = StandardScaler().fit(
            np.asarray([error_features[key] for key in train_keys], dtype=np.float64)
        )
        rescue_scaler = StandardScaler().fit(
            np.asarray(
                [rescue_features[key] for key in wrong_train_keys],
                dtype=np.float64,
            )
        )
        error_train = error_scaler.transform(
            np.asarray([error_features[key] for key in train_keys], dtype=np.float64)
        )
        error_validation = error_scaler.transform(
            np.asarray(
                [error_features[key] for key in validation_keys],
                dtype=np.float64,
            )
        )
        error_test = error_scaler.transform(
            np.asarray([error_features[key] for key in test_keys], dtype=np.float64)
        )
        rescue_train = rescue_scaler.transform(
            np.asarray(
                [rescue_features[key] for key in wrong_train_keys],
                dtype=np.float64,
            )
        )
        rescue_validation = rescue_scaler.transform(
            np.asarray(
                [rescue_features[key] for key in validation_keys],
                dtype=np.float64,
            )
        )
        rescue_test = rescue_scaler.transform(
            np.asarray([rescue_features[key] for key in test_keys], dtype=np.float64)
        )
        error_train_labels = np.asarray(
            [error_labels[key] for key in train_keys],
            dtype=np.int64,
        )
        rescue_train_labels = np.asarray(
            [rescue_labels[key] for key in wrong_train_keys],
            dtype=np.int64,
        )
        error_models = []
        rescue_models = []
        for c_value in c_values:
            error_model = LogisticRegression(
                C=float(c_value),
                class_weight="balanced",
                solver="liblinear",
                max_iter=2000,
                random_state=seed + fold_index,
            ).fit(error_train, error_train_labels)
            error_models.append(
                (
                    float(c_value),
                    error_model,
                    error_model.predict_proba(error_validation)[:, 1],
                )
            )
            rescue_model = LogisticRegression(
                C=float(c_value),
                class_weight="balanced",
                solver="liblinear",
                max_iter=2000,
                random_state=seed + fold_index,
            ).fit(rescue_train, rescue_train_labels)
            rescue_models.append(
                (
                    float(c_value),
                    rescue_model,
                    rescue_model.predict_proba(rescue_validation)[:, 1],
                )
            )
        validation_utilities = [
            float(outcomes[key]["expected_utility"]) for key in validation_keys
        ]
        candidates: list[tuple[float, float, float, float, float, Any, Any]] = []
        for error_c, error_model, validation_error_probabilities in error_models:
            for rescue_c, rescue_model, validation_rescue_probabilities in rescue_models:
                validation_scores = (
                    validation_error_probabilities * validation_rescue_probabilities
                )
                threshold, utility, tool_rate = tune_rescue_gate_threshold(
                    validation_scores.tolist(),
                    validation_utilities,
                )
                candidates.append(
                    (
                        utility,
                        -tool_rate,
                        -error_c,
                        -rescue_c,
                        threshold,
                        error_model,
                        rescue_model,
                    )
                )
        (
            validation_utility,
            negative_tool_rate,
            negative_error_c,
            negative_rescue_c,
            threshold,
            error_model,
            rescue_model,
        ) = max(candidates, key=lambda value: value[:4])
        test_error_probabilities = error_model.predict_proba(error_test)[:, 1]
        test_rescue_probabilities = rescue_model.predict_proba(rescue_test)[:, 1]
        test_scores_array = test_error_probabilities * test_rescue_probabilities
        test_scores = {
            key: float(score)
            for key, score in zip(test_keys, test_scores_array.tolist())
        }
        overlap = set(test_scores) & set(pooled_actions)
        if overlap:
            raise RuntimeError(f"nested OOF test decisions overlap: {sorted(overlap)[:5]}")
        for key, score, error_probability, rescue_probability in zip(
            test_keys,
            test_scores_array.tolist(),
            test_error_probabilities.tolist(),
            test_rescue_probabilities.tolist(),
        ):
            pooled_actions[key] = float(score >= threshold)
            pooled_scores[key] = float(score) - threshold
            pooled_error_probabilities[key] = float(error_probability)
            pooled_rescue_probabilities[key] = float(rescue_probability)
        fold_policy = PrecomputedRescueGatePolicy(
            test_scores,
            threshold=threshold,
            name=f"nested_oof_factorized_{rescue_feature_mode}_random_expectation",
        )
        fold_result = dict(evaluate_policy(outer_test, fold_policy, lambda_cost=lambda_cost))
        test_error_labels = np.asarray(
            [error_labels[key] for key in test_keys],
            dtype=np.int64,
        )
        test_helpful_labels = np.asarray(
            [rescue_labels[key] for key in test_keys],
            dtype=np.int64,
        )
        wrong_test_indices = [
            index for index, key in enumerate(test_keys) if error_labels[key]
        ]
        wrong_test_labels = test_helpful_labels[wrong_test_indices]
        wrong_test_rescue_scores = test_rescue_probabilities[wrong_test_indices]
        has_both_error_classes = len(set(test_error_labels.tolist())) == 2
        has_both_helpful_classes = len(set(test_helpful_labels.tolist())) == 2
        has_both_conditional_classes = len(set(wrong_test_labels.tolist())) == 2
        fold_reports.append(
            {
                "fold": fold_index,
                "model_train_decisions": len(train_keys),
                "model_train_wrong_decisions": len(wrong_train_keys),
                "validation_decisions": len(validation_keys),
                "test_decisions": len(test_keys),
                "selected_error_c": -negative_error_c,
                "selected_rescue_c": -negative_rescue_c,
                "validation_threshold": threshold,
                "validation_utility": validation_utility,
                "validation_tool_rate": -negative_tool_rate,
                "test_error_roc_auc": (
                    float(roc_auc_score(test_error_labels, test_error_probabilities))
                    if has_both_error_classes
                    else None
                ),
                "test_conditional_rescue_roc_auc": (
                    float(roc_auc_score(wrong_test_labels, wrong_test_rescue_scores))
                    if has_both_conditional_classes
                    else None
                ),
                "test_helpful_roc_auc": (
                    float(roc_auc_score(test_helpful_labels, test_scores_array))
                    if has_both_helpful_classes
                    else None
                ),
                "test_helpful_average_precision": (
                    float(average_precision_score(test_helpful_labels, test_scores_array))
                    if has_both_helpful_classes
                    else None
                ),
                "policy_result": fold_result,
            }
        )
        fold_models.append(
            {
                "fold": fold_index,
                "selected_error_c": -negative_error_c,
                "selected_rescue_c": -negative_rescue_c,
                "threshold": threshold,
                "error_scaler_mean": [
                    float(value) for value in error_scaler.mean_.tolist()
                ],
                "error_scaler_scale": [
                    float(value) for value in error_scaler.scale_.tolist()
                ],
                "error_coefficient": [
                    float(value) for value in error_model.coef_[0].tolist()
                ],
                "error_intercept": float(error_model.intercept_[0]),
                "rescue_scaler_mean": [
                    float(value) for value in rescue_scaler.mean_.tolist()
                ],
                "rescue_scaler_scale": [
                    float(value) for value in rescue_scaler.scale_.tolist()
                ],
                "rescue_coefficient": [
                    float(value) for value in rescue_model.coef_[0].tolist()
                ],
                "rescue_intercept": float(rescue_model.intercept_[0]),
            }
        )
    if set(pooled_actions) != set(all_keys):
        missing_oof = sorted(set(all_keys) - set(pooled_actions))
        raise RuntimeError(f"nested OOF predictions are incomplete: {missing_oof[:5]}")
    policy = PrecomputedRescueGatePolicy(
        pooled_actions,
        threshold=0.5,
        name=f"nested_oof_factorized_{rescue_feature_mode}_random_expectation",
    )
    policy_result: dict[str, Any] = dict(
        evaluate_policy(records, policy, lambda_cost=lambda_cost)
    )
    policy_result["bootstrap"] = bootstrap_policy_evaluation(
        records,
        policy,
        lambda_cost=lambda_cost,
        n_resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    pooled_error_labels = np.asarray(
        [error_labels[key] for key in all_keys],
        dtype=np.int64,
    )
    pooled_helpful_labels = np.asarray(
        [rescue_labels[key] for key in all_keys],
        dtype=np.int64,
    )
    wrong_keys = [key for key in all_keys if error_labels[key]]
    report = {
        "scientific_status": (
            "nested grouped OOF factorized diagnostic; error and conditional-rescue "
            "models both exclude each evaluated image group"
        ),
        "seed": seed,
        "error_feature_mode": error_feature_mode,
        "rescue_feature_mode": rescue_feature_mode,
        "error_feature_count": len(error_features[all_keys[0]]),
        "rescue_feature_count": len(rescue_features[all_keys[0]]),
        "split_group": split_group,
        "n_outer_folds": n_outer_folds,
        "validation_fraction_within_outer_train": validation_fraction,
        "n_decisions": len(all_keys),
        "pooled_error_roc_auc": float(
            roc_auc_score(
                pooled_error_labels,
                [pooled_error_probabilities[key] for key in all_keys],
            )
        ),
        "pooled_conditional_rescue_roc_auc": float(
            roc_auc_score(
                [rescue_labels[key] for key in wrong_keys],
                [pooled_rescue_probabilities[key] for key in wrong_keys],
            )
        ),
        "pooled_helpful_roc_auc_of_fold_margin": float(
            roc_auc_score(
                pooled_helpful_labels,
                [pooled_scores[key] for key in all_keys],
            )
        ),
        "pooled_helpful_average_precision_of_fold_margin": float(
            average_precision_score(
                pooled_helpful_labels,
                [pooled_scores[key] for key in all_keys],
            )
        ),
        "folds": fold_reports,
        "policy_result": policy_result,
    }
    model_payload = {
        "model_type": "nested_oof_factorized_error_and_rescue_gate",
        "seed": seed,
        "error_feature_mode": error_feature_mode,
        "rescue_feature_mode": rescue_feature_mode,
        "n_outer_folds": n_outer_folds,
        "fold_models": fold_models,
    }
    return report, model_payload


def fit_nested_oof_two_stage_gate(
    records: Sequence[ActionRecord],
    decision_by_key: Mapping[DecisionKey, Mapping[str, Any]],
    *,
    split_group: str = "image_id",
    n_outer_folds: int = 5,
    validation_fraction: float = 0.2,
    lambda_cost: float = 0.05,
    state_feature_mode: str = "context",
    c_values: Sequence[float] = (0.001, 0.01, 0.1, 1.0, 10.0),
    bootstrap_resamples: int = 5000,
    bootstrap_seed: int = 0,
    seed: int = 17,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Nested OOF state gating plus learned concrete-crop selection."""

    import numpy as np  # type: ignore[import-not-found]
    from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
    from sklearn.metrics import (  # type: ignore[import-untyped]
        average_precision_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

    if not c_values or any(value <= 0.0 for value in c_values):
        raise ValueError("c_values must contain positive regularization values")
    grouped = group_by_decision(records)
    all_keys = sorted(grouped)
    missing = set(all_keys) - set(decision_by_key)
    if missing:
        raise ValueError(f"rescue features are missing decisions: {sorted(missing)[:5]}")
    baselines = _decision_baselines(records)
    outcomes = _decision_outcomes(records, lambda_cost=lambda_cost)
    state_features = _rescue_feature_map(
        all_keys,
        decision_by_key,
        baselines,
        feature_mode=state_feature_mode,
    )
    state_labels = {key: int(bool(outcomes[key]["helpful"])) for key in all_keys}
    zooms_by_key: dict[DecisionKey, list[ActionRecord]] = {}
    action_features: dict[tuple[DecisionKey, str], list[float]] = {}
    for key in all_keys:
        zooms = sorted(
            (record for record in grouped[key] if record.action_type == "ZOOM"),
            key=lambda record: record.action_id,
        )
        expected_ids = list(decision_by_key[key]["action_ids"])
        if [record.action_id for record in zooms] != expected_ids:
            raise ValueError(f"semantic action IDs differ for decision {key!r}")
        zooms_by_key[key] = zooms
        for action_index, record in enumerate(zooms):
            action_features[(key, record.action_id)] = compact_action_features(
                decision_by_key[key],
                action_index,
            )
    outer_folds = _grouped_crossfit_records(
        records,
        split_group=split_group,
        n_folds=n_outer_folds,
        seed=seed,
    )
    pooled_selected_actions: dict[DecisionKey, str | None] = {}
    pooled_top_actions: dict[DecisionKey, str] = {}
    pooled_state_margins: dict[DecisionKey, float] = {}
    fold_reports: list[dict[str, Any]] = []
    fold_models: list[dict[str, Any]] = []
    for fold_index, (outer_train, outer_test) in enumerate(outer_folds):
        model_train, validation = split_by_group(
            outer_train,
            group=split_group,  # type: ignore[arg-type]
            train_fraction=1.0 - validation_fraction,
            seed=seed + 211 + fold_index,
        )
        train_keys = _keys(model_train)
        validation_keys = _keys(validation)
        test_keys = _keys(outer_test)
        state_scaler = StandardScaler().fit(
            np.asarray([state_features[key] for key in train_keys], dtype=np.float64)
        )
        state_train_features = state_scaler.transform(
            np.asarray([state_features[key] for key in train_keys], dtype=np.float64)
        )
        state_validation_features = state_scaler.transform(
            np.asarray([state_features[key] for key in validation_keys], dtype=np.float64)
        )
        state_test_features = state_scaler.transform(
            np.asarray([state_features[key] for key in test_keys], dtype=np.float64)
        )
        state_train_labels = np.asarray(
            [state_labels[key] for key in train_keys],
            dtype=np.int64,
        )
        state_models = []
        for state_c in c_values:
            state_model = LogisticRegression(
                C=float(state_c),
                class_weight="balanced",
                solver="liblinear",
                max_iter=2000,
                random_state=seed + fold_index,
            ).fit(state_train_features, state_train_labels)
            state_models.append(
                (
                    float(state_c),
                    state_model,
                    state_model.decision_function(state_validation_features),
                )
            )

        action_train_rows = [
            (key, zoom)
            for key in train_keys
            if baselines[key].correct_before < 0.5
            for zoom in zooms_by_key[key]
        ]
        action_scaler = StandardScaler().fit(
            np.asarray(
                [action_features[(key, zoom.action_id)] for key, zoom in action_train_rows],
                dtype=np.float64,
            )
        )
        action_train_features = action_scaler.transform(
            np.asarray(
                [action_features[(key, zoom.action_id)] for key, zoom in action_train_rows],
                dtype=np.float64,
            )
        )
        action_train_labels = np.asarray(
            [zoom.delta_success > 0.0 for _, zoom in action_train_rows],
            dtype=np.int64,
        )
        action_models = []
        for action_c in c_values:
            action_model = LogisticRegression(
                C=float(action_c),
                class_weight="balanced",
                solver="liblinear",
                max_iter=2000,
                random_state=seed + fold_index,
            ).fit(action_train_features, action_train_labels)
            validation_top_actions = {}
            for key in validation_keys:
                candidate_features = action_scaler.transform(
                    np.asarray(
                        [
                            action_features[(key, zoom.action_id)]
                            for zoom in zooms_by_key[key]
                        ],
                        dtype=np.float64,
                    )
                )
                candidate_scores = action_model.decision_function(candidate_features)
                selected_index = max(
                    range(len(zooms_by_key[key])),
                    key=lambda index: (
                        float(candidate_scores[index]),
                        zooms_by_key[key][index].action_id,
                    ),
                )
                validation_top_actions[key] = zooms_by_key[key][selected_index]
            action_models.append((float(action_c), action_model, validation_top_actions))

        candidates: list[tuple[float, float, float, float, float, Any, Any]] = []
        for state_c, state_model, validation_state_scores in state_models:
            for action_c, action_model, validation_top_actions in action_models:
                threshold, utility, tool_rate = tune_rescue_gate_threshold(
                    validation_state_scores.tolist(),
                    [
                        validation_top_actions[key].voi(lambda_cost)
                        for key in validation_keys
                    ],
                )
                candidates.append(
                    (
                        utility,
                        -tool_rate,
                        -state_c,
                        -action_c,
                        threshold,
                        state_model,
                        action_model,
                    )
                )
        (
            validation_utility,
            negative_tool_rate,
            negative_state_c,
            negative_action_c,
            threshold,
            state_model,
            action_model,
        ) = max(candidates, key=lambda value: value[:4])
        test_state_scores = state_model.decision_function(state_test_features)
        fold_selected_actions: dict[DecisionKey, str | None] = {}
        fold_top_actions: dict[DecisionKey, str] = {}
        test_action_labels: list[int] = []
        test_action_scores: list[float] = []
        for key, state_score in zip(test_keys, test_state_scores.tolist()):
            candidate_features = action_scaler.transform(
                np.asarray(
                    [action_features[(key, zoom.action_id)] for zoom in zooms_by_key[key]],
                    dtype=np.float64,
                )
            )
            candidate_scores = action_model.decision_function(candidate_features)
            selected_index = max(
                range(len(zooms_by_key[key])),
                key=lambda index: (
                    float(candidate_scores[index]),
                    zooms_by_key[key][index].action_id,
                ),
            )
            top_action_id = zooms_by_key[key][selected_index].action_id
            fold_top_actions[key] = top_action_id
            fold_selected_actions[key] = (
                top_action_id if float(state_score) >= threshold else None
            )
            pooled_state_margins[key] = float(state_score) - threshold
            if baselines[key].correct_before < 0.5:
                test_action_labels.extend(
                    int(zoom.delta_success > 0.0) for zoom in zooms_by_key[key]
                )
                test_action_scores.extend(float(score) for score in candidate_scores.tolist())
        overlap = set(fold_selected_actions) & set(pooled_selected_actions)
        if overlap:
            raise RuntimeError(f"nested OOF test decisions overlap: {sorted(overlap)[:5]}")
        pooled_selected_actions.update(fold_selected_actions)
        pooled_top_actions.update(fold_top_actions)
        fold_policy = PrecomputedActionGatePolicy(
            fold_selected_actions,
            name="nested_oof_two_stage_concrete_crop",
        )
        fold_result = dict(evaluate_policy(outer_test, fold_policy, lambda_cost=lambda_cost))
        test_state_labels = np.asarray(
            [state_labels[key] for key in test_keys],
            dtype=np.int64,
        )
        has_both_state_classes = len(set(test_state_labels.tolist())) == 2
        has_both_action_classes = len(set(test_action_labels)) == 2
        helpful_keys = [key for key in test_keys if state_labels[key]]
        fold_reports.append(
            {
                "fold": fold_index,
                "model_train_decisions": len(train_keys),
                "validation_decisions": len(validation_keys),
                "test_decisions": len(test_keys),
                "selected_state_c": -negative_state_c,
                "selected_action_c": -negative_action_c,
                "validation_threshold": threshold,
                "validation_utility": validation_utility,
                "validation_tool_rate": -negative_tool_rate,
                "test_state_helpful_roc_auc": (
                    float(roc_auc_score(test_state_labels, test_state_scores))
                    if has_both_state_classes
                    else None
                ),
                "test_action_rescue_roc_auc_on_wrong_states": (
                    float(roc_auc_score(test_action_labels, test_action_scores))
                    if has_both_action_classes
                    else None
                ),
                "test_action_rescue_average_precision_on_wrong_states": (
                    float(average_precision_score(test_action_labels, test_action_scores))
                    if has_both_action_classes
                    else None
                ),
                "top1_rescue_rate_within_helpful_states": mean(
                    next(
                        zoom.delta_success
                        for zoom in zooms_by_key[key]
                        if zoom.action_id == fold_top_actions[key]
                    )
                    > 0.0
                    for key in helpful_keys
                ),
                "random_rescue_rate_within_helpful_states": mean(
                    mean(zoom.delta_success > 0.0 for zoom in zooms_by_key[key])
                    for key in helpful_keys
                ),
                "policy_result": fold_result,
            }
        )
        fold_models.append(
            {
                "fold": fold_index,
                "selected_state_c": -negative_state_c,
                "selected_action_c": -negative_action_c,
                "threshold": threshold,
                "state_scaler_mean": [
                    float(value) for value in state_scaler.mean_.tolist()
                ],
                "state_scaler_scale": [
                    float(value) for value in state_scaler.scale_.tolist()
                ],
                "state_coefficient": [
                    float(value) for value in state_model.coef_[0].tolist()
                ],
                "state_intercept": float(state_model.intercept_[0]),
                "action_scaler_mean": [
                    float(value) for value in action_scaler.mean_.tolist()
                ],
                "action_scaler_scale": [
                    float(value) for value in action_scaler.scale_.tolist()
                ],
                "action_coefficient": [
                    float(value) for value in action_model.coef_[0].tolist()
                ],
                "action_intercept": float(action_model.intercept_[0]),
            }
        )
    if set(pooled_selected_actions) != set(all_keys):
        missing_oof = sorted(set(all_keys) - set(pooled_selected_actions))
        raise RuntimeError(f"nested OOF predictions are incomplete: {missing_oof[:5]}")
    policy = PrecomputedActionGatePolicy(
        pooled_selected_actions,
        name="nested_oof_two_stage_concrete_crop",
    )
    policy_result: dict[str, Any] = dict(
        evaluate_policy(records, policy, lambda_cost=lambda_cost)
    )
    policy_result["bootstrap"] = bootstrap_policy_evaluation(
        records,
        policy,
        lambda_cost=lambda_cost,
        n_resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    pooled_state_labels = np.asarray(
        [state_labels[key] for key in all_keys],
        dtype=np.int64,
    )
    pooled_state_margin_array = np.asarray(
        [pooled_state_margins[key] for key in all_keys],
        dtype=np.float64,
    )
    helpful_keys = [key for key in all_keys if state_labels[key]]
    report = {
        "scientific_status": (
            "nested grouped OOF two-stage diagnostic; state and action models both "
            "exclude each evaluated image group"
        ),
        "seed": seed,
        "state_feature_mode": state_feature_mode,
        "state_feature_count": len(state_features[all_keys[0]]),
        "action_feature_count": len(next(iter(action_features.values()))),
        "split_group": split_group,
        "n_outer_folds": n_outer_folds,
        "validation_fraction_within_outer_train": validation_fraction,
        "n_decisions": len(all_keys),
        "pooled_state_helpful_roc_auc_of_fold_margin": float(
            roc_auc_score(pooled_state_labels, pooled_state_margin_array)
        ),
        "top1_rescue_rate_within_helpful_states": mean(
            next(
                zoom.delta_success
                for zoom in zooms_by_key[key]
                if zoom.action_id == pooled_top_actions[key]
            )
            > 0.0
            for key in helpful_keys
        ),
        "random_rescue_rate_within_helpful_states": mean(
            mean(zoom.delta_success > 0.0 for zoom in zooms_by_key[key])
            for key in helpful_keys
        ),
        "folds": fold_reports,
        "policy_result": policy_result,
    }
    model_payload = {
        "model_type": "nested_oof_two_stage_state_and_action_gate",
        "seed": seed,
        "state_feature_mode": state_feature_mode,
        "n_outer_folds": n_outer_folds,
        "fold_models": fold_models,
    }
    return report, model_payload


def aggregate_rescue_gate_splits(split_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not split_reports:
        raise ValueError("at least one rescue-gate split report is required")
    utilities = [
        float(report["policy_result"]["mean_policy_utility"]) for report in split_reports
    ]
    gains = [float(report["policy_result"]["accuracy_gain"]) for report in split_reports]
    tool_rates = [float(report["policy_result"]["tool_use_rate"]) for report in split_reports]

    def summary(values: Sequence[float]) -> dict[str, float]:
        return {
            "mean": mean(values),
            "median": median(values),
            "min": min(values),
            "max": max(values),
        }

    return {
        "scientific_status": (
            "repeated grouped-split diagnostic; overlapping test sets are not independent"
        ),
        "seeds": sorted(int(report["seed"]) for report in split_reports),
        "n_splits": len(split_reports),
        "accuracy_gain": summary(gains),
        "tool_use_rate": summary(tool_rates),
        "mean_policy_utility": summary(utilities),
        "positive_utility_splits": sum(value > 0.0 for value in utilities),
        "nonnegative_utility_splits": sum(value >= 0.0 for value in utilities),
    }


def build_rescue_gate_markdown(report: Mapping[str, Any]) -> str:
    aggregate = report["aggregate"]
    lines = [
        "# Compact rescuability-gate diagnostic",
        "",
        "> Expected random-crop outcomes are label-averaged only for seed-free evaluation.",
        "> Repeated image splits overlap and are not an independent confidence interval.",
        "",
        "| Seed | Accuracy gain | Tool rate | Utility [95% state-bootstrap CI] | Helpful ROC-AUC |",
        "|---:|---:|---:|---:|---:|",
    ]
    for split in report["splits"]:
        result = split["policy_result"]
        interval = result["bootstrap"]["metrics"]["mean_policy_utility"]
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} [{:.4f}, {:.4f}] | {:.4f} |".format(
                split["seed"],
                result["accuracy_gain"],
                result["tool_use_rate"],
                result["mean_policy_utility"],
                interval["ci_low"],
                interval["ci_high"],
                split["test_helpful_roc_auc"],
            )
        )
    utility = aggregate["mean_policy_utility"]
    lines.extend(
        [
            "",
            "Mean utility: {:.4f} [{:.4f}, {:.4f}] across split point estimates; positive splits: {}/{}.".format(
                utility["mean"],
                utility["min"],
                utility["max"],
                aggregate["positive_utility_splits"],
                aggregate["n_splits"],
            ),
            "",
        ]
    )
    return "\n".join(lines)
