from __future__ import annotations

import random
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


def compact_rescue_features(decision: Mapping[str, Any]) -> list[float]:
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
    return [float(value) for value in features.tolist()]


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
    features = {
        name: np.asarray(
            [compact_rescue_features(decision_by_key[key]) for key in keys],
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
    raw_features = {
        key: compact_rescue_features(decision_by_key[key])
        for key in outer_keys + test_keys
    }
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
    features = {
        name: np.asarray(
            [compact_rescue_features(decision_by_key[key]) for key in keys],
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
        "seed": seed,
        "selected_alpha": -negative_alpha,
        "threshold": threshold,
        "scaler_mean": [float(value) for value in scaler.mean_.tolist()],
        "scaler_scale": [float(value) for value in scaler.scale_.tolist()],
        "coefficient": [float(value) for value in model.coef_.tolist()],
        "intercept": float(model.intercept_),
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
