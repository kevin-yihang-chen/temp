from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

from .action_value import (
    _action_features,
    _calibrate_domain_robust_threshold,
    _ranking_diagnostics,
    _score_factorized_candidates,
    _semantic_feature_index,
    _state_features,
    _validate_domains,
)
from .metrics import bootstrap_policy_evaluation, evaluate_policy
from .rescue_gate import DecisionKey, PrecomputedActionGatePolicy
from .risk_control import (
    AcquisitionCalibrationRow,
    RiskConstraint,
    calibrate_source_risk_threshold,
)
from .schema import ActionRecord


def _domain_source_balanced_weights(
    domains: Sequence[str],
    sources: Sequence[str],
) -> list[float]:
    """Give every domain, then every source within a domain, equal mass.

    Rows from a source share that source's mass.  This prevents sources with
    several questions or action candidates from dominating the OOF heads while
    retaining equal total weight for every development domain.
    """

    if not domains or len(domains) != len(sources):
        raise ValueError("domains and sources must be non-empty and aligned")
    rows_per_source: dict[tuple[str, str], int] = {}
    sources_per_domain: dict[str, set[str]] = {}
    for domain, source in zip(domains, sources):
        pair = (domain, source)
        rows_per_source[pair] = rows_per_source.get(pair, 0) + 1
        sources_per_domain.setdefault(domain, set()).add(source)
    n_domains = len(sources_per_domain)
    raw = [
        1.0
        / (
            n_domains
            * len(sources_per_domain[domain])
            * rows_per_source[(domain, source)]
        )
        for domain, source in zip(domains, sources)
    ]
    scale = len(raw) / sum(raw)
    return [value * scale for value in raw]


def _source_folds(
    domain_by_key: Mapping[DecisionKey, str],
    baselines: Mapping[DecisionKey, ActionRecord],
    *,
    n_folds: int,
    seed: int,
) -> tuple[dict[DecisionKey, int], dict[str, list[int]]]:
    if n_folds < 2:
        raise ValueError("OOF action value requires at least two folds")
    sources_by_domain: dict[str, set[str]] = {}
    for key, domain in domain_by_key.items():
        sources_by_domain.setdefault(domain, set()).add(baselines[key].source_id)
    source_fold: dict[tuple[str, str], int] = {}
    fold_source_counts: dict[str, list[int]] = {}
    for domain, sources in sorted(sources_by_domain.items()):
        if len(sources) < n_folds:
            raise ValueError(
                f"domain {domain!r} needs at least {n_folds} source groups"
            )
        ordered = sorted(
            sources,
            key=lambda source: (
                hashlib.sha256(
                    f"action-value-oof-v1\0{seed}\0{domain}\0{source}".encode()
                ).digest(),
                source,
            ),
        )
        counts = [0] * n_folds
        for index, source in enumerate(ordered):
            fold = index % n_folds
            source_fold[(domain, source)] = fold
            counts[fold] += 1
        fold_source_counts[domain] = counts
    fold_by_key = {
        key: source_fold[(domain, baselines[key].source_id)]
        for key, domain in domain_by_key.items()
    }
    return fold_by_key, fold_source_counts


def _fit_heads(
    keys: Sequence[DecisionKey],
    *,
    alpha: float,
    seed: int,
    feature_mode: str,
    baselines: Mapping[DecisionKey, ActionRecord],
    zooms: Mapping[DecisionKey, Sequence[ActionRecord]],
    domain_by_key: Mapping[DecisionKey, str],
    semantic_by_key: Mapping[DecisionKey, Mapping[str, Any]],
) -> dict[str, Any]:
    import numpy as np  # type: ignore[import-not-found]
    from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

    state_features = [
        _state_features(
            baselines[key],
            feature_mode=feature_mode,
            semantic_decision=semantic_by_key.get(key),
        )
        for key in keys
    ]
    error_labels = [int(baselines[key].correct_before < 0.5) for key in keys]
    rescue_rows = [
        (key, action)
        for key in keys
        if baselines[key].correct_before < 0.5
        for action in zooms[key]
    ]
    harm_rows = [
        (key, action)
        for key in keys
        if baselines[key].correct_before >= 0.5
        for action in zooms[key]
    ]
    if not rescue_rows or not harm_rows:
        raise ValueError("factorized action value requires wrong and correct baselines")
    rescue_features = [
        _action_features(
            baselines[key],
            action,
            feature_mode=feature_mode,
            semantic_decision=semantic_by_key.get(key),
        )
        for key, action in rescue_rows
    ]
    harm_features = [
        _action_features(
            baselines[key],
            action,
            feature_mode=feature_mode,
            semantic_decision=semantic_by_key.get(key),
        )
        for key, action in harm_rows
    ]
    rescue_labels = [int(action.delta_success > 0.0) for _, action in rescue_rows]
    harm_labels = [int(action.delta_success < 0.0) for _, action in harm_rows]
    for name, labels in (
        ("error", error_labels),
        ("rescue", rescue_labels),
        ("harm", harm_labels),
    ):
        if len(set(labels)) != 2:
            raise ValueError(f"factorized {name} head requires both outcome classes")

    state_array = np.asarray(state_features, dtype=np.float64)
    rescue_array = np.asarray(rescue_features, dtype=np.float64)
    harm_array = np.asarray(harm_features, dtype=np.float64)
    error_scaler = StandardScaler().fit(state_array)
    rescue_scaler = StandardScaler().fit(rescue_array)
    harm_scaler = StandardScaler().fit(harm_array)
    model_kwargs = {
        "C": 1.0 / float(alpha),
        "solver": "liblinear",
        "max_iter": 2000,
        "random_state": seed,
    }
    error_domains = [domain_by_key[key] for key in keys]
    rescue_domains = [domain_by_key[key] for key, _ in rescue_rows]
    harm_domains = [domain_by_key[key] for key, _ in harm_rows]
    error_sources = [baselines[key].source_id for key in keys]
    rescue_sources = [baselines[key].source_id for key, _ in rescue_rows]
    harm_sources = [baselines[key].source_id for key, _ in harm_rows]
    error_weights = np.asarray(
        _domain_source_balanced_weights(error_domains, error_sources),
        dtype=np.float64,
    )
    rescue_weights = np.asarray(
        _domain_source_balanced_weights(rescue_domains, rescue_sources),
        dtype=np.float64,
    )
    harm_weights = np.asarray(
        _domain_source_balanced_weights(harm_domains, harm_sources),
        dtype=np.float64,
    )
    error_model = LogisticRegression(**model_kwargs).fit(
        error_scaler.transform(state_array),
        np.asarray(error_labels, dtype=np.int64),
        sample_weight=error_weights,
    )
    rescue_model = LogisticRegression(**model_kwargs).fit(
        rescue_scaler.transform(rescue_array),
        np.asarray(rescue_labels, dtype=np.int64),
        sample_weight=rescue_weights,
    )
    harm_model = LogisticRegression(**model_kwargs).fit(
        harm_scaler.transform(harm_array),
        np.asarray(harm_labels, dtype=np.int64),
        sample_weight=harm_weights,
    )
    positive_rescues = [
        index for index, label in enumerate(rescue_labels) if label == 1
    ]
    positive_harms = [index for index, label in enumerate(harm_labels) if label == 1]
    rescue_magnitude = float(
        np.average(
            [rescue_rows[index][1].delta_success for index in positive_rescues],
            weights=rescue_weights[positive_rescues],
        )
    )
    harm_magnitude = float(
        np.average(
            [-harm_rows[index][1].delta_success for index in positive_harms],
            weights=harm_weights[positive_harms],
        )
    )
    return {
        "error_scaler": error_scaler,
        "rescue_scaler": rescue_scaler,
        "harm_scaler": harm_scaler,
        "error_model": error_model,
        "rescue_model": rescue_model,
        "harm_model": harm_model,
        "rescue_magnitude": rescue_magnitude,
        "harm_magnitude": harm_magnitude,
        "state_feature_count": len(state_features[0]),
        "action_feature_count": len(rescue_features[0]),
        "error_train_rows": len(keys),
        "rescue_train_rows": len(rescue_rows),
        "harm_train_rows": len(harm_rows),
    }


def _score_heads(
    heads: Mapping[str, Any],
    keys: Sequence[DecisionKey],
    *,
    lambda_cost: float,
    feature_mode: str,
    baselines: Mapping[DecisionKey, ActionRecord],
    zooms: Mapping[DecisionKey, Sequence[ActionRecord]],
    semantic_by_key: Mapping[DecisionKey, Mapping[str, Any]],
) -> tuple[dict[DecisionKey, float], dict[DecisionKey, str]]:
    return _score_factorized_candidates(
        error_model=heads["error_model"],
        rescue_model=heads["rescue_model"],
        harm_model=heads["harm_model"],
        error_scaler=heads["error_scaler"],
        rescue_scaler=heads["rescue_scaler"],
        harm_scaler=heads["harm_scaler"],
        rescue_magnitude=float(heads["rescue_magnitude"]),
        harm_magnitude=float(heads["harm_magnitude"]),
        keys=keys,
        baselines=baselines,
        zooms=zooms,
        lambda_cost=lambda_cost,
        feature_mode=feature_mode,
        semantic_by_key=semantic_by_key,
    )


def _serialized_head(prefix: str, scaler: Any, model: Any) -> dict[str, Any]:
    return {
        f"{prefix}_scaler_mean": [float(value) for value in scaler.mean_.tolist()],
        f"{prefix}_scaler_scale": [float(value) for value in scaler.scale_.tolist()],
        f"{prefix}_coefficient": [
            float(value) for value in model.coef_[0].tolist()
        ],
        f"{prefix}_intercept": float(model.intercept_[0]),
    }


def _development_tail_risk_diagnostic(
    *,
    values: Mapping[DecisionKey, float],
    actions: Mapping[DecisionKey, str],
    baselines: Mapping[DecisionKey, ActionRecord],
    zooms: Mapping[DecisionKey, Sequence[ActionRecord]],
    lambda_cost: float,
    target_call_rates: Sequence[float],
) -> dict[str, Any]:
    """Summarize high-score OOF tails without claiming fresh calibration validity."""

    rates = [float(value) for value in target_call_rates]
    if not rates or any(not 0.0 < value <= 1.0 for value in rates):
        raise ValueError("tail call rates must lie in (0,1]")
    if rates != sorted(set(rates)):
        raise ValueError("tail call rates must be sorted and unique")
    ordered_scores = sorted((float(value) for value in values.values()), reverse=True)
    requested_thresholds = [
        {
            "target_pooled_call_rate": rate,
            "threshold": ordered_scores[
                min(
                    len(ordered_scores) - 1,
                    max(0, math.ceil(rate * len(ordered_scores)) - 1),
                )
            ],
        }
        for rate in rates
    ]
    thresholds = list(
        dict.fromkeys(float(item["threshold"]) for item in requested_thresholds)
    )
    rows: list[AcquisitionCalibrationRow] = []
    for key in sorted(values):
        action_id = actions[key]
        matches = [action for action in zooms[key] if action.action_id == action_id]
        if len(matches) != 1:
            raise RuntimeError("OOF tail action does not uniquely match a candidate")
        action = matches[0]
        rows.append(
            AcquisitionCalibrationRow(
                source_id=baselines[key].source_id,
                score=float(values[key]),
                gain=action.delta_success,
                tool_cost=action.tool_cost,
            )
        )
    diagnostic = calibrate_source_risk_threshold(
        rows,
        thresholds,
        constraints=[
            RiskConstraint("induced_harm", 0.005),
            RiskConstraint("net_negative_call_mass", 0.02),
        ],
        lambda_cost=lambda_cost,
        max_tool_cost=1.0,
        family_error=0.05,
        min_source_call_rate=0.01,
        min_source_utility=0.001,
        selection_objective="source_utility",
    )
    diagnostic["scientific_status"] = (
        "development-only source-OOF tail diagnostic; hyperparameters and tails "
        "use development outcomes and cannot replace fresh risk calibration"
    )
    diagnostic["valid_for_formal_selection"] = False
    diagnostic["requested_thresholds"] = requested_thresholds
    return diagnostic


def fit_oof_factorized_action_value_model(
    records_by_domain: Mapping[str, Sequence[ActionRecord]],
    *,
    feature_mode: str = "context-geometry",
    semantic_decisions_by_domain: Mapping[
        str, Mapping[DecisionKey, Mapping[str, Any]]
    ]
    | None = None,
    n_folds: int = 5,
    lambda_cost: float = 0.05,
    alpha_values: Sequence[float] = (0.1, 1.0, 10.0, 100.0, 1000.0),
    seed: int = 20260828,
    bootstrap_resamples: int = 2000,
    tail_call_rates: Sequence[float] = (0.005, 0.01, 0.015, 0.02, 0.03, 0.05),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select a factorized model from source-grouped out-of-fold predictions.

    Every development decision is scored by heads trained without its source.
    Hyperparameters and the no-call margin are selected on the pooled OOF action
    bank, after which the chosen heads are refit once on all development sources.
    Formal outcomes remain untouched.
    """

    if lambda_cost < 0.0:
        raise ValueError("lambda_cost must be non-negative")
    if not alpha_values or any(alpha <= 0.0 for alpha in alpha_values):
        raise ValueError("alpha_values must be positive")
    domain_by_key, baselines, zooms = _validate_domains(records_by_domain)
    semantic_by_key = _semantic_feature_index(
        feature_mode=feature_mode,
        records_by_domain=records_by_domain,
        domain_by_key=domain_by_key,
        semantic_decisions_by_domain=semantic_decisions_by_domain,
    )
    fold_by_key, fold_source_counts = _source_folds(
        domain_by_key,
        baselines,
        n_folds=n_folds,
        seed=seed,
    )
    all_keys = sorted(baselines)
    candidate_reports: list[dict[str, Any]] = []
    for alpha in alpha_values:
        oof_values: dict[DecisionKey, float] = {}
        oof_actions: dict[DecisionKey, str] = {}
        fold_train_counts: list[dict[str, int]] = []
        for fold in range(n_folds):
            train_keys = [key for key in all_keys if fold_by_key[key] != fold]
            test_keys = [key for key in all_keys if fold_by_key[key] == fold]
            heads = _fit_heads(
                train_keys,
                alpha=float(alpha),
                seed=seed + fold,
                feature_mode=feature_mode,
                baselines=baselines,
                zooms=zooms,
                domain_by_key=domain_by_key,
                semantic_by_key=semantic_by_key,
            )
            values, actions = _score_heads(
                heads,
                test_keys,
                lambda_cost=lambda_cost,
                feature_mode=feature_mode,
                baselines=baselines,
                zooms=zooms,
                semantic_by_key=semantic_by_key,
            )
            oof_values.update(values)
            oof_actions.update(actions)
            fold_train_counts.append(
                {
                    "fold": fold,
                    "train_decisions": len(train_keys),
                    "test_decisions": len(test_keys),
                }
            )
        if set(oof_values) != set(all_keys) or set(oof_actions) != set(all_keys):
            raise RuntimeError("OOF predictions do not exactly cover development keys")
        threshold, selected, metrics = _calibrate_domain_robust_threshold(
            oof_actions,
            oof_values,
            all_keys,
            zooms,
            domain_by_key,
            lambda_cost=lambda_cost,
        )
        candidate_reports.append(
            {
                "alpha": float(alpha),
                "threshold": threshold,
                "selected": selected,
                "metrics": metrics,
                "ranking_diagnostics": _ranking_diagnostics(
                    oof_actions,
                    all_keys,
                    zooms,
                    domain_by_key,
                    lambda_cost=lambda_cost,
                ),
                "fold_counts": fold_train_counts,
                "oof_values": oof_values,
                "oof_actions": oof_actions,
            }
        )
    winner = max(
        candidate_reports,
        key=lambda candidate: (
            candidate["metrics"]["worst_domain_utility"],
            candidate["metrics"]["domain_balanced_mean_utility"],
            candidate["metrics"]["pooled_mean_utility"],
            -candidate["metrics"]["pooled_tool_rate"],
            -candidate["alpha"],
        ),
    )
    full_heads = _fit_heads(
        all_keys,
        alpha=float(winner["alpha"]),
        seed=seed,
        feature_mode=feature_mode,
        baselines=baselines,
        zooms=zooms,
        domain_by_key=domain_by_key,
        semantic_by_key=semantic_by_key,
    )
    all_records = [record for records in records_by_domain.values() for record in records]
    oof_policy = PrecomputedActionGatePolicy(
        winner["selected"], name="oof_factorized_action_value"
    )
    policy_result = dict(
        evaluate_policy(all_records, oof_policy, lambda_cost=lambda_cost)
    )
    bootstrap = bootstrap_policy_evaluation(
        all_records,
        oof_policy,
        lambda_cost=lambda_cost,
        n_resamples=bootstrap_resamples,
        seed=seed,
        cluster_by="source_id",
    )
    report = {
        "scientific_status": (
            "development-only source-grouped OOF selection and full-development "
            "refit; formal benchmark outcomes are excluded"
        ),
        "model_type": "multidomain_factorized_action_value",
        "training_protocol": "source_grouped_oof_domain_source_balanced_v2",
        "sample_weighting": "equal_domain_then_equal_source_then_equal_row",
        "feature_mode": feature_mode,
        "decision_rule": (
            "P(error)*P(rescue|error,action)*rescue_magnitude - "
            "P(correct)*P(harm|correct,action)*harm_magnitude - cost; call above "
            "source-grouped OOF margin"
        ),
        "seed": seed,
        "n_folds": n_folds,
        "fold_source_counts": fold_source_counts,
        "lambda_cost": lambda_cost,
        "domains": sorted(set(domain_by_key.values())),
        "development_decisions": len(all_keys),
        "selected_alpha": winner["alpha"],
        "selected_threshold": winner["threshold"],
        "selection_objective": (
            "worst-domain OOF utility, then domain-balanced mean utility, pooled "
            "utility, lower tool rate, lower alpha; no-call is explicit"
        ),
        "candidate_oof_metrics": [
            {
                "alpha": candidate["alpha"],
                "threshold": candidate["threshold"],
                **candidate["metrics"],
                "ranking_diagnostics": candidate["ranking_diagnostics"],
                "fold_counts": candidate["fold_counts"],
            }
            for candidate in candidate_reports
        ],
        "oof_metrics": winner["metrics"],
        "oof_policy_result": policy_result,
        "oof_bootstrap": bootstrap,
        "development_tail_risk_diagnostic": _development_tail_risk_diagnostic(
            values=winner["oof_values"],
            actions=winner["oof_actions"],
            baselines=baselines,
            zooms=zooms,
            lambda_cost=lambda_cost,
            target_call_rates=tail_call_rates,
        ),
        "refit": {
            key: full_heads[key]
            for key in (
                "state_feature_count",
                "action_feature_count",
                "error_train_rows",
                "rescue_train_rows",
                "harm_train_rows",
                "rescue_magnitude",
                "harm_magnitude",
            )
        },
    }
    model_payload = {
        "model_type": "multidomain_factorized_action_value",
        "training_protocol": "source_grouped_oof_domain_source_balanced_v2",
        "sample_weighting": "equal_domain_then_equal_source_then_equal_row",
        "feature_mode": feature_mode,
        "decision_rule": "factorized_expected_net_value_above_frozen_oof_margin",
        "seed": seed,
        "n_folds": n_folds,
        "lambda_cost": lambda_cost,
        "selected_alpha": winner["alpha"],
        "threshold": winner["threshold"],
        "domains": sorted(set(domain_by_key.values())),
        "state_feature_count": full_heads["state_feature_count"],
        "action_feature_count": full_heads["action_feature_count"],
        "rescue_magnitude": full_heads["rescue_magnitude"],
        "harm_magnitude": full_heads["harm_magnitude"],
        **_serialized_head(
            "error", full_heads["error_scaler"], full_heads["error_model"]
        ),
        **_serialized_head(
            "rescue", full_heads["rescue_scaler"], full_heads["rescue_model"]
        ),
        **_serialized_head(
            "harm", full_heads["harm_scaler"], full_heads["harm_model"]
        ),
    }
    return report, model_payload
