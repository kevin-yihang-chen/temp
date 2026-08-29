from __future__ import annotations

import math
from statistics import mean
from typing import Any, Mapping, Sequence

from .action_value import predict_frozen_factorized_action_values
from .dataset import group_by_decision
from .rescue_gate import DecisionKey
from .scaled_evaluation import bootstrap_source_balanced_metrics
from .schema import ActionRecord


def _source_means(
    values: Mapping[DecisionKey, float],
    source_by_key: Mapping[DecisionKey, str],
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for key, value in values.items():
        grouped.setdefault(source_by_key[key], []).append(float(value))
    return {source: mean(items) for source, items in grouped.items()}


def _validate_calibrated_model(model: Mapping[str, Any]) -> float:
    if model.get("model_type") != "multidomain_factorized_action_value":
        raise ValueError("formal evaluation requires a factorized action-value model")
    raw_threshold = model.get("threshold")
    calibration = model.get("risk_calibration")
    if not isinstance(raw_threshold, (int, float)) or not isinstance(
        calibration, Mapping
    ):
        raise ValueError("formal evaluation requires a non-degenerate calibrated model")
    threshold = float(raw_threshold)
    if not math.isfinite(threshold):
        raise ValueError("calibrated threshold must be finite")
    if calibration.get("selection_status") != (
        "selected_non_degenerate_safe_threshold"
    ):
        raise ValueError("formal evaluation requires successful risk calibration")
    selected_threshold = calibration.get("selected_threshold")
    if not isinstance(selected_threshold, (int, float)) or float(
        selected_threshold
    ) != threshold:
        raise ValueError("model threshold does not match the calibration report")
    return threshold


def evaluate_factorized_risk_controlled_policy(
    model: Mapping[str, Any],
    records: Sequence[ActionRecord],
    *,
    semantic_decisions: Mapping[DecisionKey, Mapping[str, Any]] | None = None,
    bootstrap_resamples: int = 20000,
    bootstrap_confidence: float = 0.975,
    bootstrap_seed: int = 20260828,
) -> dict[str, Any]:
    """Evaluate the one frozen factorized policy without tuning its gate.

    Source-balanced utility is primary. Question-weighted metrics, crop-ranking
    diagnostics, and always-call/same-gate baselines are reported without
    changing the calibrated threshold or selected crop.
    """

    threshold = _validate_calibrated_model(model)
    lambda_cost = float(model["lambda_cost"])
    if lambda_cost != 0.05:
        raise ValueError("factorized formal evaluation requires lambda=0.05")
    actions, scores = predict_frozen_factorized_action_values(
        model,
        records,
        semantic_decisions=semantic_decisions,
    )
    grouped = group_by_decision(records)
    if set(actions) != set(scores) or set(actions) != set(grouped):
        raise ValueError("factorized predictions do not cover every formal decision")

    metric_names = (
        "utility",
        "gain",
        "call",
        "baseline_accuracy",
        "policy_accuracy",
        "induced_harm",
        "net_negative_call",
        "negative_net_value",
        "oracle_utility",
        "random_always_call_utility",
        "random_same_gate_utility",
        "post_action_entropy_always_call_utility",
        "post_action_entropy_same_gate_utility",
        "matched_budget_entropy_gate_learned_crop_utility",
        "matched_budget_entropy_gate_random_crop_utility",
        "matched_budget_random_gate_random_crop_expected_utility",
    )
    values: dict[str, dict[DecisionKey, float]] = {
        name: {} for name in metric_names
    }
    fixed_always_call: dict[str, dict[DecisionKey, float]] = {}
    fixed_same_gate: dict[str, dict[DecisionKey, float]] = {}
    fixed_entropy_gate: dict[str, dict[DecisionKey, float]] = {}
    decision_diagnostics: dict[DecisionKey, dict[str, Any]] = {}
    source_by_key: dict[DecisionKey, str] = {}
    helpful_states = 0
    selected_rescues = 0.0
    random_rescue_total = 0.0
    calls = 0
    positive_utility_calls = 0
    unnecessary_calls = 0
    stoppable_states = 0
    correct_stops = 0

    for key, siblings in grouped.items():
        answers = [record for record in siblings if record.action_type == "ANSWER"]
        zooms = sorted(
            (record for record in siblings if record.action_type == "ZOOM"),
            key=lambda record: record.action_id,
        )
        if len(answers) != 1 or len(zooms) != 4:
            raise ValueError(f"invalid factorized formal decision {key!r}")
        baseline = answers[0]
        matches = [zoom for zoom in zooms if zoom.action_id == actions[key]]
        if len(matches) != 1:
            raise ValueError(f"factorized formal action mismatch for {key!r}")
        selected = matches[0]
        called = scores[key] >= threshold
        gain = selected.delta_success if called else 0.0
        utility = gain - lambda_cost * selected.tool_cost if called else 0.0
        source_by_key[key] = baseline.source_id
        values["utility"][key] = utility
        values["gain"][key] = gain
        values["call"][key] = float(called)
        values["baseline_accuracy"][key] = baseline.correct_before
        values["policy_accuracy"][key] = (
            selected.correct_after if called else baseline.correct_before
        )
        values["induced_harm"][key] = max(-gain, 0.0)
        values["net_negative_call"][key] = float(called and utility < 0.0)
        values["negative_net_value"][key] = max(-utility, 0.0)

        net_values = [zoom.voi(lambda_cost) for zoom in zooms]
        values["oracle_utility"][key] = max(0.0, max(net_values))
        random_utility = mean(net_values)
        values["random_always_call_utility"][key] = random_utility
        values["random_same_gate_utility"][key] = (
            random_utility if called else 0.0
        )
        entropy_action = min(
            zooms,
            key=lambda zoom: (zoom.entropy_after, zoom.action_id),
        )
        entropy_utility = entropy_action.voi(lambda_cost)
        values["post_action_entropy_always_call_utility"][key] = entropy_utility
        values["post_action_entropy_same_gate_utility"][key] = (
            entropy_utility if called else 0.0
        )
        for zoom in zooms:
            action_utility = zoom.voi(lambda_cost)
            fixed_always_call.setdefault(zoom.action_id, {})[key] = action_utility
            fixed_same_gate.setdefault(zoom.action_id, {})[key] = (
                action_utility if called else 0.0
            )
        decision_diagnostics[key] = {
            "entropy_before": baseline.entropy_before,
            "learned_crop_utility": selected.voi(lambda_cost),
            "random_crop_utility": random_utility,
            "fixed_crop_utilities": {
                zoom.action_id: zoom.voi(lambda_cost) for zoom in zooms
            },
        }

        helpful = any(zoom.delta_success > 0.0 for zoom in zooms)
        if helpful:
            helpful_states += 1
            selected_rescues += float(selected.delta_success > 0.0)
            random_rescue_total += mean(
                zoom.delta_success > 0.0 for zoom in zooms
            )
        no_positive_net_action = max(net_values) <= 0.0
        if no_positive_net_action:
            stoppable_states += 1
            correct_stops += int(not called)
        if called:
            calls += 1
            positive_utility_calls += int(utility > 0.0)
            unnecessary_calls += int(utility <= 0.0)

    entropy_order = sorted(
        decision_diagnostics,
        key=lambda key: (
            float(decision_diagnostics[key]["entropy_before"]),
            key,
        ),
        reverse=True,
    )
    entropy_gate_keys = set(entropy_order[:calls])
    random_gate_probability = calls / len(grouped) if grouped else 0.0
    for key, diagnostic in decision_diagnostics.items():
        entropy_called = key in entropy_gate_keys
        learned_crop_utility = float(diagnostic["learned_crop_utility"])
        random_crop_utility = float(diagnostic["random_crop_utility"])
        values["matched_budget_entropy_gate_learned_crop_utility"][key] = (
            learned_crop_utility if entropy_called else 0.0
        )
        values["matched_budget_entropy_gate_random_crop_utility"][key] = (
            random_crop_utility if entropy_called else 0.0
        )
        values["matched_budget_random_gate_random_crop_expected_utility"][key] = (
            random_gate_probability * random_crop_utility
        )
        raw_fixed = diagnostic["fixed_crop_utilities"]
        if not isinstance(raw_fixed, Mapping):
            raise RuntimeError("fixed-crop diagnostic is invalid")
        for action_id, action_utility in raw_fixed.items():
            fixed_entropy_gate.setdefault(str(action_id), {})[key] = (
                float(action_utility) if entropy_called else 0.0
            )

    source_metric_values = {
        name: _source_means(metric_values, source_by_key)
        for name, metric_values in values.items()
    }
    sources = sorted(source_metric_values["utility"])
    source_metrics = {
        source: {
            name: source_metric_values[name][source] for name in metric_names
        }
        for source in sources
    }
    bootstrap = bootstrap_source_balanced_metrics(
        source_metrics,
        n_resamples=bootstrap_resamples,
        confidence_level=bootstrap_confidence,
        seed=bootstrap_seed,
    )
    source_point = {
        name: mean(source_metric_values[name].values()) for name in metric_names
    }
    question_point = {
        name: mean(metric_values.values()) for name, metric_values in values.items()
    }
    fixed_source_always = {
        action_id: mean(_source_means(metric, source_by_key).values())
        for action_id, metric in sorted(fixed_always_call.items())
    }
    fixed_source_same_gate = {
        action_id: mean(_source_means(metric, source_by_key).values())
        for action_id, metric in sorted(fixed_same_gate.items())
    }
    fixed_source_entropy_gate = {
        action_id: mean(_source_means(metric, source_by_key).values())
        for action_id, metric in sorted(fixed_entropy_gate.items())
    }
    fixed_question_entropy_gate = {
        action_id: mean(metric.values())
        for action_id, metric in sorted(fixed_entropy_gate.items())
    }
    primary_interval = bootstrap["metrics"]["utility"]
    pass_rule = {
        "source_utility_positive": source_point["utility"] > 0.0,
        "source_utility_97_5pct_ci_low_positive": float(
            primary_interval["ci_low"]
        )
        > 0.0,
        "question_weighted_utility_positive": question_point["utility"] > 0.0,
        "source_call_rate_at_least_0_01": source_point["call"] >= 0.01,
    }
    return {
        "scientific_status": (
            "one-shot evaluation of the frozen factorized fixed-sequence policy"
        ),
        "passed": all(pass_rule.values()),
        "threshold": threshold,
        "lambda_cost": lambda_cost,
        "n_sources": len(sources),
        "n_decisions": len(grouped),
        "source_balanced": source_point,
        "question_weighted": question_point,
        "source_bootstrap": bootstrap,
        "risk_diagnostics": {
            "source_balanced_induced_harm_mass": source_point["induced_harm"],
            "source_balanced_net_negative_call_mass": source_point[
                "net_negative_call"
            ],
            "source_balanced_negative_net_value": source_point[
                "negative_net_value"
            ],
        },
        "ranking": {
            "helpful_states": helpful_states,
            "top1_rescue_rate_within_helpful_states": (
                selected_rescues / helpful_states if helpful_states else 0.0
            ),
            "random_rescue_rate_within_helpful_states": (
                random_rescue_total / helpful_states if helpful_states else 0.0
            ),
        },
        "baselines": {
            "post_action_entropy_is_diagnostic_not_deployable": True,
            "matched_budget_gate_uses_outcomes": False,
            "matched_budget_call_count": calls,
            "matched_budget_entropy_tie_break": (
                "descending entropy_before then descending state_id replicate_id"
            ),
            "matched_budget_question_call_rate": (
                calls / len(grouped) if grouped else 0.0
            ),
            "matched_budget_entropy_threshold": (
                float(
                    decision_diagnostics[entropy_order[calls - 1]][
                        "entropy_before"
                    ]
                )
                if calls
                else None
            ),
            "matched_budget_entropy_gate_source_utility_learned_crop": (
                source_point[
                    "matched_budget_entropy_gate_learned_crop_utility"
                ]
            ),
            "matched_budget_entropy_gate_source_utility_random_crop": (
                source_point[
                    "matched_budget_entropy_gate_random_crop_utility"
                ]
            ),
            "matched_budget_random_gate_source_utility_random_crop_expected": (
                source_point[
                    "matched_budget_random_gate_random_crop_expected_utility"
                ]
            ),
            "matched_budget_entropy_gate_question_utility_learned_crop": (
                question_point[
                    "matched_budget_entropy_gate_learned_crop_utility"
                ]
            ),
            "matched_budget_entropy_gate_question_utility_random_crop": (
                question_point[
                    "matched_budget_entropy_gate_random_crop_utility"
                ]
            ),
            "fixed_crop_source_utility_entropy_gate": fixed_source_entropy_gate,
            "fixed_crop_question_utility_entropy_gate": fixed_question_entropy_gate,
            "fixed_crop_source_utility_always_call": fixed_source_always,
            "fixed_crop_source_utility_same_gate": fixed_source_same_gate,
        },
        "selection": {
            "calls": calls,
            "source_balanced_raw_gain_per_call": (
                source_point["gain"] / source_point["call"]
                if source_point["call"] > 0.0
                else 0.0
            ),
            "source_balanced_utility_per_call": (
                source_point["utility"] / source_point["call"]
                if source_point["call"] > 0.0
                else 0.0
            ),
            "question_weighted_raw_gain_per_call": (
                sum(values["gain"].values()) / calls if calls else 0.0
            ),
            "question_weighted_utility_per_call": (
                sum(values["utility"].values()) / calls if calls else 0.0
            ),
            "positive_utility_call_precision": (
                positive_utility_calls / calls if calls else 0.0
            ),
            "unnecessary_call_rate": unnecessary_calls / calls if calls else 0.0,
            "stoppable_states": stoppable_states,
            "correct_stopping_rate": (
                correct_stops / stoppable_states if stoppable_states else 0.0
            ),
        },
        "oracle_regret": source_point["oracle_utility"]
        - source_point["utility"],
        "pass_rule": pass_rule,
    }
