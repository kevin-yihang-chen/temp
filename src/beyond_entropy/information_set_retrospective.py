from __future__ import annotations

import math
from statistics import mean
from typing import Any, Mapping, Sequence

from .dataset import group_by_decision
from .rescue_gate import DecisionKey
from .scaled_evaluation import bootstrap_source_balanced_metrics
from .schema import ActionRecord


def _top_k_keys(
    scores: Mapping[DecisionKey, float],
    *,
    count: int,
) -> tuple[DecisionKey, ...]:
    """Select an outcome-blind exact top-k set with a frozen key tie break."""

    if not scores or not 0 < count < len(scores):
        raise ValueError(
            "top-k count must be strictly between zero and population size"
        )
    normalized = {key: float(value) for key, value in scores.items()}
    if any(not math.isfinite(value) for value in normalized.values()):
        raise ValueError("top-k scores must be finite")
    ordered = sorted(
        normalized,
        key=lambda key: (-normalized[key], key[0], key[1]),
    )
    return tuple(ordered[:count])


def _source_means(
    values: Mapping[DecisionKey, float],
    source_by_key: Mapping[DecisionKey, str],
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for key, value in values.items():
        grouped.setdefault(source_by_key[key], []).append(float(value))
    return {source: mean(items) for source, items in grouped.items()}


def _cost_metric_suffix(value: float) -> str:
    rendered = f"{value:.12g}".replace("-", "m").replace(".", "p")
    return rendered


def _validate_prediction(
    name: str,
    *,
    keys: set[DecisionKey],
    actions: Mapping[DecisionKey, str],
    scores: Mapping[DecisionKey, float],
    threshold: float,
) -> None:
    if set(actions) != keys or set(scores) != keys:
        raise ValueError(f"{name} predictions do not exactly cover the decision bank")
    if not math.isfinite(float(threshold)):
        raise ValueError(f"{name} threshold must be finite")
    if any(not math.isfinite(float(value)) for value in scores.values()):
        raise ValueError(f"{name} scores must be finite")


def evaluate_information_set_retrospective(
    records: Sequence[ActionRecord],
    *,
    lower_actions: Mapping[DecisionKey, str],
    lower_scores: Mapping[DecisionKey, float],
    lower_threshold: float,
    higher_actions: Mapping[DecisionKey, str],
    higher_scores: Mapping[DecisionKey, float],
    higher_threshold: float,
    matched_call_rate: float = 0.05,
    lambda_cost: float = 0.05,
    higher_information_extra_costs: Sequence[float] = (0.0,),
    bootstrap_resamples: int = 20000,
    bootstrap_confidence: float = 0.975,
    bootstrap_seed: int = 20260903,
) -> dict[str, Any]:
    """Compare two frozen nested-information policies on one sibling bank.

    The matched call sets depend only on frozen policy scores.  Action outcomes
    enter only after every call set and selected action have been fixed.  The
    higher-information acquisition cost is expressed directly in utility units
    per decision and is therefore subtracted whether or not a crop is called.
    """

    if not 0.0 < matched_call_rate < 1.0:
        raise ValueError("matched call rate must lie strictly between zero and one")
    if not math.isfinite(lambda_cost) or lambda_cost < 0.0:
        raise ValueError("lambda cost must be finite and non-negative")
    extra_costs = tuple(float(value) for value in higher_information_extra_costs)
    if (
        not extra_costs
        or any(not math.isfinite(value) or value < 0.0 for value in extra_costs)
        or tuple(sorted(set(extra_costs))) != extra_costs
        or extra_costs[0] != 0.0
    ):
        raise ValueError(
            "higher-information extra costs must be unique, sorted, finite, "
            "non-negative, and start at zero"
        )

    grouped = group_by_decision(records)
    keys = set(grouped)
    if len(keys) < 2:
        raise ValueError("information-set audit requires at least two decisions")
    _validate_prediction(
        "lower-information",
        keys=keys,
        actions=lower_actions,
        scores=lower_scores,
        threshold=lower_threshold,
    )
    _validate_prediction(
        "higher-information",
        keys=keys,
        actions=higher_actions,
        scores=higher_scores,
        threshold=higher_threshold,
    )

    target_calls = round(len(keys) * matched_call_rate)
    target_calls = min(max(1, target_calls), len(keys) - 1)
    lower_matched = set(_top_k_keys(lower_scores, count=target_calls))
    higher_matched = set(_top_k_keys(higher_scores, count=target_calls))
    entropy_scores: dict[DecisionKey, float] = {}
    oracle_scores: dict[DecisionKey, float] = {}
    source_by_key: dict[DecisionKey, str] = {}
    action_ids: tuple[str, ...] | None = None
    decision_rows: dict[DecisionKey, dict[str, float]] = {}

    for key in sorted(keys):
        siblings = grouped[key]
        answers = [record for record in siblings if record.action_type == "ANSWER"]
        zooms = sorted(
            (record for record in siblings if record.action_type == "ZOOM"),
            key=lambda record: record.action_id,
        )
        if len(answers) != 1 or len(zooms) != 4:
            raise ValueError(f"decision {key!r} must contain ANSWER plus four ZOOMs")
        current_action_ids = tuple(record.action_id for record in zooms)
        if len(set(current_action_ids)) != 4:
            raise ValueError(f"decision {key!r} contains duplicate action IDs")
        if action_ids is None:
            action_ids = current_action_ids
        elif action_ids != current_action_ids:
            raise ValueError("the registered four-action bank changes across decisions")
        zoom_by_id = {record.action_id: record for record in zooms}
        if (
            lower_actions[key] not in zoom_by_id
            or higher_actions[key] not in zoom_by_id
        ):
            raise ValueError(f"policy selected an unregistered action for {key!r}")

        answer = answers[0]
        source_by_key[key] = answer.source_id
        entropy_scores[key] = float(answer.entropy_before)
        net_by_action = {record.action_id: record.voi(lambda_cost) for record in zooms}
        oracle_scores[key] = max(net_by_action.values())
        lower_utility = net_by_action[lower_actions[key]]
        higher_utility = net_by_action[higher_actions[key]]
        random_crop_utility = mean(net_by_action.values())
        post_action_entropy = min(
            zooms,
            key=lambda record: (record.entropy_after, record.action_id),
        )
        post_action_entropy_utility = post_action_entropy.voi(lambda_cost)
        exhaustive_utility = post_action_entropy.delta_success - lambda_cost * sum(
            record.tool_cost for record in zooms
        )
        row = {
            "answer_now_utility": 0.0,
            "lower_frozen_utility": (
                lower_utility
                if float(lower_scores[key]) >= float(lower_threshold)
                else 0.0
            ),
            "higher_frozen_utility": (
                higher_utility
                if float(higher_scores[key]) >= float(higher_threshold)
                else 0.0
            ),
            "lower_frozen_call": float(
                float(lower_scores[key]) >= float(lower_threshold)
            ),
            "higher_frozen_call": float(
                float(higher_scores[key]) >= float(higher_threshold)
            ),
            "lower_matched_utility": lower_utility if key in lower_matched else 0.0,
            "higher_matched_utility": (
                higher_utility if key in higher_matched else 0.0
            ),
            "lower_matched_call": float(key in lower_matched),
            "higher_matched_call": float(key in higher_matched),
            "random_gate_random_crop_expected_utility": (
                target_calls / len(keys) * random_crop_utility
            ),
            "unrestricted_privileged_oracle_utility": max(
                0.0, max(net_by_action.values())
            ),
        }
        for action_id, utility in net_by_action.items():
            row[f"entropy_gate_fixed_crop_{action_id}_utility"] = utility
        row["_random_crop_utility"] = random_crop_utility
        row["_post_action_entropy_single_utility"] = post_action_entropy_utility
        row["_ug_exhaustive_utility"] = exhaustive_utility
        decision_rows[key] = row

    if action_ids is None:
        raise RuntimeError("action bank was not initialized")
    entropy_matched = set(_top_k_keys(entropy_scores, count=target_calls))
    oracle_matched = set(_top_k_keys(oracle_scores, count=target_calls))
    values: dict[str, dict[DecisionKey, float]] = {}
    for key, raw_row in decision_rows.items():
        row = dict(raw_row)
        random_crop_utility = row.pop("_random_crop_utility")
        post_action_entropy_utility = row.pop("_post_action_entropy_single_utility")
        exhaustive_utility = row.pop("_ug_exhaustive_utility")
        entropy_called = key in entropy_matched
        row["entropy_gate_random_crop_expected_utility"] = (
            random_crop_utility if entropy_called else 0.0
        )
        row["post_action_entropy_single_execution_idealized_utility"] = (
            post_action_entropy_utility if entropy_called else 0.0
        )
        row["ug_style_exhaustive_entropy_four_cost_utility"] = (
            exhaustive_utility if entropy_called else 0.0
        )
        row["matched_privileged_oracle_utility"] = (
            max(0.0, oracle_scores[key]) if key in oracle_matched else 0.0
        )
        for action_id in action_ids:
            fixed_name = f"entropy_gate_fixed_crop_{action_id}_utility"
            if not entropy_called:
                row[fixed_name] = 0.0
        row["higher_minus_lower_frozen_utility"] = (
            row["higher_frozen_utility"] - row["lower_frozen_utility"]
        )
        row["higher_minus_lower_matched_utility"] = (
            row["higher_matched_utility"] - row["lower_matched_utility"]
        )
        for cost in extra_costs:
            suffix = _cost_metric_suffix(cost)
            adjusted = row["higher_matched_utility"] - cost
            row[f"higher_matched_utility_extra_cost_{suffix}"] = adjusted
            row[f"higher_minus_lower_matched_extra_cost_{suffix}"] = (
                adjusted - row["lower_matched_utility"]
            )
        for name, value in row.items():
            values.setdefault(name, {})[key] = float(value)

    source_metric_values = {
        name: _source_means(metric, source_by_key) for name, metric in values.items()
    }
    sources = sorted(source_metric_values["lower_matched_utility"])
    source_metrics = {
        source: {
            name: source_metric_values[name][source]
            for name in sorted(source_metric_values)
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
        name: mean(metric.values()) for name, metric in source_metric_values.items()
    }
    question_point = {name: mean(metric.values()) for name, metric in values.items()}

    deployable_names = [
        "answer_now_utility",
        "lower_matched_utility",
        "higher_matched_utility",
        "entropy_gate_random_crop_expected_utility",
        "random_gate_random_crop_expected_utility",
        "ug_style_exhaustive_entropy_four_cost_utility",
        *(f"entropy_gate_fixed_crop_{action_id}_utility" for action_id in action_ids),
    ]
    ranked_pairs = sorted(
        ((name, source_point[name]) for name in deployable_names),
        key=lambda item: (-item[1], item[0]),
    )
    deployable_ranking = [
        {"method": name, "source_balanced_utility": utility}
        for name, utility in ranked_pairs
    ]
    overlap = len(lower_matched & higher_matched)
    return {
        "scientific_status": (
            "retrospective same-bank necessary-condition audit; not formal evidence"
        ),
        "lambda_cost": lambda_cost,
        "n_decisions": len(keys),
        "n_sources": len(sources),
        "action_ids": list(action_ids),
        "matched_budget": {
            "registered_call_rate": matched_call_rate,
            "target_calls": target_calls,
            "realized_question_call_rate": target_calls / len(keys),
            "lower_higher_call_overlap": overlap,
            "lower_higher_call_jaccard": overlap / len(lower_matched | higher_matched),
            "action_agreement_rate": mean(
                lower_actions[key] == higher_actions[key] for key in keys
            ),
            "selection_uses_outcomes": False,
            "tie_break": ("descending_score_then_ascending_state_id_replicate_id"),
        },
        "source_balanced": source_point,
        "question_weighted": question_point,
        "source_bootstrap": bootstrap,
        "deployable_source_ranking_at_matched_budget": deployable_ranking,
        "baseline_contract": {
            "entropy_gate_uses_pre_action_entropy_only": True,
            "random_crop_is_expected_uniform_over_four_actions": True,
            "random_gate_is_expected_uniform_over_decisions": True,
            "post_action_entropy_single_execution_is_idealized": True,
            "ug_style_exhaustive_candidate_count": 4,
            "ug_style_exhaustive_charges_all_candidate_costs": True,
            "privileged_oracle_uses_outcomes": True,
        },
        "higher_information_extra_cost": {
            "scope": "per_decision",
            "unit": "already_lambda_weighted_utility",
            "registered_values": list(extra_costs),
            "zero_cost_is_optimistic_upper_bound": True,
        },
    }


def decide_n5_calibration_opening(
    evaluation: Mapping[str, Any],
    *,
    minimum_material_utility: float,
    screenqa_higher_minus_lower_utility: float,
    screenqa_higher_has_safe_non_degenerate_threshold: bool,
    exact_registered_information_sets_available: bool,
    same_method_factorial_available: bool,
) -> dict[str, Any]:
    """Apply the frozen N5 necessary-condition gate without tuning metrics."""

    minimum = float(minimum_material_utility)
    screenqa_gap = float(screenqa_higher_minus_lower_utility)
    if not math.isfinite(minimum) or minimum <= 0.0:
        raise ValueError("minimum material utility must be finite and positive")
    if not math.isfinite(screenqa_gap):
        raise ValueError("ScreenQA utility gap must be finite")
    source = evaluation.get("source_balanced")
    bootstrap = evaluation.get("source_bootstrap")
    if not isinstance(source, Mapping) or not isinstance(bootstrap, Mapping):
        raise ValueError("N5 evaluation is missing source-balanced evidence")
    raw_metrics = bootstrap.get("metrics")
    if not isinstance(raw_metrics, Mapping):
        raise ValueError("N5 evaluation is missing bootstrap metrics")
    higher_utility = float(source["higher_matched_utility"])
    paired_gap = float(source["higher_minus_lower_matched_utility"])
    higher_interval = raw_metrics.get("higher_matched_utility")
    paired_interval = raw_metrics.get("higher_minus_lower_matched_utility")
    if not isinstance(higher_interval, Mapping) or not isinstance(
        paired_interval, Mapping
    ):
        raise ValueError("N5 evaluation is missing primary intervals")
    higher_ci_low = float(higher_interval["ci_low"])
    paired_ci_low = float(paired_interval["ci_low"])
    if any(
        not math.isfinite(value)
        for value in (higher_utility, paired_gap, higher_ci_low, paired_ci_low)
    ):
        raise ValueError("N5 primary evidence must be finite")

    checks = {
        "exact_registered_information_sets_available": bool(
            exact_registered_information_sets_available
        ),
        "same_method_factorial_available": bool(same_method_factorial_available),
        "higher_information_matched_utility_positive": higher_utility > 0.0,
        "higher_information_matched_utility_ci_low_above_zero": (higher_ci_low > 0.0),
        "higher_minus_lower_matched_at_least_minimum": paired_gap >= minimum,
        "higher_minus_lower_matched_ci_low_above_zero": paired_ci_low > 0.0,
        "screenqa_oof_higher_minus_lower_at_least_minimum": (screenqa_gap >= minimum),
        "screenqa_higher_has_safe_non_degenerate_threshold": bool(
            screenqa_higher_has_safe_non_degenerate_threshold
        ),
    }
    passed = all(checks.values())
    return {
        "decision": (
            "n5_necessary_condition_passed_write_calibration_protocol"
            if passed
            else "n5_current_information_boundary_candidate_not_supported_before_calibration"
        ),
        "passed": passed,
        "checks": checks,
        "minimum_material_utility": minimum,
        "observed": {
            "higher_matched_source_utility": higher_utility,
            "higher_matched_source_utility_ci_low": higher_ci_low,
            "higher_minus_lower_matched_source_utility": paired_gap,
            "higher_minus_lower_matched_source_utility_ci_low": paired_ci_low,
            "screenqa_oof_higher_minus_lower_utility": screenqa_gap,
        },
        "screenqa_calibration_protocol_may_be_written": passed,
    }
