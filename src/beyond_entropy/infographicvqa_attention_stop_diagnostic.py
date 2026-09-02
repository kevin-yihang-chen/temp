from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .dataset import DecisionKey
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
    build_decar_outcomes,
)
from .schema import ActionRecord

ATTENTION_STOP_DIAGNOSTIC_SCHEMA = (
    "infographicvqa_attention_stop_factorization_diagnostic_v1"
)


def _positive_net_keys(
    outcomes: Mapping[DecisionKey, Any],
    action_by_key: Mapping[DecisionKey, str],
) -> tuple[set[DecisionKey], dict[DecisionKey, float]]:
    if set(action_by_key) != set(outcomes):
        raise ValueError("attention-stop action coverage changed")
    net_values: dict[DecisionKey, float] = {}
    for key, outcome in outcomes.items():
        crops = {crop.action_id: crop for crop in outcome.crops}
        action_id = action_by_key[key]
        if action_id not in crops:
            raise ValueError("attention-stop action is absent from outcome siblings")
        crop = crops[action_id]
        net_values[key] = crop.delta_success - DECAR_LAMBDA_COST * crop.tool_cost
    return {key for key, value in net_values.items() if value > 0.0}, net_values


def _at_most_positive_top(
    net_values: Mapping[DecisionKey, float],
    *,
    budget: int,
) -> set[DecisionKey]:
    if budget < 0:
        raise ValueError("attention-stop oracle budget must be nonnegative")
    ordered = sorted(net_values, key=lambda key: (-float(net_values[key]), key))
    return {key for key in ordered[:budget] if float(net_values[key]) > 0.0}


def _selection_diagnostic(
    called: set[DecisionKey],
    positive_net: set[DecisionKey],
    outcomes: Mapping[DecisionKey, Any],
) -> dict[str, float | int | None]:
    if not called.issubset(outcomes) or not positive_net.issubset(outcomes):
        raise ValueError("attention-stop selection identity changed")
    true_positive = len(called.intersection(positive_net))
    called_sources = {outcomes[key].source_id for key in called}
    positive_sources = {outcomes[key].source_id for key in positive_net}
    return {
        "calls": len(called),
        "called_sources": len(called_sources),
        "positive_net_states": len(positive_net),
        "positive_net_sources": len(positive_sources),
        "positive_net_calls": true_positive,
        "positive_net_precision": (true_positive / len(called) if called else None),
        "positive_net_recall": (
            true_positive / len(positive_net) if positive_net else None
        ),
    }


def evaluate_attention_stop_factorization(
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
    expected_bootstrap_resamples: int = DECAR_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Decompose stop versus where headroom using already opened train outcomes.

    This is deliberately descriptive. Realized-utility call sets are privileged
    ceilings, while entropy/max/margin call sets remain outcome-free diagnostics.
    """

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
    keys = sorted(outcomes)
    attention_keys = list(
        zip(attention.state_ids, attention.replicate_ids, strict=True)
    )
    if attention_keys != keys:
        raise ValueError("attention-stop feature ordering changed")
    sources = sorted({outcome.source_id for outcome in outcomes.values()})
    if (
        tuple(bootstrap_indices.shape)
        != (
            expected_bootstrap_resamples,
            len(sources),
        )
        or str(bootstrap_indices.dtype) != "int32"
    ):
        raise ValueError("attention-stop bootstrap contract changed")

    action_by_key = {
        key: DECAR_ACTION_IDS[int(attention.selected_indices[index])]
        for index, key in enumerate(keys)
    }
    positive_net, net_values = _positive_net_keys(outcomes, action_by_key)
    task_positive = {
        key for key, outcome in outcomes.items() if outcome.oracle_stop_utility > 0.0
    }
    max_scores = {
        key: float(attention.scores[index].max()) for index, key in enumerate(keys)
    }
    margins = {key: float(attention.margins[index]) for index, key in enumerate(keys)}
    entropy = {key: float(outcomes[key].baseline.entropy_before) for key in keys}

    policy_values: dict[str, dict[DecisionKey, dict[str, float]]] = {
        "answer_now": _policy_metrics(outcomes, called_keys=set()),
        "raw_action_positive_net_oracle": _policy_metrics(
            outcomes,
            called_keys=positive_net,
            action_by_key=action_by_key,
        ),
        "task_action_positive_net_oracle": _policy_metrics(
            outcomes,
            called_keys=task_positive,
            task_action=True,
        ),
    }
    operating: list[dict[str, Any]] = []
    for rate in DECAR_CALL_RATES:
        name = f"rate-{rate:.3f}"
        target_calls = min(len(keys), max(1, math.ceil(rate * len(keys))))
        entropy_called, entropy_audit = _complete_tie_exact_match(
            entropy, target_calls=target_calls
        )
        max_called, max_audit = _complete_tie_exact_match(
            max_scores, target_calls=target_calls
        )
        margin_called, margin_audit = _complete_tie_exact_match(
            margins, target_calls=target_calls
        )
        budget_oracle = _at_most_positive_top(net_values, budget=target_calls)
        call_sets = {
            "entropy_stop": entropy_called,
            "attention_max_stop": max_called,
            "attention_margin_stop": margin_called,
            "raw_action_budget_oracle_stop": budget_oracle,
        }
        for policy_name, called in call_sets.items():
            policy_values[f"{name}/{policy_name}"] = _policy_metrics(
                outcomes,
                called_keys=called,
                action_by_key=action_by_key,
            )
        operating.append(
            {
                "name": name,
                "nominal_question_call_rate": rate,
                "target_calls": target_calls,
                "selection_audits": {
                    "entropy_stop": entropy_audit,
                    "attention_max_stop": max_audit,
                    "attention_margin_stop": margin_audit,
                    "raw_action_budget_oracle_stop": {
                        "target_calls": target_calls,
                        "actual_calls": len(budget_oracle),
                        "at_most_budget": True,
                        "selection_uses_outcomes": True,
                    },
                },
                "selection_diagnostics": {
                    policy_name: _selection_diagnostic(called, positive_net, outcomes)
                    for policy_name, called in call_sets.items()
                },
            }
        )

    aggregates = {
        name: _aggregate_policy(values, outcomes, sources)
        for name, values in policy_values.items()
    }
    public = {
        name: {key: value for key, value in aggregate.items() if key != "source_values"}
        for name, aggregate in aggregates.items()
    }
    bootstrap, _ = _bootstrap_all_policies(aggregates, sources, bootstrap_indices)
    points: list[dict[str, Any]] = []
    for registered in operating:
        name = str(registered["name"])
        policy_names = (
            "entropy_stop",
            "attention_max_stop",
            "attention_margin_stop",
            "raw_action_budget_oracle_stop",
        )
        differences = {
            policy_name: _paired_utility_differences(
                primary=aggregates[f"{name}/{policy_name}"],
                comparators={"entropy_stop": aggregates[f"{name}/entropy_stop"]},
                sources=sources,
                bootstrap_indices=bootstrap_indices,
            )["entropy_stop"]
            for policy_name in policy_names
            if policy_name != "entropy_stop"
        }
        points.append(
            {
                **registered,
                "policies": {
                    policy_name: public[f"{name}/{policy_name}"]
                    for policy_name in policy_names
                },
                "source_bootstrap": {
                    policy_name: bootstrap["policies"][f"{name}/{policy_name}"]
                    for policy_name in policy_names
                },
                "paired_source_utility_difference_from_entropy_stop": differences,
            }
        )

    ceiling_names = (
        "answer_now",
        "raw_action_positive_net_oracle",
        "task_action_positive_net_oracle",
    )
    return {
        "schema": ATTENTION_STOP_DIAGNOSTIC_SCHEMA,
        "scientific_status": (
            "post-hoc opened-train stop-versus-where diagnostic; privileged "
            "ceilings are not deployable or valid for formal selection"
        ),
        "population": {
            "decisions": len(outcomes),
            "sources": len(sources),
            "images": len({outcome.image_id for outcome in outcomes.values()}),
        },
        "lambda_cost": DECAR_LAMBDA_COST,
        "feature_audit": feature_audit,
        "raw_action_positive_net": _selection_diagnostic(
            positive_net, positive_net, outcomes
        ),
        "ceilings": {
            name: {
                "metrics": public[name],
                "source_bootstrap": bootstrap["policies"][name],
            }
            for name in ceiling_names
        },
        "operating_points": points,
        "bootstrap": bootstrap["metadata"],
        "validation_or_test_inputs_used": False,
        "privileged_outcomes_used_for_diagnostic": True,
        "valid_for_formal_selection": False,
    }
