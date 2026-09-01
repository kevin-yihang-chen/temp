from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any

from .action_value import _decision_rows
from .minimum_rank_consensus_gate import _SERIALIZED_FIELDS
from .rescue_gate import DecisionKey
from .schema import ActionRecord


DECOMPOSITION_DECISIONS = 13_580
DECOMPOSITION_SOURCES = 3_500
DECOMPOSITION_CALLS = 225
DECOMPOSITION_INTERSECTION = 180
DECOMPOSITION_EXCLUSIVE = 45
DECOMPOSITION_LAMBDA_COST = 0.05


def _score_index(
    rows: Sequence[Mapping[str, Any]], keys: set[DecisionKey]
) -> dict[str, dict[DecisionKey, Any]]:
    result: dict[str, dict[DecisionKey, Any]] = {
        "incumbent_action": {},
        "consensus_action": {},
        "incumbent_call": {},
        "consensus_call": {},
    }
    for row in rows:
        if set(row) != _SERIALIZED_FIELDS:
            raise ValueError("consensus decomposition score-row schema changed")
        key = (str(row["state_id"]), str(row["replicate_id"]))
        incumbent_action = str(row["incumbent_action_id"])
        consensus_action = str(row["minimum_rank_consensus_gate_action_id"])
        incumbent_call = row["incumbent_called"]
        consensus_call = row["minimum_rank_consensus_gate_called"]
        if (
            not all(key)
            or key in result["incumbent_action"]
            or not incumbent_action
            or not consensus_action
            or not isinstance(incumbent_call, bool)
            or not isinstance(consensus_call, bool)
        ):
            raise ValueError("consensus decomposition score row is invalid")
        result["incumbent_action"][key] = incumbent_action
        result["consensus_action"][key] = consensus_action
        result["incumbent_call"][key] = incumbent_call
        result["consensus_call"][key] = consensus_call
    if any(set(values) != keys for values in result.values()):
        raise ValueError("consensus decomposition score coverage differs")
    return result


def _chosen_actions(
    action_ids: Mapping[DecisionKey, str],
    zooms: Mapping[DecisionKey, Sequence[ActionRecord]],
) -> dict[DecisionKey, ActionRecord]:
    result: dict[DecisionKey, ActionRecord] = {}
    for key, action_id in action_ids.items():
        matches = [action for action in zooms[key] if action.action_id == action_id]
        if len(matches) != 1:
            raise ValueError(f"decomposition action is invalid for {key!r}")
        result[key] = matches[0]
    return result


def _bucket_metrics(
    selected_keys: set[DecisionKey],
    *,
    actions: Mapping[DecisionKey, ActionRecord],
    baselines: Mapping[DecisionKey, ActionRecord],
) -> dict[str, Any]:
    if not selected_keys <= set(baselines) or set(actions) != set(baselines):
        raise ValueError("decomposition bucket coverage is invalid")
    source_totals: dict[str, int] = {}
    source_sums: dict[str, dict[str, float]] = {}
    names = ("gain", "utility", "call", "induced_harm", "negative_value_call", "helpful_call")
    for key, baseline in baselines.items():
        source_totals[baseline.source_id] = source_totals.get(baseline.source_id, 0) + 1
        source_sums.setdefault(
            baseline.source_id, {name: 0.0 for name in names}
        )
    selected_values: dict[str, list[float]] = {
        name: [] for name in ("gain", "utility", "induced_harm", "negative_value_call", "helpful_call")
    }
    for key in sorted(selected_keys):
        action = actions[key]
        gain = float(action.delta_success)
        utility = float(action.voi(DECOMPOSITION_LAMBDA_COST))
        values = {
            "gain": gain,
            "utility": utility,
            "call": 1.0,
            "induced_harm": max(-gain, 0.0),
            "negative_value_call": float(utility < 0.0),
            "helpful_call": float(gain > 0.0),
        }
        source = baselines[key].source_id
        for name, value in values.items():
            source_sums[source][name] += value
            if name in selected_values:
                selected_values[name].append(value)
    source_balanced = {
        name: mean(
            source_sums[source][name] / source_totals[source]
            for source in source_totals
        )
        for name in names
    }
    question_balanced = {
        name: sum(source_sums[source][name] for source in source_sums)
        / len(baselines)
        for name in names
    }
    per_call = {
        name: (mean(values) if values else None)
        for name, values in selected_values.items()
    }
    return {
        "calls": len(selected_keys),
        "source_balanced_contribution": source_balanced,
        "question_balanced_contribution": question_balanced,
        "per_called_decision": per_call,
    }


def _intersection_paired(
    intersection: set[DecisionKey],
    *,
    incumbent_actions: Mapping[DecisionKey, ActionRecord],
    consensus_actions: Mapping[DecisionKey, ActionRecord],
) -> dict[str, Any]:
    differences: dict[str, list[float]] = {
        name: [] for name in ("gain", "utility", "induced_harm")
    }
    better = equal = worse = 0
    disagreements = 0
    for key in sorted(intersection):
        incumbent = incumbent_actions[key]
        consensus = consensus_actions[key]
        incumbent_utility = float(incumbent.voi(DECOMPOSITION_LAMBDA_COST))
        consensus_utility = float(consensus.voi(DECOMPOSITION_LAMBDA_COST))
        difference = consensus_utility - incumbent_utility
        if difference > 0.0:
            better += 1
        elif difference < 0.0:
            worse += 1
        else:
            equal += 1
        disagreements += incumbent.action_id != consensus.action_id
        differences["gain"].append(
            float(consensus.delta_success - incumbent.delta_success)
        )
        differences["utility"].append(difference)
        differences["induced_harm"].append(
            max(-float(consensus.delta_success), 0.0)
            - max(-float(incumbent.delta_success), 0.0)
        )
    return {
        "calls": len(intersection),
        "action_disagreements": disagreements,
        "consensus_action_better_equal_worse": {
            "better": better,
            "equal": equal,
            "worse": worse,
        },
        "mean_consensus_minus_incumbent_per_intersection_call": {
            name: mean(values) for name, values in differences.items()
        },
    }


def decompose_minimum_rank_consensus(
    records: Sequence[ActionRecord],
    score_rows: Sequence[Mapping[str, Any]],
    evaluation_report: Mapping[str, Any],
) -> dict[str, Any]:
    if evaluation_report.get("decision") != "minimum_rank_consensus_gate_not_advanced":
        raise ValueError("consensus decomposition requires the frozen negative result")
    baselines, zoom_lists = _decision_rows(records)
    zooms = {key: tuple(values) for key, values in zoom_lists.items()}
    keys = set(baselines)
    if (
        len(keys) != DECOMPOSITION_DECISIONS
        or len({baseline.source_id for baseline in baselines.values()})
        != DECOMPOSITION_SOURCES
        or any(len(values) != 4 for values in zooms.values())
    ):
        raise ValueError("consensus decomposition population changed")
    indexed = _score_index(score_rows, keys)
    incumbent_calls = {
        key for key, value in indexed["incumbent_call"].items() if value
    }
    consensus_calls = {
        key for key, value in indexed["consensus_call"].items() if value
    }
    intersection = incumbent_calls & consensus_calls
    incumbent_only = incumbent_calls - consensus_calls
    consensus_only = consensus_calls - incumbent_calls
    if (
        len(incumbent_calls) != DECOMPOSITION_CALLS
        or len(consensus_calls) != DECOMPOSITION_CALLS
        or len(intersection) != DECOMPOSITION_INTERSECTION
        or len(incumbent_only) != DECOMPOSITION_EXCLUSIVE
        or len(consensus_only) != DECOMPOSITION_EXCLUSIVE
    ):
        raise ValueError("consensus decomposition call-set contract changed")

    incumbent_actions = _chosen_actions(indexed["incumbent_action"], zooms)
    consensus_actions = _chosen_actions(indexed["consensus_action"], zooms)
    buckets = {
        "incumbent_intersection": _bucket_metrics(
            intersection, actions=incumbent_actions, baselines=baselines
        ),
        "incumbent_only": _bucket_metrics(
            incumbent_only, actions=incumbent_actions, baselines=baselines
        ),
        "consensus_intersection": _bucket_metrics(
            intersection, actions=consensus_actions, baselines=baselines
        ),
        "consensus_only": _bucket_metrics(
            consensus_only, actions=consensus_actions, baselines=baselines
        ),
    }
    metric_names = (
        "gain",
        "utility",
        "call",
        "induced_harm",
        "negative_value_call",
        "helpful_call",
    )
    for weighting, field in (
        ("source_balanced", "source_balanced_contribution"),
        ("question_balanced", "question_balanced_contribution"),
    ):
        for method, bucket_names in (
            ("incumbent", ("incumbent_intersection", "incumbent_only")),
            (
                "minimum_rank_consensus_gate",
                ("consensus_intersection", "consensus_only"),
            ),
        ):
            for metric in metric_names:
                reconstructed = sum(
                    float(buckets[bucket][field][metric]) for bucket in bucket_names
                )
                expected = float(evaluation_report[weighting][method][metric])
                if not math.isclose(
                    reconstructed, expected, rel_tol=0.0, abs_tol=1e-15
                ):
                    raise ValueError(
                        f"consensus decomposition does not reproduce {weighting} {method} {metric}"
                    )

    return {
        "scientific_status": (
            "post-hoc descriptive decomposition of two frozen realized policies; "
            "no new policy, action combination, threshold, or oracle is evaluated"
        ),
        "n_sources": DECOMPOSITION_SOURCES,
        "n_decisions": DECOMPOSITION_DECISIONS,
        "call_sets": {
            "incumbent": len(incumbent_calls),
            "consensus": len(consensus_calls),
            "intersection": len(intersection),
            "incumbent_only": len(incumbent_only),
            "consensus_only": len(consensus_only),
        },
        "buckets": buckets,
        "intersection_paired_action_description": _intersection_paired(
            intersection,
            incumbent_actions=incumbent_actions,
            consensus_actions=consensus_actions,
        ),
        "audits": {
            "frozen_negative_result_required": True,
            "score_rows_exact_and_outcome_free": True,
            "population_exact": True,
            "call_sets_exact": True,
            "source_and_question_metrics_reproduced": True,
            "new_policy_evaluated": False,
            "oracle_used": False,
            "screenqa_inputs_used": False,
            "protected_role_inputs_used": False,
        },
    }
