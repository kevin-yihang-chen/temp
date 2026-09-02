from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

from .dataset import DecisionKey
from .infographicvqa_attention_where import assemble_attention_where_features
from .infographicvqa_decar import DECAR_ACTION_IDS
from .infographicvqa_decar_evaluation import (
    DECAR_BOOTSTRAP_RESAMPLES,
    DECAR_CALL_RATES,
    DECAR_HYBRID_EVALUATION_SCHEMA,
    DECAR_LAMBDA_COST,
    DECAR_ORACLE_WHERE_EVALUATION_SCHEMA,
    _aggregate_policy,
    _bootstrap_all_policies,
    _complete_tie_exact_match,
    _policy_metrics,
    _source_concentration,
    build_decar_outcomes,
    parse_decar_predictions,
)
from .infographicvqa_relative_where_evaluation import (
    FROZEN_COMPARATOR_FLOAT_ABS_TOL,
    FROZEN_COMPARATOR_FLOAT_REL_TOL,
    RELATIVE_WHERE_EVALUATION_SCHEMA,
    RELATIVE_WHERE_PRIMARY,
    _oracle_gap_closure,
    _require_frozen_comparator,
    parse_relative_where_predictions,
    privileged_teacher_actions,
)
from .schema import ActionRecord

ATTENTION_WHERE_EVALUATION_SCHEMA = "infographicvqa_attention_where_evaluation_v1"
ATTENTION_WHERE_PRIMARY = "attention_where"
ATTENTION_WHERE_COMPARATORS = (
    "entropy_fixed_ug_grid_00",
    "entropy_random",
    "old_decar_where",
    "relative_where",
)


def _crop_nll_by_key(
    rows: Sequence[Mapping[str, Any]],
    outcomes: Mapping[DecisionKey, Any],
) -> dict[DecisionKey, dict[str, float]]:
    indexed: dict[tuple[str, str, str], float] = {}
    for row in rows:
        if row.get("schema") != "visual_action_answer_nll_v1":
            raise ValueError("attention-where teacher NLL schema changed")
        row_key: tuple[str, str, str] = (
            str(row.get("state_id")),
            str(row.get("replicate_id")),
            str(row.get("action_id")),
        )
        value = float(row.get("answer_mean_nll", math.nan))
        if row_key in indexed or not math.isfinite(value):
            raise ValueError("attention-where teacher NLL rows are invalid")
        indexed[row_key] = value
    result: dict[DecisionKey, dict[str, float]] = {}
    expected: set[tuple[str, str, str]] = set()
    for decision_key, outcome in outcomes.items():
        values: dict[str, float] = {}
        for record in (outcome.baseline, *outcome.crops):
            indexed_key = (decision_key[0], decision_key[1], record.action_id)
            if indexed_key not in indexed:
                raise ValueError("attention-where teacher NLL coverage is incomplete")
            expected.add(indexed_key)
            if record.action_type == "ZOOM":
                values[record.action_id] = indexed[indexed_key]
        if tuple(sorted(values)) != DECAR_ACTION_IDS:
            raise ValueError("attention-where teacher NLL action family changed")
        result[decision_key] = values
    if set(indexed) != expected:
        raise ValueError("attention-where teacher NLL coverage is not exact")
    return result


def _conditional_localization_metrics(
    *,
    action_by_key: Mapping[DecisionKey, str],
    selected_keys: set[DecisionKey],
    outcomes: Mapping[DecisionKey, Any],
    teacher_actions: Mapping[DecisionKey, str],
    crop_nll: Mapping[DecisionKey, Mapping[str, float]],
) -> dict[str, Any]:
    if set(action_by_key) != set(outcomes) or not selected_keys.issubset(outcomes):
        raise ValueError("attention-where localization coverage changed")
    names = (
        "exact_nll_teacher",
        "row_nll_teacher",
        "column_nll_teacher",
        "exact_task_oracle",
        "nll_regret",
        "helpful_state_rescue",
    )
    numerators: dict[str, defaultdict[str, float]] = {
        name: defaultdict(float) for name in names
    }
    denominators: dict[str, defaultdict[str, float]] = {
        name: defaultdict(float) for name in names
    }
    source_decisions: dict[str, int] = defaultdict(int)
    raw_numerators = {name: 0.0 for name in names}
    raw_denominators = {name: 0.0 for name in names}
    action_index = {
        action_id: index for index, action_id in enumerate(DECAR_ACTION_IDS)
    }
    for key, outcome in outcomes.items():
        source = outcome.source_id
        source_decisions[source] += 1
        if key not in selected_keys:
            continue
        selected_action = action_by_key[key]
        teacher_action = teacher_actions[key]
        if selected_action not in action_index:
            raise ValueError("attention-where selected an unknown action")
        selected_index = action_index[selected_action]
        teacher_index = action_index[teacher_action]
        crop_by_action = {record.action_id: record for record in outcome.crops}
        selected_crop = crop_by_action[selected_action]
        helpful = max(record.delta_success for record in outcome.crops) > 0.0
        values = {
            "exact_nll_teacher": float(selected_action == teacher_action),
            "row_nll_teacher": float(selected_index // 2 == teacher_index // 2),
            "column_nll_teacher": float(selected_index % 2 == teacher_index % 2),
            "exact_task_oracle": float(
                selected_action == outcome.task_action.action_id
            ),
            "nll_regret": crop_nll[key][selected_action] - min(crop_nll[key].values()),
            "helpful_state_rescue": float(
                helpful and selected_crop.delta_success > 0.0
            ),
        }
        metric_denominators = {
            name: float(helpful) if name == "helpful_state_rescue" else 1.0
            for name in names
        }
        for name in names:
            denominator = metric_denominators[name]
            numerators[name][source] += values[name]
            denominators[name][source] += denominator
            raw_numerators[name] += values[name]
            raw_denominators[name] += denominator

    sources = sorted(source_decisions)
    result: dict[str, Any] = {}
    for name in names:
        source_numerator = sum(
            numerators[name][source] / source_decisions[source] for source in sources
        ) / len(sources)
        source_denominator = sum(
            denominators[name][source] / source_decisions[source] for source in sources
        ) / len(sources)
        result[name] = {
            "question_balanced": (
                raw_numerators[name] / raw_denominators[name]
                if raw_denominators[name] > 0.0
                else None
            ),
            "source_balanced": (
                source_numerator / source_denominator
                if source_denominator > 0.0
                else None
            ),
            "raw_denominator": raw_denominators[name],
        }
    result["selected_decisions"] = len(selected_keys)
    result["selected_sources"] = len({outcomes[key].source_id for key in selected_keys})
    return result


def _attention_deciles(
    *,
    field_by_key: Mapping[DecisionKey, float],
    action_by_key: Mapping[DecisionKey, str],
    outcomes: Mapping[DecisionKey, Any],
    teacher_actions: Mapping[DecisionKey, str],
    crop_nll: Mapping[DecisionKey, Mapping[str, float]],
) -> list[dict[str, Any]]:
    if set(field_by_key) != set(outcomes):
        raise ValueError("attention-where decile field coverage changed")
    ordered = sorted(outcomes, key=lambda key: (field_by_key[key], key))
    result: list[dict[str, Any]] = []
    bin_count = min(10, len(ordered))
    for index in range(bin_count):
        start = index * len(ordered) // bin_count
        stop = (index + 1) * len(ordered) // bin_count
        keys = set(ordered[start:stop])
        values = [field_by_key[key] for key in ordered[start:stop]]
        result.append(
            {
                "name": f"decile-{index:02d}",
                "rank_low_inclusive": index / bin_count,
                "rank_high_exclusive": (index + 1) / bin_count,
                "field_min": min(values),
                "field_max": max(values),
                "localization": _conditional_localization_metrics(
                    action_by_key=action_by_key,
                    selected_keys=keys,
                    outcomes=outcomes,
                    teacher_actions=teacher_actions,
                    crop_nll=crop_nll,
                ),
            }
        )
    return result


def _paired_utility_differences(
    *,
    primary: Mapping[str, Any],
    comparators: Mapping[str, Mapping[str, Any]],
    sources: Sequence[str],
    bootstrap_indices: Any,
) -> dict[str, dict[str, float]]:
    import numpy as np  # type: ignore[import-not-found]

    indices = np.asarray(bootstrap_indices)
    primary_values = np.asarray(
        [primary["source_values"][source]["utility"] for source in sources],
        dtype=np.float64,
    )
    result: dict[str, dict[str, float]] = {}
    for name, comparator in comparators.items():
        comparator_values = np.asarray(
            [comparator["source_values"][source]["utility"] for source in sources],
            dtype=np.float64,
        )
        difference = primary_values - comparator_values
        draws: list[Any] = []
        for start in range(0, indices.shape[0], 256):
            sampled = indices[start : start + 256]
            draws.append(difference[sampled].mean(axis=1))
        values = np.concatenate(draws)
        result[name] = {
            "point_estimate": float(difference.mean()),
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
        }
    return result


def evaluate_attention_where(
    records: Sequence[ActionRecord],
    attention_feature_payload: Mapping[str, Any],
    decar_prediction_rows: Sequence[Mapping[str, Any]],
    relative_prediction_rows: Sequence[Mapping[str, Any]],
    nll_rows: Sequence[Mapping[str, Any]],
    hybrid_evaluation: Mapping[str, Any],
    oracle_evaluation: Mapping[str, Any],
    relative_evaluation: Mapping[str, Any],
    *,
    expected_attention_code_revision: str,
    expected_model_revision: str,
    expected_source_features_sha256: str,
    expected_rollouts_sha256: str,
    bootstrap_indices: Any,
    expected_decisions: int | None = 23_946,
    expected_sources: int | None = 2_204,
    expected_bootstrap_resamples: int = DECAR_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
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
    decar_predictions = parse_decar_predictions(decar_prediction_rows, outcomes)
    relative_predictions = parse_relative_where_predictions(
        relative_prediction_rows, outcomes
    )
    teacher_actions = privileged_teacher_actions(nll_rows, outcomes)
    crop_nll = _crop_nll_by_key(nll_rows, outcomes)
    keys = sorted(outcomes)
    attention_keys = list(
        zip(attention.state_ids, attention.replicate_ids, strict=True)
    )
    if attention_keys != keys:
        raise ValueError("attention-where feature ordering changed")
    sources = sorted({outcome.source_id for outcome in outcomes.values()})
    hybrid_points = hybrid_evaluation.get("operating_points")
    oracle_points = oracle_evaluation.get("operating_points")
    relative_points = relative_evaluation.get("operating_points")
    if (
        hybrid_evaluation.get("schema") != DECAR_HYBRID_EVALUATION_SCHEMA
        or hybrid_evaluation.get("decision") != "hybrid_train_not_supported"
        or oracle_evaluation.get("schema") != DECAR_ORACLE_WHERE_EVALUATION_SCHEMA
        or oracle_evaluation.get("decision") != "where_bottleneck_supported"
        or relative_evaluation.get("schema") != RELATIVE_WHERE_EVALUATION_SCHEMA
        or relative_evaluation.get("decision") != "relative_where_train_not_supported"
        or hybrid_evaluation.get("validation_or_test_inputs_used") is not False
        or oracle_evaluation.get("validation_or_test_inputs_used") is not False
        or relative_evaluation.get("validation_or_test_inputs_used") is not False
        or tuple(hybrid_evaluation.get("registered_call_rates", ())) != DECAR_CALL_RATES
        or tuple(oracle_evaluation.get("registered_call_rates", ())) != DECAR_CALL_RATES
        or tuple(relative_evaluation.get("registered_call_rates", ()))
        != DECAR_CALL_RATES
        or not isinstance(hybrid_points, list)
        or not isinstance(oracle_points, list)
        or not isinstance(relative_points, list)
        or len(hybrid_points) != len(DECAR_CALL_RATES)
        or len(oracle_points) != len(DECAR_CALL_RATES)
        or len(relative_points) != len(DECAR_CALL_RATES)
    ):
        raise ValueError("attention-where frozen evaluation dependency changed")
    if tuple(bootstrap_indices.shape) != (expected_bootstrap_resamples, len(sources)):
        raise ValueError("attention-where bootstrap shape changed")
    if str(bootstrap_indices.dtype) != "int32":
        raise ValueError("attention-where bootstrap dtype changed")

    attention_actions = {
        key: DECAR_ACTION_IDS[int(attention.selected_indices[index])]
        for index, key in enumerate(keys)
    }
    old_decar_actions = {
        key: decar_predictions[key].variants["decar"].action_id for key in keys
    }
    relative_actions = {
        key: str(
            relative_predictions[key]["variants"][RELATIVE_WHERE_PRIMARY][
                "selected_action_id"
            ]
        )
        for key in keys
    }
    entropy_scores = {key: outcomes[key].baseline.entropy_before for key in keys}
    answer_now = _policy_metrics(outcomes, called_keys=set())
    policy_values: dict[str, dict[DecisionKey, dict[str, float]]] = {}
    operating: list[dict[str, Any]] = []
    called_by_point: dict[str, set[DecisionKey]] = {}
    for hybrid_point, oracle_point, relative_point, rate in zip(
        hybrid_points,
        oracle_points,
        relative_points,
        DECAR_CALL_RATES,
        strict=True,
    ):
        point_name = f"rate-{rate:.3f}"
        actual_calls = int(hybrid_point.get("actual_calls", -1))
        hybrid_selection = hybrid_point.get("selection_audits")
        if (
            hybrid_point.get("name") != point_name
            or oracle_point.get("name") != point_name
            or relative_point.get("name") != point_name
            or oracle_point.get("actual_calls") != actual_calls
            or relative_point.get("actual_calls") != actual_calls
            or not isinstance(hybrid_selection, Mapping)
        ):
            raise ValueError("attention-where operating-point family changed")
        called, selection_audit = _complete_tie_exact_match(
            entropy_scores, target_calls=actual_calls
        )
        if selection_audit != hybrid_selection.get("entropy_one_crop"):
            raise ValueError("attention-where entropy identity audit failed")
        called_by_point[point_name] = called
        values = {
            ATTENTION_WHERE_PRIMARY: _policy_metrics(
                outcomes, called_keys=called, action_by_key=attention_actions
            ),
            "entropy_fixed_ug_grid_00": _policy_metrics(
                outcomes,
                called_keys=called,
                action_by_key={key: "ug-grid-00" for key in keys},
            ),
            "entropy_random": _policy_metrics(
                outcomes, called_keys=called, random_action=True
            ),
            "old_decar_where": _policy_metrics(
                outcomes, called_keys=called, action_by_key=old_decar_actions
            ),
            "relative_where": _policy_metrics(
                outcomes, called_keys=called, action_by_key=relative_actions
            ),
            "answer_now": answer_now,
            "privileged_teacher_nll_where": _policy_metrics(
                outcomes, called_keys=called, action_by_key=teacher_actions
            ),
            "task_oracle_where": _policy_metrics(
                outcomes, called_keys=called, task_action=True
            ),
        }
        for name, rows in values.items():
            policy_values[f"{point_name}/{name}"] = rows
        operating.append(
            {
                "name": point_name,
                "nominal_question_call_rate": rate,
                "actual_calls": actual_calls,
                "selection_audit": selection_audit,
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
    for registered, hybrid_point, oracle_point, relative_point in zip(
        operating, hybrid_points, oracle_points, relative_points, strict=True
    ):
        point_name = str(registered["name"])
        frozen = {
            "entropy_fixed_ug_grid_00": hybrid_point["policies"][
                "entropy_fixed_ug_grid_00"
            ],
            "entropy_random": hybrid_point["policies"]["entropy_random"],
            "old_decar_where": hybrid_point["policies"]["entropy_when_decar_where"],
            "answer_now": hybrid_point["policies"]["answer_now"],
            "task_oracle_where": oracle_point["policies"][
                "entropy_when_task_oracle_where"
            ],
            "relative_where": relative_point["policies"][RELATIVE_WHERE_PRIMARY],
            "privileged_teacher_nll_where": relative_point["policies"][
                "privileged_teacher_nll_where"
            ],
        }
        for name, expected in frozen.items():
            expected_base = {
                key: value
                for key, value in expected.items()
                if key != "teacher_agreement"
            }
            _require_frozen_comparator(
                name=f"{point_name}/{name}",
                recomputed=public[f"{point_name}/{name}"],
                frozen=expected_base,
            )
        registered["frozen_comparators_exact_match"] = True

    bootstrap, _ = _bootstrap_all_policies(aggregates, sources, bootstrap_indices)
    policy_names = (
        ATTENTION_WHERE_PRIMARY,
        "entropy_fixed_ug_grid_00",
        "entropy_random",
        "old_decar_where",
        "relative_where",
        "answer_now",
        "privileged_teacher_nll_where",
        "task_oracle_where",
    )
    points: list[dict[str, Any]] = []
    for registered in operating:
        point_name = str(registered["name"])
        policies = {name: public[f"{point_name}/{name}"] for name in policy_names}
        policy_bootstrap = {
            name: bootstrap["policies"][f"{point_name}/{name}"] for name in policy_names
        }
        primary_aggregate = aggregates[f"{point_name}/{ATTENTION_WHERE_PRIMARY}"]
        comparator_aggregates = {
            name: aggregates[f"{point_name}/{name}"]
            for name in ATTENTION_WHERE_COMPARATORS
        }
        differences = _paired_utility_differences(
            primary=primary_aggregate,
            comparators=comparator_aggregates,
            sources=sources,
            bootstrap_indices=bootstrap_indices,
        )
        closure = _oracle_gap_closure(
            primary_aggregate,
            aggregates[f"{point_name}/entropy_random"],
            aggregates[f"{point_name}/task_oracle_where"],
            sources,
            bootstrap_indices,
        )
        best_comparator = max(
            ATTENTION_WHERE_COMPARATORS,
            key=lambda name: float(policies[name]["source_balanced"]["utility"]),
        )
        primary = policies[ATTENTION_WHERE_PRIMARY]["source_balanced"]
        primary_interval = policy_bootstrap[ATTENTION_WHERE_PRIMARY]["additive"][
            "utility"
        ]
        paired_lows = [
            float(differences[name]["ci_low"]) for name in ATTENTION_WHERE_COMPARATORS
        ]
        rules = {
            "all_audits_passed": bool(
                feature_audit["passed"]
                and registered["selection_audit"]["matched_call_count"]
                and registered["frozen_comparators_exact_match"]
            ),
            "primary_source_utility_positive": float(primary["utility"]) > 0.0,
            "primary_source_utility_ci_low_strictly_positive": float(
                primary_interval["ci_low"]
            )
            > 0.0,
            "paired_noninferior_to_all_registered_comparators": all(
                value >= 0.0 for value in paired_lows
            ),
            "paired_strictly_superior_to_at_least_one_comparator": any(
                value > 0.0 for value in paired_lows
            ),
            "induced_harm_within_registered_tolerance": float(primary["induced_harm"])
            <= float(policies[best_comparator]["source_balanced"]["induced_harm"])
            + 0.00025,
            "random_to_oracle_gap_closure_at_least_quarter": closure["point_estimate"]
            is not None
            and float(closure["point_estimate"]) >= 0.25,
        }
        called = called_by_point[point_name]
        points.append(
            {
                **registered,
                "policies": policies,
                "source_bootstrap": policy_bootstrap,
                "paired_source_utility_differences": differences,
                "random_to_oracle_gap_closure": closure,
                "best_deployable_comparator": best_comparator,
                "attention_localization_on_called_states": _conditional_localization_metrics(
                    action_by_key=attention_actions,
                    selected_keys=called,
                    outcomes=outcomes,
                    teacher_actions=teacher_actions,
                    crop_nll=crop_nll,
                ),
                "source_concentration": _source_concentration(
                    policy_values[f"{point_name}/{ATTENTION_WHERE_PRIMARY}"], outcomes
                ),
                "qualification_rules": rules,
                "qualified": all(rules.values()),
            }
        )
    qualified = [point for point in points if point["qualified"]]
    selected = (
        min(
            qualified,
            key=lambda point: (
                -float(
                    point["source_bootstrap"][ATTENTION_WHERE_PRIMARY]["additive"][
                        "utility"
                    ]["ci_low"]
                ),
                -float(
                    point["policies"][ATTENTION_WHERE_PRIMARY]["source_balanced"][
                        "utility"
                    ]
                ),
                float(
                    point["policies"][ATTENTION_WHERE_PRIMARY]["source_balanced"][
                        "induced_harm"
                    ]
                ),
                float(point["nominal_question_call_rate"]),
            ),
        )
        if qualified
        else None
    )
    max_scores = {
        key: float(attention.scores[index].max()) for index, key in enumerate(keys)
    }
    margins = {key: float(attention.margins[index]) for index, key in enumerate(keys)}
    return {
        "schema": ATTENTION_WHERE_EVALUATION_SCHEMA,
        "scientific_status": "frozen InfographicVQA official-train raw-attention where evaluation",
        "population": {
            "decisions": len(outcomes),
            "sources": len(sources),
            "images": len({outcome.image_id for outcome in outcomes.values()}),
        },
        "primary": ATTENTION_WHERE_PRIMARY,
        "lambda_cost": DECAR_LAMBDA_COST,
        "registered_call_rates": list(DECAR_CALL_RATES),
        "frozen_comparator_float_tolerance": {
            "relative": FROZEN_COMPARATOR_FLOAT_REL_TOL,
            "absolute": FROZEN_COMPARATOR_FLOAT_ABS_TOL,
            "discrete_fields_exact": True,
        },
        "feature_audit": feature_audit,
        "all_state_attention_localization": _conditional_localization_metrics(
            action_by_key=attention_actions,
            selected_keys=set(keys),
            outcomes=outcomes,
            teacher_actions=teacher_actions,
            crop_nll=crop_nll,
        ),
        "attention_max_score_deciles": _attention_deciles(
            field_by_key=max_scores,
            action_by_key=attention_actions,
            outcomes=outcomes,
            teacher_actions=teacher_actions,
            crop_nll=crop_nll,
        ),
        "attention_margin_deciles": _attention_deciles(
            field_by_key=margins,
            action_by_key=attention_actions,
            outcomes=outcomes,
            teacher_actions=teacher_actions,
            crop_nll=crop_nll,
        ),
        "operating_points": points,
        "bootstrap": bootstrap["metadata"],
        "decision": (
            "attention_where_train_supported"
            if selected is not None
            else "attention_where_train_not_supported"
        ),
        "selected_operating_point": (
            None
            if selected is None
            else {
                "name": selected["name"],
                "nominal_question_call_rate": selected["nominal_question_call_rate"],
                "actual_calls": selected["actual_calls"],
                "source_balanced_utility": selected["policies"][
                    ATTENTION_WHERE_PRIMARY
                ]["source_balanced"]["utility"],
                "source_balanced_utility_ci_low": selected["source_bootstrap"][
                    ATTENTION_WHERE_PRIMARY
                ]["additive"]["utility"]["ci_low"],
                "source_balanced_induced_harm": selected["policies"][
                    ATTENTION_WHERE_PRIMARY
                ]["source_balanced"]["induced_harm"],
                "random_to_oracle_gap_closure": selected[
                    "random_to_oracle_gap_closure"
                ]["point_estimate"],
            }
        ),
        "attention_features_outcomes_included": False,
        "privileged_teacher_used_only_in_evaluation": True,
        "validation_or_test_inputs_used": False,
    }
