from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

from .dataset import DecisionKey
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
    _finite,
    _policy_metrics,
    _prediction_forbidden_fields,
    build_decar_outcomes,
    parse_decar_predictions,
)
from .infographicvqa_relative_where import (
    RELATIVE_WHERE_SCHEMA,
    RELATIVE_WHERE_VARIANTS,
)
from .schema import ActionRecord


RELATIVE_WHERE_EVALUATION_SCHEMA = "infographicvqa_relative_where_oof_evaluation_v1"
RELATIVE_WHERE_PRIMARY = "relative_teacher_entropy"


def parse_relative_where_predictions(
    rows: Sequence[Mapping[str, Any]],
    outcomes: Mapping[DecisionKey, Any],
) -> dict[DecisionKey, dict[str, Any]]:
    if len(rows) != len(outcomes):
        raise ValueError("relative-where prediction coverage changed")
    parsed: dict[DecisionKey, dict[str, Any]] = {}
    for row in rows:
        forbidden = _prediction_forbidden_fields(row)
        if forbidden:
            raise ValueError(
                f"relative-where prediction contains forbidden outcomes: {sorted(forbidden)}"
            )
        if row.get("schema") != RELATIVE_WHERE_SCHEMA:
            raise ValueError("relative-where prediction schema changed")
        key = (str(row.get("state_id")), str(row.get("replicate_id")))
        outcome = outcomes.get(key)
        variants = row.get("variants")
        if (
            outcome is None
            or key in parsed
            or str(row.get("source_id")) != outcome.source_id
            or str(row.get("image_id")) != outcome.image_id
            or int(row.get("outer_fold", -1)) not in range(5)
            or not isinstance(variants, Mapping)
            or set(variants) != set(RELATIVE_WHERE_VARIANTS)
        ):
            raise ValueError("relative-where prediction identity changed")
        parsed_variants: dict[str, Any] = {}
        for name in RELATIVE_WHERE_VARIANTS:
            value = variants[name]
            if not isinstance(value, Mapping):
                raise ValueError("relative-where variant prediction is invalid")
            scores = value.get("action_scores")
            probabilities = value.get("action_probabilities")
            if (
                not isinstance(scores, list)
                or not isinstance(probabilities, list)
                or len(scores) != len(DECAR_ACTION_IDS)
                or len(probabilities) != len(DECAR_ACTION_IDS)
            ):
                raise ValueError("relative-where action vector shape changed")
            finite_scores = [_finite(item, "relative score") for item in scores]
            finite_probabilities = [
                _finite(item, "relative probability") for item in probabilities
            ]
            if any(item < 0.0 for item in finite_probabilities) or not math.isclose(
                sum(finite_probabilities), 1.0, rel_tol=0.0, abs_tol=1e-5
            ):
                raise ValueError("relative-where action probabilities are invalid")
            selected_index = finite_scores.index(max(finite_scores))
            ordered = sorted(finite_scores, reverse=True)
            expected_margin = ordered[0] - ordered[1]
            selected_action = str(value.get("selected_action_id"))
            margin = _finite(value.get("predicted_margin"), "relative margin")
            if (
                selected_action != DECAR_ACTION_IDS[selected_index]
                or margin < 0.0
                or not math.isclose(margin, expected_margin, rel_tol=1e-6, abs_tol=1e-6)
            ):
                raise ValueError("relative-where selected action or margin changed")
            parsed_variants[name] = {
                "selected_action_id": selected_action,
                "action_scores": finite_scores,
                "action_probabilities": finite_probabilities,
                "predicted_margin": margin,
            }
        parsed[key] = {
            "outer_fold": int(row["outer_fold"]),
            "variants": parsed_variants,
        }
    if set(parsed) != set(outcomes):
        raise ValueError("relative-where prediction join is incomplete")
    return parsed


def privileged_teacher_actions(
    nll_rows: Sequence[Mapping[str, Any]],
    outcomes: Mapping[DecisionKey, Any],
) -> dict[DecisionKey, str]:
    indexed: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in nll_rows:
        if row.get("schema") != "visual_action_answer_nll_v1":
            raise ValueError("relative-where teacher NLL schema changed")
        indexed_key = (
            str(row.get("state_id")),
            str(row.get("replicate_id")),
            str(row.get("action_id")),
        )
        if indexed_key in indexed:
            raise ValueError("relative-where teacher NLL row is duplicated")
        _finite(row.get("answer_mean_nll"), "teacher NLL")
        indexed[indexed_key] = row
    result: dict[DecisionKey, str] = {}
    expected: set[tuple[str, str, str]] = set()
    for decision_key, outcome in outcomes.items():
        action_ids = (outcome.baseline.action_id,) + tuple(
            crop.action_id for crop in outcome.crops
        )
        for action_id in action_ids:
            nll_key = (decision_key[0], decision_key[1], action_id)
            candidate_row = indexed.get(nll_key)
            if (
                candidate_row is None
                or str(candidate_row.get("source_id")) != outcome.source_id
                or str(candidate_row.get("image_id")) != outcome.image_id
            ):
                raise ValueError("relative-where teacher NLL identity changed")
            expected.add(nll_key)
        result[decision_key] = min(
            (crop.action_id for crop in outcome.crops),
            key=lambda action_id: (
                float(
                    indexed[(decision_key[0], decision_key[1], action_id)][
                        "answer_mean_nll"
                    ]
                ),
                action_id,
            ),
        )
    if set(indexed) != expected:
        raise ValueError("relative-where teacher NLL coverage is not exact")
    return result


def _percentile_interval(values: Any) -> dict[str, float]:
    import numpy as np  # type: ignore[import-not-found]

    return {
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
    }


def _teacher_agreement(
    *,
    actions: Mapping[DecisionKey, str] | None,
    teacher_actions: Mapping[DecisionKey, str],
    called_keys: set[DecisionKey],
    outcomes: Mapping[DecisionKey, Any],
    sources: Sequence[str],
    bootstrap_indices: Any,
    random_action: bool = False,
) -> dict[str, Any]:
    import numpy as np  # type: ignore[import-not-found]

    calls_by_source: dict[str, float] = defaultdict(float)
    agreements_by_source: dict[str, float] = defaultdict(float)
    decisions_by_source: dict[str, int] = defaultdict(int)
    raw_calls = 0.0
    raw_agreements = 0.0
    for key, outcome in outcomes.items():
        source = outcome.source_id
        decisions_by_source[source] += 1
        if key not in called_keys:
            continue
        raw_calls += 1.0
        calls_by_source[source] += 1.0
        agreement = (
            1.0 / len(DECAR_ACTION_IDS)
            if random_action
            else float(actions is not None and actions[key] == teacher_actions[key])
        )
        raw_agreements += agreement
        agreements_by_source[source] += agreement
    source_calls = np.asarray(
        [calls_by_source[source] / decisions_by_source[source] for source in sources],
        dtype=np.float64,
    )
    source_agreements = np.asarray(
        [
            agreements_by_source[source] / decisions_by_source[source]
            for source in sources
        ],
        dtype=np.float64,
    )
    source_call_point = float(source_calls.mean())
    source_agreement_point = float(source_agreements.mean())
    point = (
        source_agreement_point / source_call_point if source_call_point > 0.0 else None
    )
    draws: list[Any] = []
    for start in range(0, bootstrap_indices.shape[0], 256):
        sampled = bootstrap_indices[start : start + 256]
        call_draw = source_calls[sampled].mean(axis=1)
        agreement_draw = source_agreements[sampled].mean(axis=1)
        valid = call_draw > 0.0
        draws.append(agreement_draw[valid] / call_draw[valid])
    values = np.concatenate(draws) if draws else np.asarray([], dtype=np.float64)
    if values.size:
        agreement_interval: dict[str, Any] = _percentile_interval(values)
    else:
        agreement_interval = {"ci_low": None, "ci_high": None}
    return {
        "question_balanced": raw_agreements / raw_calls if raw_calls else None,
        "source_balanced": point,
        "source_bootstrap": {
            "point_estimate": point,
            "valid_resamples": int(values.size),
            **agreement_interval,
        },
    }


def _oracle_gap_closure(
    primary: Mapping[str, Any],
    old: Mapping[str, Any],
    oracle: Mapping[str, Any],
    sources: Sequence[str],
    bootstrap_indices: Any,
) -> dict[str, Any]:
    import numpy as np  # type: ignore[import-not-found]

    def utility_array(aggregate: Mapping[str, Any]) -> Any:
        source_values = aggregate["source_values"]
        return np.asarray(
            [float(source_values[source]["utility"]) for source in sources],
            dtype=np.float64,
        )

    primary_values = utility_array(primary)
    old_values = utility_array(old)
    oracle_values = utility_array(oracle)
    numerator = float(primary_values.mean() - old_values.mean())
    denominator = float(oracle_values.mean() - old_values.mean())
    point = numerator / denominator if denominator > 0.0 else None
    draws: list[Any] = []
    for start in range(0, bootstrap_indices.shape[0], 256):
        sampled = bootstrap_indices[start : start + 256]
        primary_draw = primary_values[sampled].mean(axis=1)
        old_draw = old_values[sampled].mean(axis=1)
        oracle_draw = oracle_values[sampled].mean(axis=1)
        denominator_draw = oracle_draw - old_draw
        valid = denominator_draw > 0.0
        draws.append((primary_draw[valid] - old_draw[valid]) / denominator_draw[valid])
    values = np.concatenate(draws) if draws else np.asarray([], dtype=np.float64)
    if values.size:
        closure_interval: dict[str, Any] = _percentile_interval(values)
    else:
        closure_interval = {"ci_low": None, "ci_high": None}
    return {
        "point_estimate": point,
        "valid_resamples": int(values.size),
        **closure_interval,
    }


def evaluate_relative_where_oof(
    records: Sequence[ActionRecord],
    relative_prediction_rows: Sequence[Mapping[str, Any]],
    decar_prediction_rows: Sequence[Mapping[str, Any]],
    nll_rows: Sequence[Mapping[str, Any]],
    hybrid_evaluation: Mapping[str, Any],
    oracle_evaluation: Mapping[str, Any],
    *,
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
    relative_predictions = parse_relative_where_predictions(
        relative_prediction_rows, outcomes
    )
    decar_predictions = parse_decar_predictions(decar_prediction_rows, outcomes)
    teacher_actions = privileged_teacher_actions(nll_rows, outcomes)
    keys = sorted(outcomes)
    sources = sorted({outcome.source_id for outcome in outcomes.values()})
    hybrid_points = hybrid_evaluation.get("operating_points")
    oracle_points = oracle_evaluation.get("operating_points")
    if (
        hybrid_evaluation.get("schema") != DECAR_HYBRID_EVALUATION_SCHEMA
        or hybrid_evaluation.get("decision") != "hybrid_train_not_supported"
        or hybrid_evaluation.get("selected_operating_point") is not None
        or hybrid_evaluation.get("validation_or_test_inputs_used") is not False
        or oracle_evaluation.get("schema") != DECAR_ORACLE_WHERE_EVALUATION_SCHEMA
        or oracle_evaluation.get("decision") != "where_bottleneck_supported"
        or oracle_evaluation.get("validation_or_test_inputs_used") is not False
        or tuple(hybrid_evaluation.get("registered_call_rates", ())) != DECAR_CALL_RATES
        or tuple(oracle_evaluation.get("registered_call_rates", ())) != DECAR_CALL_RATES
        or not isinstance(hybrid_points, list)
        or not isinstance(oracle_points, list)
        or len(hybrid_points) != len(DECAR_CALL_RATES)
        or len(oracle_points) != len(DECAR_CALL_RATES)
    ):
        raise ValueError("relative-where frozen evaluation dependency changed")
    if tuple(bootstrap_indices.shape) != (expected_bootstrap_resamples, len(sources)):
        raise ValueError("relative-where bootstrap shape changed")
    if str(bootstrap_indices.dtype) != "int32":
        raise ValueError("relative-where bootstrap dtype changed")
    if any(
        relative_predictions[key]["outer_fold"] != decar_predictions[key].outer_fold
        for key in keys
    ):
        raise ValueError("relative-where outer fold differs from frozen DECAR")

    relative_actions = {
        name: {
            key: str(relative_predictions[key]["variants"][name]["selected_action_id"])
            for key in keys
        }
        for name in RELATIVE_WHERE_VARIANTS
    }
    old_actions = {
        "old_decar_where": {
            key: decar_predictions[key].variants["decar"].action_id for key in keys
        },
        "old_task_value_where": {
            key: decar_predictions[key].variants["task_value_only"].action_id
            for key in keys
        },
    }
    entropy_scores = {key: outcomes[key].baseline.entropy_before for key in keys}
    policy_values: dict[str, dict[DecisionKey, dict[str, float]]] = {}
    called_by_point: dict[str, set[DecisionKey]] = {}
    operating: list[dict[str, Any]] = []
    answer_now = _policy_metrics(outcomes, called_keys=set())
    for hybrid_point, oracle_point, rate in zip(
        hybrid_points, oracle_points, DECAR_CALL_RATES, strict=True
    ):
        if not isinstance(hybrid_point, Mapping) or not isinstance(
            oracle_point, Mapping
        ):
            raise ValueError("relative-where frozen operating point is invalid")
        point_name = f"rate-{rate:.3f}"
        actual_calls = int(hybrid_point.get("actual_calls", -1))
        hybrid_selection = hybrid_point.get("selection_audits")
        if (
            hybrid_point.get("name") != point_name
            or oracle_point.get("name") != point_name
            or oracle_point.get("actual_calls") != actual_calls
            or not isinstance(hybrid_selection, Mapping)
        ):
            raise ValueError("relative-where operating-point family changed")
        entropy_calls, entropy_audit = _complete_tie_exact_match(
            entropy_scores, target_calls=actual_calls
        )
        if entropy_audit != hybrid_selection.get("entropy_one_crop"):
            raise ValueError("relative-where entropy identity audit failed")
        called_by_point[point_name] = entropy_calls
        point_values = {
            **{
                name: _policy_metrics(
                    outcomes,
                    called_keys=entropy_calls,
                    action_by_key=relative_actions[name],
                )
                for name in RELATIVE_WHERE_VARIANTS
            },
            "old_decar_where": _policy_metrics(
                outcomes,
                called_keys=entropy_calls,
                action_by_key=old_actions["old_decar_where"],
            ),
            "old_task_value_where": _policy_metrics(
                outcomes,
                called_keys=entropy_calls,
                action_by_key=old_actions["old_task_value_where"],
            ),
            "entropy_random": _policy_metrics(
                outcomes, called_keys=entropy_calls, random_action=True
            ),
            "entropy_fixed_ug_grid_00": _policy_metrics(
                outcomes,
                called_keys=entropy_calls,
                action_by_key={key: "ug-grid-00" for key in keys},
            ),
            "answer_now": answer_now,
            "privileged_teacher_nll_where": _policy_metrics(
                outcomes,
                called_keys=entropy_calls,
                action_by_key=teacher_actions,
            ),
            "task_oracle_where": _policy_metrics(
                outcomes, called_keys=entropy_calls, task_action=True
            ),
        }
        for name, values in point_values.items():
            policy_values[f"{point_name}/{name}"] = values
        operating.append(
            {
                "name": point_name,
                "nominal_question_call_rate": rate,
                "actual_calls": actual_calls,
                "selection_audit": entropy_audit,
            }
        )

    aggregates = {
        name: _aggregate_policy(values, outcomes, sources)
        for name, values in policy_values.items()
    }
    bootstrap, paired_differences = _bootstrap_all_policies(
        aggregates, sources, bootstrap_indices
    )
    public_aggregates = {
        name: {key: value for key, value in aggregate.items() if key != "source_values"}
        for name, aggregate in aggregates.items()
    }
    policy_names = (
        *RELATIVE_WHERE_VARIANTS,
        "old_decar_where",
        "old_task_value_where",
        "entropy_random",
        "entropy_fixed_ug_grid_00",
        "answer_now",
        "privileged_teacher_nll_where",
        "task_oracle_where",
    )
    deployable_comparators = (
        "absolute_teacher_entropy",
        "relative_teacher_uniform",
        "relative_task_entropy",
        "old_decar_where",
        "old_task_value_where",
        "entropy_random",
        "entropy_fixed_ug_grid_00",
        "answer_now",
    )
    points: list[dict[str, Any]] = []
    for registered, hybrid_point, oracle_point in zip(
        operating, hybrid_points, oracle_points, strict=True
    ):
        point_name = str(registered["name"])
        policies = {
            name: public_aggregates[f"{point_name}/{name}"] for name in policy_names
        }
        policy_bootstrap = {
            name: bootstrap["policies"][f"{point_name}/{name}"] for name in policy_names
        }
        frozen_matches = (
            policies["old_decar_where"]
            == hybrid_point["policies"]["entropy_when_decar_where"]
            and policies["old_task_value_where"]
            == hybrid_point["policies"]["entropy_when_task_value_where"]
            and policies["entropy_random"] == hybrid_point["policies"]["entropy_random"]
            and policies["entropy_fixed_ug_grid_00"]
            == hybrid_point["policies"]["entropy_fixed_ug_grid_00"]
            and policies["answer_now"] == hybrid_point["policies"]["answer_now"]
            and policies["task_oracle_where"]
            == oracle_point["policies"]["entropy_when_task_oracle_where"]
        )
        if not frozen_matches:
            raise ValueError("relative-where frozen comparator aggregate changed")
        entropy_calls = called_by_point[point_name]
        action_maps: dict[str, Mapping[DecisionKey, str] | None] = {
            **relative_actions,
            **old_actions,
            "entropy_random": None,
            "entropy_fixed_ug_grid_00": {key: "ug-grid-00" for key in keys},
            "answer_now": None,
            "privileged_teacher_nll_where": teacher_actions,
            "task_oracle_where": {
                key: outcomes[key].task_action.action_id for key in keys
            },
        }
        teacher_agreement: dict[str, Any] = {}
        for name in policy_names:
            called = set() if name == "answer_now" else entropy_calls
            agreement = _teacher_agreement(
                actions=action_maps[name],
                teacher_actions=teacher_actions,
                called_keys=called,
                outcomes=outcomes,
                sources=sources,
                bootstrap_indices=bootstrap_indices,
                random_action=name == "entropy_random",
            )
            teacher_agreement[name] = agreement
            policies[name]["teacher_agreement"] = {
                "question_balanced": agreement["question_balanced"],
                "source_balanced": agreement["source_balanced"],
            }
            policy_bootstrap[name]["teacher_agreement"] = agreement["source_bootstrap"]
        closure = _oracle_gap_closure(
            aggregates[f"{point_name}/{RELATIVE_WHERE_PRIMARY}"],
            aggregates[f"{point_name}/old_decar_where"],
            aggregates[f"{point_name}/task_oracle_where"],
            sources,
            bootstrap_indices,
        )
        primary = policies[RELATIVE_WHERE_PRIMARY]["source_balanced"]
        primary_interval = policy_bootstrap[RELATIVE_WHERE_PRIMARY]["additive"][
            "utility"
        ]
        differences = paired_differences[point_name]
        rules = {
            "minimum_calls_and_sources": (
                float(policies[RELATIVE_WHERE_PRIMARY]["raw_calls"]) >= 100.0
                and int(policies[RELATIVE_WHERE_PRIMARY]["distinct_called_sources"])
                >= 50
            ),
            "primary_source_utility_ci_low_strictly_positive": float(
                primary_interval["ci_low"]
            )
            > 0.0,
            "paired_above_old_decar_ci_low_strictly_positive": float(
                differences["old_decar_where"]["ci_low"]
            )
            > 0.0,
            "paired_above_old_task_value_ci_low_strictly_positive": float(
                differences["old_task_value_where"]["ci_low"]
            )
            > 0.0,
            "strictly_above_all_deployable_comparators": all(
                float(primary["utility"])
                > float(policies[name]["source_balanced"]["utility"])
                for name in deployable_comparators
            ),
            "harm_no_greater_than_one_crop_entropy_baselines": all(
                float(primary[metric])
                <= float(policies[name]["source_balanced"][metric]) + 1e-15
                for metric in ("induced_harm", "negative_utility_call")
                for name in ("entropy_random", "entropy_fixed_ug_grid_00")
            ),
            "oracle_gap_closure_at_least_half": closure["point_estimate"] is not None
            and float(closure["point_estimate"]) >= 0.5,
            "all_audits_passed": bool(
                registered["selection_audit"]["matched_call_count"] and frozen_matches
            ),
        }
        points.append(
            {
                **registered,
                "policies": policies,
                "source_bootstrap": policy_bootstrap,
                "paired_source_utility_differences": differences,
                "primary_oracle_gap_closure": closure,
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
                    point["policies"][RELATIVE_WHERE_PRIMARY]["source_balanced"][
                        "utility"
                    ]
                ),
                float(
                    point["policies"][RELATIVE_WHERE_PRIMARY]["source_balanced"][
                        "induced_harm"
                    ]
                ),
                float(point["nominal_question_call_rate"]),
            ),
        )
        if qualified
        else None
    )
    return {
        "schema": RELATIVE_WHERE_EVALUATION_SCHEMA,
        "scientific_status": "frozen official-train source-OOF relative-where evaluation",
        "population": {
            "decisions": len(outcomes),
            "sources": len(sources),
            "images": len({outcome.image_id for outcome in outcomes.values()}),
        },
        "primary": RELATIVE_WHERE_PRIMARY,
        "variants": list(RELATIVE_WHERE_VARIANTS),
        "lambda_cost": DECAR_LAMBDA_COST,
        "registered_call_rates": list(DECAR_CALL_RATES),
        "operating_points": points,
        "bootstrap": bootstrap["metadata"],
        "decision": (
            "relative_where_train_supported"
            if selected is not None
            else "relative_where_train_not_supported"
        ),
        "selected_operating_point": (
            None
            if selected is None
            else {
                "name": selected["name"],
                "nominal_question_call_rate": selected["nominal_question_call_rate"],
                "actual_calls": selected["actual_calls"],
                "source_balanced_utility": selected["policies"][RELATIVE_WHERE_PRIMARY][
                    "source_balanced"
                ]["utility"],
                "source_balanced_induced_harm": selected["policies"][
                    RELATIVE_WHERE_PRIMARY
                ]["source_balanced"]["induced_harm"],
                "oracle_gap_closure": selected["primary_oracle_gap_closure"][
                    "point_estimate"
                ],
            }
        ),
        "relative_prediction_outcomes_included": False,
        "privileged_teacher_used_only_in_evaluation": True,
        "validation_or_test_inputs_used": False,
    }
