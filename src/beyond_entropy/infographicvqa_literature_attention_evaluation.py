from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .dataset import DecisionKey
from .infographicvqa_attention_where import assemble_attention_where_features
from .infographicvqa_attention_where_evaluation import (
    ATTENTION_WHERE_EVALUATION_SCHEMA,
    ATTENTION_WHERE_PRIMARY,
    _attention_deciles,
    _conditional_localization_metrics,
    _crop_nll_by_key,
)
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
from .infographicvqa_literature_attention_where import (
    LiteratureAttentionFeatures,
    assemble_literature_attention_where_features,
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

LITERATURE_ATTENTION_EVALUATION_SCHEMA = (
    "infographicvqa_literature_attention_where_evaluation_v1"
)
LITERATURE_ATTENTION_VARIANTS = (
    "vicrop_relative_bank",
    "laser_contrastive_all_head_bank",
)
LITERATURE_ATTENTION_COMPARATORS = (
    "entropy_fixed_ug_grid_00",
    "entropy_random",
    "old_decar_where",
    "relative_where",
    "raw_attention_where",
)
LITERATURE_ATTENTION_CI_LOW = 0.0125
LITERATURE_ATTENTION_CI_HIGH = 0.9875


def _bootstrap_mean_interval(
    aggregate: Mapping[str, Any],
    sources: Sequence[str],
    bootstrap_indices: Any,
    *,
    field: str = "utility",
) -> dict[str, float]:
    import numpy as np  # type: ignore[import-not-found]

    values = np.asarray(
        [aggregate["source_values"][source][field] for source in sources],
        dtype=np.float64,
    )
    indices = np.asarray(bootstrap_indices)
    draws = []
    for start in range(0, indices.shape[0], 256):
        sampled = indices[start : start + 256]
        draws.append(values[sampled].mean(axis=1))
    bootstrap = np.concatenate(draws)
    return {
        "point_estimate": float(values.mean()),
        "ci_low": float(np.quantile(bootstrap, LITERATURE_ATTENTION_CI_LOW)),
        "ci_high": float(np.quantile(bootstrap, LITERATURE_ATTENTION_CI_HIGH)),
        "confidence_level": 0.975,
    }


def _corrected_paired_differences(
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
        draws = []
        for start in range(0, indices.shape[0], 256):
            sampled = indices[start : start + 256]
            draws.append(difference[sampled].mean(axis=1))
        bootstrap = np.concatenate(draws)
        result[name] = {
            "point_estimate": float(difference.mean()),
            "ci_low": float(np.quantile(bootstrap, LITERATURE_ATTENTION_CI_LOW)),
            "ci_high": float(np.quantile(bootstrap, LITERATURE_ATTENTION_CI_HIGH)),
            "confidence_level": 0.975,
        }
    return result


def _corrected_point_intervals(
    *,
    aggregates: Mapping[str, Mapping[str, Any]],
    variants: Sequence[str],
    comparators: Sequence[str],
    sources: Sequence[str],
    bootstrap_indices: Any,
) -> dict[str, dict[str, Any]]:
    """Bootstrap all corrected contrasts with one batched source gather."""

    import numpy as np  # type: ignore[import-not-found]

    names = (*variants, *comparators)
    matrix = np.asarray(
        [
            [aggregates[name]["source_values"][source]["utility"] for source in sources]
            for name in names
        ],
        dtype=np.float64,
    )
    indices = np.asarray(bootstrap_indices)
    draw_chunks = []
    for start in range(0, indices.shape[0], 256):
        sampled = indices[start : start + 256]
        draw_chunks.append(matrix[:, sampled].mean(axis=2))
    draws = np.concatenate(draw_chunks, axis=1)
    positions = {name: index for index, name in enumerate(names)}
    result: dict[str, dict[str, Any]] = {}
    for variant in variants:
        variant_position = positions[variant]
        variant_draws = draws[variant_position]
        result[variant] = {
            "utility": {
                "point_estimate": float(matrix[variant_position].mean()),
                "ci_low": float(
                    np.quantile(variant_draws, LITERATURE_ATTENTION_CI_LOW)
                ),
                "ci_high": float(
                    np.quantile(variant_draws, LITERATURE_ATTENTION_CI_HIGH)
                ),
                "confidence_level": 0.975,
            },
            "differences": {},
        }
        for comparator in comparators:
            comparator_position = positions[comparator]
            point_difference = matrix[variant_position] - matrix[comparator_position]
            draw_difference = variant_draws - draws[comparator_position]
            result[variant]["differences"][comparator] = {
                "point_estimate": float(point_difference.mean()),
                "ci_low": float(
                    np.quantile(draw_difference, LITERATURE_ATTENTION_CI_LOW)
                ),
                "ci_high": float(
                    np.quantile(draw_difference, LITERATURE_ATTENTION_CI_HIGH)
                ),
                "confidence_level": 0.975,
            }
    return result


def _average_ranks(values: Any) -> Any:
    import numpy as np  # type: ignore[import-not-found]

    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.shape[0], dtype=np.float64)
    start = 0
    while start < order.shape[0]:
        stop = start + 1
        while stop < order.shape[0] and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def _spearman(left: Any, right: Any) -> float | None:
    import numpy as np  # type: ignore[import-not-found]

    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    if left_ranks.size < 2 or left_ranks.std() == 0.0 or right_ranks.std() == 0.0:
        return None
    value = float(np.corrcoef(left_ranks, right_ranks)[0, 1])
    return value if math.isfinite(value) else None


def _encore_diagnostics(
    *,
    features: LiteratureAttentionFeatures,
    keys: Sequence[DecisionKey],
    outcomes: Mapping[DecisionKey, Any],
    actions_by_variant: Mapping[str, Mapping[DecisionKey, str]],
    crop_nll: Mapping[DecisionKey, Mapping[str, float]],
) -> dict[str, Any]:
    import numpy as np  # type: ignore[import-not-found]

    baseline = np.asarray(
        [outcomes[key].baseline.correct_before for key in keys], dtype=np.float64
    )
    helpful = np.asarray(
        [
            float(max(crop.delta_success for crop in outcomes[key].crops) > 0.0)
            for key in keys
        ],
        dtype=np.float64,
    )
    regrets = {
        variant: np.asarray(
            [
                crop_nll[key][action_by_key[key]] - min(crop_nll[key].values())
                for key in keys
            ],
            dtype=np.float64,
        )
        for variant, action_by_key in actions_by_variant.items()
    }
    entropy = features.encore_early_entropy.numpy().astype(np.float64)
    result: dict[str, Any] = {
        "scientific_status": "descriptive_only_not_used_for_selection",
        "helpful_states": int(helpful.sum()),
        "layers": {},
    }
    for layer_position, layer_index in enumerate((0, 1)):
        values = entropy[:, layer_position]
        helpful_values = values[helpful == 1.0]
        nonhelpful_values = values[helpful == 0.0]
        result["layers"][str(layer_index)] = {
            "entropy_mean": float(values.mean()),
            "entropy_min": float(values.min()),
            "entropy_max": float(values.max()),
            "helpful_state_entropy_mean": (
                float(helpful_values.mean()) if helpful_values.size else None
            ),
            "nonhelpful_state_entropy_mean": (
                float(nonhelpful_values.mean()) if nonhelpful_values.size else None
            ),
            "spearman_baseline_anls": _spearman(values, baseline),
            "spearman_helpful_crop_exists": _spearman(values, helpful),
            "spearman_selected_nll_regret": {
                variant: _spearman(values, regret)
                for variant, regret in regrets.items()
            },
        }
    return result


def evaluate_literature_attention_where(
    records: Sequence[ActionRecord],
    literature_feature_payload: Mapping[str, Any],
    raw_attention_feature_payload: Mapping[str, Any],
    decar_prediction_rows: Sequence[Mapping[str, Any]],
    relative_prediction_rows: Sequence[Mapping[str, Any]],
    nll_rows: Sequence[Mapping[str, Any]],
    hybrid_evaluation: Mapping[str, Any],
    oracle_evaluation: Mapping[str, Any],
    relative_evaluation: Mapping[str, Any],
    raw_attention_evaluation: Mapping[str, Any],
    *,
    expected_literature_code_revision: str,
    expected_raw_attention_code_revision: str,
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
    literature, feature_audit = assemble_literature_attention_where_features(
        records,
        literature_feature_payload,
        expected_code_revision=expected_literature_code_revision,
        expected_model_revision=expected_model_revision,
        expected_source_features_sha256=expected_source_features_sha256,
        expected_rollouts_sha256=expected_rollouts_sha256,
    )
    raw_attention, raw_feature_audit = assemble_attention_where_features(
        records,
        raw_attention_feature_payload,
        expected_code_revision=expected_raw_attention_code_revision,
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
    literature_keys = list(
        zip(literature.state_ids, literature.replicate_ids, strict=True)
    )
    raw_keys = list(
        zip(raw_attention.state_ids, raw_attention.replicate_ids, strict=True)
    )
    if literature_keys != keys or raw_keys != keys:
        raise ValueError("literature attention feature ordering changed")
    sources = sorted({outcome.source_id for outcome in outcomes.values()})
    hybrid_points = hybrid_evaluation.get("operating_points")
    oracle_points = oracle_evaluation.get("operating_points")
    relative_points = relative_evaluation.get("operating_points")
    raw_points = raw_attention_evaluation.get("operating_points")
    if (
        hybrid_evaluation.get("schema") != DECAR_HYBRID_EVALUATION_SCHEMA
        or hybrid_evaluation.get("decision") != "hybrid_train_not_supported"
        or oracle_evaluation.get("schema") != DECAR_ORACLE_WHERE_EVALUATION_SCHEMA
        or oracle_evaluation.get("decision") != "where_bottleneck_supported"
        or relative_evaluation.get("schema") != RELATIVE_WHERE_EVALUATION_SCHEMA
        or relative_evaluation.get("decision") != "relative_where_train_not_supported"
        or raw_attention_evaluation.get("schema") != ATTENTION_WHERE_EVALUATION_SCHEMA
        or raw_attention_evaluation.get("decision")
        not in (
            "attention_where_train_supported",
            "attention_where_train_not_supported",
        )
        or any(
            evaluation.get("validation_or_test_inputs_used") is not False
            for evaluation in (
                hybrid_evaluation,
                oracle_evaluation,
                relative_evaluation,
                raw_attention_evaluation,
            )
        )
        or tuple(hybrid_evaluation.get("registered_call_rates", ())) != DECAR_CALL_RATES
        or tuple(oracle_evaluation.get("registered_call_rates", ())) != DECAR_CALL_RATES
        or tuple(relative_evaluation.get("registered_call_rates", ()))
        != DECAR_CALL_RATES
        or tuple(raw_attention_evaluation.get("registered_call_rates", ()))
        != DECAR_CALL_RATES
        or not all(
            isinstance(points, list) and len(points) == len(DECAR_CALL_RATES)
            for points in (hybrid_points, oracle_points, relative_points, raw_points)
        )
    ):
        raise ValueError("literature attention frozen evaluation dependency changed")
    if tuple(bootstrap_indices.shape) != (expected_bootstrap_resamples, len(sources)):
        raise ValueError("literature attention bootstrap shape changed")
    if str(bootstrap_indices.dtype) != "int32":
        raise ValueError("literature attention bootstrap dtype changed")
    assert isinstance(hybrid_points, list)
    assert isinstance(oracle_points, list)
    assert isinstance(relative_points, list)
    assert isinstance(raw_points, list)

    actions_by_variant = {
        "vicrop_relative_bank": {
            key: DECAR_ACTION_IDS[int(literature.vicrop_selected_indices[index])]
            for index, key in enumerate(keys)
        },
        "laser_contrastive_all_head_bank": {
            key: DECAR_ACTION_IDS[int(literature.laser_selected_indices[index])]
            for index, key in enumerate(keys)
        },
    }
    raw_actions = {
        key: DECAR_ACTION_IDS[int(raw_attention.selected_indices[index])]
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
    frozen_exact = True
    for hybrid_point, oracle_point, relative_point, raw_point, rate in zip(
        hybrid_points,
        oracle_points,
        relative_points,
        raw_points,
        DECAR_CALL_RATES,
        strict=True,
    ):
        point_name = f"rate-{rate:.3f}"
        actual_calls = int(hybrid_point.get("actual_calls", -1))
        hybrid_selection = hybrid_point.get("selection_audits")
        if (
            any(
                point.get("name") != point_name
                for point in (hybrid_point, oracle_point, relative_point, raw_point)
            )
            or any(
                point.get("actual_calls") != actual_calls
                for point in (oracle_point, relative_point, raw_point)
            )
            or not isinstance(hybrid_selection, Mapping)
        ):
            raise ValueError("literature attention operating-point family changed")
        called, selection_audit = _complete_tie_exact_match(
            entropy_scores, target_calls=actual_calls
        )
        if selection_audit != hybrid_selection.get("entropy_one_crop"):
            raise ValueError("literature attention entropy identity audit failed")
        called_by_point[point_name] = called
        values = {
            variant: _policy_metrics(
                outcomes, called_keys=called, action_by_key=actions_by_variant[variant]
            )
            for variant in LITERATURE_ATTENTION_VARIANTS
        }
        values.update(
            {
                "raw_attention_where": _policy_metrics(
                    outcomes, called_keys=called, action_by_key=raw_actions
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
        )
        for name, rows in values.items():
            policy_values[f"{point_name}/{name}"] = rows
        operating.append(
            {
                "name": point_name,
                "nominal_question_call_rate": rate,
                "actual_calls": actual_calls,
                "selection_audit": selection_audit,
                "raw_point": raw_point,
                "hybrid_point": hybrid_point,
                "oracle_point": oracle_point,
                "relative_point": relative_point,
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
    for registered in operating:
        point_name = str(registered["name"])
        hybrid_point = registered.pop("hybrid_point")
        oracle_point = registered.pop("oracle_point")
        relative_point = registered.pop("relative_point")
        raw_point = registered.pop("raw_point")
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
            "raw_attention_where": raw_point["policies"][ATTENTION_WHERE_PRIMARY],
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
        *LITERATURE_ATTENTION_VARIANTS,
        *LITERATURE_ATTENTION_COMPARATORS,
        "answer_now",
        "privileged_teacher_nll_where",
        "task_oracle_where",
    )
    points: list[dict[str, Any]] = []
    qualified: list[tuple[str, dict[str, Any]]] = []
    for registered in operating:
        point_name = str(registered["name"])
        policies = {name: public[f"{point_name}/{name}"] for name in policy_names}
        policy_bootstrap = {
            name: bootstrap["policies"][f"{point_name}/{name}"] for name in policy_names
        }
        point_aggregate_names = {
            name: aggregates[f"{point_name}/{name}"]
            for name in (
                *LITERATURE_ATTENTION_VARIANTS,
                *LITERATURE_ATTENTION_COMPARATORS,
            )
        }
        corrected = _corrected_point_intervals(
            aggregates=point_aggregate_names,
            variants=LITERATURE_ATTENTION_VARIANTS,
            comparators=LITERATURE_ATTENTION_COMPARATORS,
            sources=sources,
            bootstrap_indices=bootstrap_indices,
        )
        variant_results: dict[str, Any] = {}
        for variant in LITERATURE_ATTENTION_VARIANTS:
            primary_aggregate = aggregates[f"{point_name}/{variant}"]
            utility_interval = corrected[variant]["utility"]
            differences = corrected[variant]["differences"]
            closure = _oracle_gap_closure(
                primary_aggregate,
                aggregates[f"{point_name}/entropy_random"],
                aggregates[f"{point_name}/task_oracle_where"],
                sources,
                bootstrap_indices,
            )
            best_comparator = max(
                LITERATURE_ATTENTION_COMPARATORS,
                key=lambda name: float(policies[name]["source_balanced"]["utility"]),
            )
            primary = policies[variant]["source_balanced"]
            paired_lows = [
                float(differences[name]["ci_low"])
                for name in LITERATURE_ATTENTION_COMPARATORS
            ]
            rules = {
                "all_audits_passed": bool(
                    feature_audit["passed"]
                    and raw_feature_audit["passed"]
                    and registered["selection_audit"]["matched_call_count"]
                    and registered["frozen_comparators_exact_match"]
                    and frozen_exact
                ),
                "primary_source_utility_positive": float(primary["utility"]) > 0.0,
                "corrected_source_utility_ci_low_strictly_positive": float(
                    utility_interval["ci_low"]
                )
                > 0.0,
                "corrected_paired_noninferior_to_all_registered_comparators": all(
                    value >= 0.0 for value in paired_lows
                ),
                "corrected_paired_strictly_superior_to_at_least_one_comparator": any(
                    value > 0.0 for value in paired_lows
                ),
                "induced_harm_within_registered_tolerance": float(
                    primary["induced_harm"]
                )
                <= float(policies[best_comparator]["source_balanced"]["induced_harm"])
                + 0.00025,
                "random_to_oracle_gap_closure_at_least_quarter": closure[
                    "point_estimate"
                ]
                is not None
                and float(closure["point_estimate"]) >= 0.25,
            }
            called = called_by_point[point_name]
            variant_result = {
                "source_utility_corrected_interval": utility_interval,
                "corrected_paired_source_utility_differences": differences,
                "random_to_oracle_gap_closure": closure,
                "best_deployable_comparator": best_comparator,
                "localization_on_called_states": _conditional_localization_metrics(
                    action_by_key=actions_by_variant[variant],
                    selected_keys=called,
                    outcomes=outcomes,
                    teacher_actions=teacher_actions,
                    crop_nll=crop_nll,
                ),
                "source_concentration": _source_concentration(
                    policy_values[f"{point_name}/{variant}"], outcomes
                ),
                "qualification_rules": rules,
                "qualified": all(rules.values()),
            }
            variant_results[variant] = variant_result
            if variant_result["qualified"]:
                qualified.append(
                    (variant, {**registered, **variant_result, "policies": policies})
                )
        points.append(
            {
                **registered,
                "policies": policies,
                "source_bootstrap_95_percent_descriptive": policy_bootstrap,
                "variants": variant_results,
            }
        )
    selected = (
        min(
            qualified,
            key=lambda item: (
                -float(item[1]["source_utility_corrected_interval"]["ci_low"]),
                -float(item[1]["policies"][item[0]]["source_balanced"]["utility"]),
                float(item[1]["policies"][item[0]]["source_balanced"]["induced_harm"]),
                float(item[1]["nominal_question_call_rate"]),
                0 if item[0] == "vicrop_relative_bank" else 1,
            ),
        )
        if qualified
        else None
    )
    score_tensors = {
        "vicrop_relative_bank": literature.vicrop_scores,
        "laser_contrastive_all_head_bank": literature.laser_scores,
    }
    margin_tensors = {
        "vicrop_relative_bank": literature.vicrop_margins,
        "laser_contrastive_all_head_bank": literature.laser_margins,
    }
    localization = {}
    for variant in LITERATURE_ATTENTION_VARIANTS:
        max_scores = {
            key: float(score_tensors[variant][index].max())
            for index, key in enumerate(keys)
        }
        margins = {
            key: float(margin_tensors[variant][index]) for index, key in enumerate(keys)
        }
        localization[variant] = {
            "all_states": _conditional_localization_metrics(
                action_by_key=actions_by_variant[variant],
                selected_keys=set(keys),
                outcomes=outcomes,
                teacher_actions=teacher_actions,
                crop_nll=crop_nll,
            ),
            "max_score_deciles": _attention_deciles(
                field_by_key=max_scores,
                action_by_key=actions_by_variant[variant],
                outcomes=outcomes,
                teacher_actions=teacher_actions,
                crop_nll=crop_nll,
            ),
            "margin_deciles": _attention_deciles(
                field_by_key=margins,
                action_by_key=actions_by_variant[variant],
                outcomes=outcomes,
                teacher_actions=teacher_actions,
                crop_nll=crop_nll,
            ),
        }
    return {
        "schema": LITERATURE_ATTENTION_EVALUATION_SCHEMA,
        "scientific_status": (
            "frozen InfographicVQA official-train literature-attention where evaluation"
        ),
        "population": {
            "decisions": len(outcomes),
            "sources": len(sources),
            "images": len({outcome.image_id for outcome in outcomes.values()}),
        },
        "registered_variants": list(LITERATURE_ATTENTION_VARIANTS),
        "registered_comparators": list(LITERATURE_ATTENTION_COMPARATORS),
        "lambda_cost": DECAR_LAMBDA_COST,
        "registered_call_rates": list(DECAR_CALL_RATES),
        "frozen_comparator_float_tolerance": {
            "relative": FROZEN_COMPARATOR_FLOAT_REL_TOL,
            "absolute": FROZEN_COMPARATOR_FLOAT_ABS_TOL,
            "discrete_fields_exact": True,
        },
        "multiplicity_correction": {
            "method": "Bonferroni for two candidate variants",
            "central_confidence_level": 0.975,
            "quantiles": [LITERATURE_ATTENTION_CI_LOW, LITERATURE_ATTENTION_CI_HIGH],
        },
        "feature_audit": feature_audit,
        "raw_attention_feature_audit": raw_feature_audit,
        "localization": localization,
        "encore_early_entropy": _encore_diagnostics(
            features=literature,
            keys=keys,
            outcomes=outcomes,
            actions_by_variant=actions_by_variant,
            crop_nll=crop_nll,
        ),
        "operating_points": points,
        "bootstrap": bootstrap["metadata"],
        "decision": (
            "literature_attention_where_train_supported"
            if selected is not None
            else "literature_attention_where_train_not_supported"
        ),
        "selected_variant_and_operating_point": (
            None
            if selected is None
            else {
                "variant": selected[0],
                "name": selected[1]["name"],
                "nominal_question_call_rate": selected[1]["nominal_question_call_rate"],
                "actual_calls": selected[1]["actual_calls"],
                "source_balanced_utility": selected[1]["policies"][selected[0]][
                    "source_balanced"
                ]["utility"],
                "corrected_source_utility_ci_low": selected[1][
                    "source_utility_corrected_interval"
                ]["ci_low"],
            }
        ),
        "features_outcomes_included": False,
        "privileged_teacher_used_only_in_evaluation": True,
        "validation_or_test_inputs_used": False,
    }
