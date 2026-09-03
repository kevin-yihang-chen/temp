from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Mapping, Sequence

from .predictability_audit import (
    AUDIT_BENCHMARKS,
    AUDIT_SEEDS,
    PREDICTOR_LEVELS,
    TARGET_FAMILIES,
    SplitIdentity,
    audit_split_disjointness,
    collapse_fixed_entropy_tool,
    matrix_completion_report,
)
from .predictability_baselines import (
    apply_strong_baselines,
    fit_strong_baselines,
    strong_baseline_report,
    trace_by_name,
    validate_fixed_tool_outcomes,
)
from .predictability_evaluation import (
    calls_at_threshold,
    paired_source_bootstrap_policy_difference,
    policy_curve,
    prediction_metrics,
)
from .predictability_modeling import (
    AuditExample,
    evaluate_frozen_audit_cell,
    fit_frozen_audit_cell,
)
from .predictability_post_action import (
    PostActionProbeExample,
    evaluate_frozen_post_action_probe,
    fit_frozen_post_action_probe,
)
from .schema import ActionRecord


STRONG_BASELINE_RANDOM_SEED = 20260903


@dataclass(frozen=True)
class BenchmarkAuditData:
    train: Sequence[AuditExample]
    validation: Sequence[AuditExample]
    test: Sequence[AuditExample]
    post_action_train: Sequence[PostActionProbeExample]
    post_action_validation: Sequence[PostActionProbeExample]
    post_action_test: Sequence[PostActionProbeExample]
    validation_siblings: Sequence[ActionRecord]
    test_siblings: Sequence[ActionRecord]

    def validate(self) -> dict[str, Any]:
        roles = {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }
        if any(not values for values in roles.values()):
            raise ValueError("all benchmark roles must be non-empty")
        post_roles = {
            "train": self.post_action_train,
            "validation": self.post_action_validation,
            "test": self.post_action_test,
        }
        if any(not values for values in post_roles.values()):
            raise ValueError("all post-action probe roles must be non-empty")
        for role in roles:
            pre = {item.outcome.decision_id: item for item in roles[role]}
            post = {item.outcome.decision_id: item for item in post_roles[role]}
            if len(pre) != len(roles[role]) or len(post) != len(post_roles[role]):
                raise ValueError("predictability role decision IDs must be unique")
            if set(pre) != set(post):
                raise ValueError("pre- and post-action role coverage differs")
            for decision_id, item in pre.items():
                counterpart = post[decision_id]
                if (
                    item.outcome != counterpart.outcome
                    or item.image_rgb_sha256 != counterpart.image_rgb_sha256
                ):
                    raise ValueError(
                        "pre- and post-action role labels or images differ"
                    )
        if not self.validation_siblings or not self.test_siblings:
            raise ValueError("strong baseline sibling records must be non-empty")
        validate_fixed_tool_outcomes(
            [item.outcome for item in self.validation],
            collapse_fixed_entropy_tool(self.validation_siblings),
        )
        validate_fixed_tool_outcomes(
            [item.outcome for item in self.test],
            collapse_fixed_entropy_tool(self.test_siblings),
        )
        identities: list[SplitIdentity] = []
        assignments = {}
        for role, examples in roles.items():
            for index, example in enumerate(examples):
                item_id = (
                    f"{role}:{index}:{example.outcome.state_id}:"
                    f"{example.outcome.replicate_id}"
                )
                identities.append(
                    SplitIdentity(
                        item_id=item_id,
                        source_id=example.outcome.source_id,
                        image_rgb_sha256=example.image_rgb_sha256,
                    )
                )
                assignments[item_id] = role
                for level in PREDICTOR_LEVELS:
                    example.inputs.feature_vector(level)
        return audit_split_disjointness(identities, assignments)  # type: ignore[arg-type]


def run_predictability_matrix(
    datasets: Mapping[str, BenchmarkAuditData],
    *,
    lambda_cost: float,
    bootstrap_resamples: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
    call_rates: Sequence[float],
    seeds: Sequence[int] = AUDIT_SEEDS,
    predictor_levels: Sequence[str] = PREDICTOR_LEVELS,
    target_families: Sequence[str] = TARGET_FAMILIES,
    strong_baseline_random_seed: int = STRONG_BASELINE_RANDOM_SEED,
    formal_claim_eligible: bool = False,
) -> dict[str, Any]:
    if set(datasets) != set(AUDIT_BENCHMARKS):
        raise ValueError("matrix runner requires exactly ChartQA, DocVQA, and HRBench")
    if not seeds or len(seeds) > 3 or len(set(seeds)) != len(seeds):
        raise ValueError("matrix runner requires one to three unique seeds")
    if set(predictor_levels) - set(PREDICTOR_LEVELS):
        raise ValueError("matrix runner received an unregistered predictor level")
    if set(target_families) - set(TARGET_FAMILIES):
        raise ValueError("matrix runner received an unregistered target family")
    if formal_claim_eligible and (
        tuple(seeds) != AUDIT_SEEDS
        or tuple(predictor_levels) != PREDICTOR_LEVELS
        or tuple(target_families) != TARGET_FAMILIES
        or strong_baseline_random_seed != STRONG_BASELINE_RANDOM_SEED
    ):
        raise ValueError(
            "formal matrix requires the complete frozen levels, targets, and seeds"
        )

    split_audits = {name: data.validate() for name, data in datasets.items()}
    baseline_freezes = {
        name: fit_strong_baselines(
            data.validation_siblings,
            lambda_cost=lambda_cost,
            random_gate_seed=strong_baseline_random_seed,
        )
        for name, data in datasets.items()
    }
    baseline_test_traces = {
        name: apply_strong_baselines(baseline_freezes[name], data.test_siblings)
        for name, data in datasets.items()
    }
    baseline_reports = {
        name: strong_baseline_report(baseline_freezes[name], baseline_test_traces[name])
        for name in datasets
    }
    post_action_reports: dict[str, Any] = {}
    for benchmark in AUDIT_BENCHMARKS:
        data = datasets[benchmark]
        test_outcomes = [item.outcome for item in data.post_action_test]
        test_traces = baseline_test_traces[benchmark]
        answer_now = trace_by_name(test_traces, "answer_now")
        strongest = trace_by_name(
            test_traces, baseline_freezes[benchmark].strongest_name
        )
        post_seed_reports: list[dict[str, Any]] = []
        for seed_index, seed in enumerate(seeds):
            probe = fit_frozen_post_action_probe(
                data.post_action_train,
                data.post_action_validation,
                seed=seed,
                lambda_cost=lambda_cost,
            )
            predictions, metrics = evaluate_frozen_post_action_probe(
                probe, data.post_action_test, lambda_cost=lambda_cost
            )
            candidate_calls = calls_at_threshold(predictions, probe.threshold)
            post_seed_reports.append(
                {
                    "seed": seed,
                    "model": "fixed_two_layer_mlp",
                    "target": "direct_gain",
                    "deployable": False,
                    "input_dimension": probe.input_dimension,
                    "validation": probe.validation_metrics,
                    "test_policy": metrics,
                    "test_prediction": prediction_metrics(
                        test_outcomes,
                        predictions,
                        lambda_cost=lambda_cost,
                    ),
                    "test_curve": policy_curve(
                        test_outcomes,
                        predictions,
                        lambda_cost=lambda_cost,
                        call_rates=call_rates,
                    ),
                    "paired_vs_answer_now": paired_source_bootstrap_policy_difference(
                        test_outcomes,
                        candidate_calls,
                        answer_now.outcomes,
                        answer_now.calls,
                        lambda_cost=lambda_cost,
                        resamples=bootstrap_resamples,
                        confidence_level=bootstrap_confidence,
                        seed=bootstrap_seed + seed_index,
                    ),
                    "paired_vs_strongest_baseline": paired_source_bootstrap_policy_difference(
                        test_outcomes,
                        candidate_calls,
                        strongest.outcomes,
                        strongest.calls,
                        lambda_cost=lambda_cost,
                        resamples=bootstrap_resamples,
                        confidence_level=bootstrap_confidence,
                        seed=bootstrap_seed + seed_index,
                    ),
                }
            )
        post_action_reports[benchmark] = {
            "schema": "predictability_post_action_probe_report_v1",
            "role": "diagnostic_only_never_deployable",
            "selection_role": "validation_only",
            "seeds": post_seed_reports,
            "mean_test_incremental_utility": mean(
                float(item["test_policy"]["incremental_utility"])
                for item in post_seed_reports
            ),
        }
    cell_reports: list[dict[str, Any]] = []
    completed_cells: list[tuple[str, str, str]] = []
    for benchmark in AUDIT_BENCHMARKS:
        data = datasets[benchmark]
        test_outcomes = [item.outcome for item in data.test]
        strongest_name = baseline_freezes[benchmark].strongest_name
        strongest = trace_by_name(baseline_test_traces[benchmark], strongest_name)
        for level in predictor_levels:
            for target in target_families:
                seed_reports: list[dict[str, Any]] = []
                for seed_index, seed in enumerate(seeds):
                    cell = fit_frozen_audit_cell(
                        data.train,
                        data.validation,
                        level=level,
                        target=target,
                        seed=seed,
                        lambda_cost=lambda_cost,
                    )
                    predictions, metrics = evaluate_frozen_audit_cell(
                        cell, data.test, lambda_cost=lambda_cost
                    )
                    candidate_calls = calls_at_threshold(predictions, cell.threshold)
                    seed_reports.append(
                        {
                            "seed": seed,
                            "selected_variant": cell.variant,
                            "validation": cell.validation_metrics,
                            "test_policy": metrics,
                            "test_prediction": prediction_metrics(
                                test_outcomes,
                                predictions,
                                lambda_cost=lambda_cost,
                            ),
                            "test_curve": policy_curve(
                                test_outcomes,
                                predictions,
                                lambda_cost=lambda_cost,
                                call_rates=call_rates,
                            ),
                            "paired_vs_strongest_baseline": paired_source_bootstrap_policy_difference(
                                test_outcomes,
                                candidate_calls,
                                strongest.outcomes,
                                strongest.calls,
                                lambda_cost=lambda_cost,
                                resamples=bootstrap_resamples,
                                confidence_level=bootstrap_confidence,
                                seed=bootstrap_seed + seed_index,
                            ),
                        }
                    )
                cell_reports.append(
                    {
                        "benchmark": benchmark,
                        "predictor_level": level,
                        "target": target,
                        "strongest_baseline": strongest_name,
                        "seeds": seed_reports,
                        "mean_test_incremental_utility": mean(
                            float(item["test_policy"]["incremental_utility"])
                            for item in seed_reports
                        ),
                    }
                )
                completed_cells.append((benchmark, level, target))
    return {
        "schema": "predictability_matrix_report_v1",
        "formal_claim_eligible": formal_claim_eligible,
        "lambda_cost": lambda_cost,
        "seeds": list(seeds),
        "split_audits": split_audits,
        "strong_baselines": baseline_reports,
        "post_action_probe": post_action_reports,
        "matrix": matrix_completion_report(completed_cells),
        "cells": cell_reports,
    }
