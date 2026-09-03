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
    matrix_completion_report,
)
from .predictability_evaluation import (
    calls_at_threshold,
    paired_source_bootstrap_utility,
    policy_curve,
    prediction_metrics,
)
from .predictability_modeling import (
    AuditExample,
    evaluate_frozen_audit_cell,
    fit_frozen_audit_cell,
)


@dataclass(frozen=True)
class BenchmarkAuditData:
    train: Sequence[AuditExample]
    validation: Sequence[AuditExample]
    test: Sequence[AuditExample]
    strongest_baseline_name: str
    strongest_baseline_test_calls: Sequence[bool]

    def validate(self) -> dict[str, Any]:
        roles = {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }
        if any(not values for values in roles.values()):
            raise ValueError("all benchmark roles must be non-empty")
        if len(self.strongest_baseline_test_calls) != len(self.test):
            raise ValueError("strongest baseline call mask must align with test role")
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
    ):
        raise ValueError(
            "formal matrix requires the complete frozen levels, targets, and seeds"
        )

    split_audits = {name: data.validate() for name, data in datasets.items()}
    cell_reports: list[dict[str, Any]] = []
    completed_cells: list[tuple[str, str, str]] = []
    for benchmark in AUDIT_BENCHMARKS:
        data = datasets[benchmark]
        test_outcomes = [item.outcome for item in data.test]
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
                            "paired_vs_strongest_baseline": paired_source_bootstrap_utility(
                                test_outcomes,
                                candidate_calls,
                                data.strongest_baseline_test_calls,
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
                        "strongest_baseline": data.strongest_baseline_name,
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
        "matrix": matrix_completion_report(completed_cells),
        "cells": cell_reports,
    }
