from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence

from .dataset import group_by_decision
from .metrics import paired_bootstrap_policy_difference
from .policies import (
    EntropySearchPolicy,
    ExpectedRandomZoomPolicy,
    FixedCenterZoomPolicy,
    OracleVOIPolicy,
    Policy,
)
from .schema import ActionRecord


def _answer_record(siblings: Sequence[ActionRecord]) -> ActionRecord:
    return next(record for record in siblings if record.action_type == "ANSWER")


def _candidate_count(records: Sequence[ActionRecord]) -> int:
    counts = {
        sum(record.action_type == "ZOOM" for record in siblings)
        for siblings in group_by_decision(records).values()
    }
    if len(counts) != 1:
        raise ValueError(f"candidate ablation requires a fixed candidate count, got {counts}")
    return next(iter(counts))


def _validate_matched_baselines(
    left_records: Sequence[ActionRecord],
    right_records: Sequence[ActionRecord],
) -> int:
    left = group_by_decision(left_records)
    right = group_by_decision(right_records)
    if set(left) != set(right):
        raise ValueError("candidate sets contain different decision keys")
    for decision_key in sorted(left):
        left_answer = _answer_record(left[decision_key])
        right_answer = _answer_record(right[decision_key])
        exact_fields = (
            "state_id",
            "image_id",
            "source_id",
            "question",
            "replicate_id",
            "generation_seed",
            "answer_before",
            "answer_after",
        )
        for field in exact_fields:
            if getattr(left_answer, field) != getattr(right_answer, field):
                raise ValueError(
                    f"paired decision {decision_key!r} differs in baseline field {field}"
                )
        if Path(left_answer.original_image).name != Path(right_answer.original_image).name:
            raise ValueError(
                f"paired decision {decision_key!r} differs in baseline image filename"
            )
        numeric_fields = (
            "entropy_before",
            "entropy_after",
            "correct_before",
            "correct_after",
        )
        for field in numeric_fields:
            if abs(float(getattr(left_answer, field)) - float(getattr(right_answer, field))) > 1e-12:
                raise ValueError(
                    f"paired decision {decision_key!r} differs in baseline field {field}"
                )
    return len(left)


def compare_candidate_sets(
    left_records: Sequence[ActionRecord],
    right_records: Sequence[ActionRecord],
    *,
    lambda_cost: float = 0.05,
    bootstrap_resamples: int = 2000,
    seed: int = 0,
    cluster_by: Literal["state_id", "image_id", "source_id"] = "state_id",
) -> dict[str, Any]:
    """Compare candidate sets with paired state-cluster resampling."""

    n_decisions = _validate_matched_baselines(left_records, right_records)
    policies: list[Policy] = [
        ExpectedRandomZoomPolicy(),
        FixedCenterZoomPolicy(),
        EntropySearchPolicy(),
        OracleVOIPolicy(lambda_cost),
    ]
    differences = [
        paired_bootstrap_policy_difference(
            left_records,
            policy,
            right_records,
            policy,
            lambda_cost=lambda_cost,
            n_resamples=bootstrap_resamples,
            seed=seed + index,
            cluster_by=cluster_by,
        )
        for index, policy in enumerate(policies)
    ]
    return {
        "scientific_status": (
            "matched candidate-count diagnostic; not a final benchmark claim"
        ),
        "comparison": "right_minus_left",
        "lambda_cost": lambda_cost,
        "bootstrap_resamples": bootstrap_resamples,
        "seed": seed,
        "resampling_unit": cluster_by,
        "n_decisions": n_decisions,
        "left_candidates_per_decision": _candidate_count(left_records),
        "right_candidates_per_decision": _candidate_count(right_records),
        "baseline_validation": (
            "matched decision keys and identical answer-now content, answers, correctness, and entropy"
        ),
        "policy_differences": differences,
    }


def build_candidate_ablation_markdown(report: dict[str, Any]) -> str:
    def interval(comparison: dict[str, Any], metric: str) -> str:
        result = comparison["metrics"][metric]
        return "{:.4f} [{:.4f}, {:.4f}]".format(
            result["estimate"],
            result["ci_low"],
            result["ci_high"],
        )

    lines = [
        "# Matched candidate-count ablation",
        "",
        "> Right minus left; confidence intervals resample matched state clusters.",
        "",
        "- Left candidates/decision: {}".format(report["left_candidates_per_decision"]),
        "- Right candidates/decision: {}".format(report["right_candidates_per_decision"]),
        "- Cost coefficient: {}".format(report["lambda_cost"]),
        "- Baseline validation: {}".format(report["baseline_validation"]),
        "",
        "| Policy | Accuracy-gain difference [95% CI] | Calls difference [95% CI] | Utility difference [95% CI] |",
        "|---|---:|---:|---:|",
    ]
    for comparison in report["policy_differences"]:
        lines.append(
            "| {} | {} | {} | {} |".format(
                comparison["right_policy"],
                interval(comparison, "accuracy_gain"),
                interval(comparison, "avg_tool_calls"),
                interval(comparison, "mean_policy_utility"),
            )
        )
    lines.extend(
        [
            "",
            "The expected-random policy uses sibling labels only for seed-free off-policy evaluation.",
            "Oracle VOI consumes counterfactual labels and is not deployable.",
            "",
        ]
    )
    return "\n".join(lines)
