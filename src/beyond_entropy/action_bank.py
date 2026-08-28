from __future__ import annotations

from statistics import mean
from typing import Any, Sequence

from .dataset import group_by_decision
from .metrics import bootstrap_policy_evaluation, evaluate_policy
from .policies import (
    AnswerNowPolicy,
    EntropySearchPolicy,
    ExpectedRandomZoomPolicy,
    FixedCenterZoomPolicy,
    OracleVOIPolicy,
    Policy,
)
from .schema import ActionRecord


def summarize_action_bank(
    records: Sequence[ActionRecord],
    *,
    lambda_cost: float = 0.05,
    bootstrap_resamples: int = 5000,
    bootstrap_seed: int = 0,
    cluster_by: str = "source_id",
) -> dict[str, Any]:
    """Summarize counterfactual headroom without fitting a learned policy."""

    if lambda_cost < 0.0:
        raise ValueError("lambda_cost must be non-negative")
    grouped = group_by_decision(records)
    if not grouped:
        raise ValueError("action bank must be non-empty")
    baselines = []
    zooms_by_key = {}
    for key, siblings in grouped.items():
        answers = [record for record in siblings if record.action_type == "ANSWER"]
        zooms = [record for record in siblings if record.action_type == "ZOOM"]
        if len(answers) != 1 or not zooms:
            raise ValueError(f"decision {key!r} has incomplete sibling actions")
        baselines.append(answers[0])
        zooms_by_key[key] = zooms
    helpful_keys = [
        key
        for key, zooms in zooms_by_key.items()
        if any(record.delta_success > 0.0 for record in zooms)
    ]
    harmful_keys = [
        key
        for key, zooms in zooms_by_key.items()
        if any(record.delta_success < 0.0 for record in zooms)
    ]
    wrong_keys = [
        key
        for key, siblings in grouped.items()
        if next(record for record in siblings if record.action_type == "ANSWER").correct_before
        < 0.5
    ]
    wrong_key_set = set(wrong_keys)
    correct_keys = [key for key in grouped if key not in wrong_key_set]
    policies: dict[str, Policy] = {
        "answer_now": AnswerNowPolicy(),
        "random_one_crop": ExpectedRandomZoomPolicy(),
        "fixed_center_crop": FixedCenterZoomPolicy(),
        "exhaustive_entropy_search": EntropySearchPolicy(),
        "oracle_voi": OracleVOIPolicy(lambda_cost=lambda_cost),
    }
    evaluated = {}
    for name, policy in policies.items():
        result: dict[str, Any] = dict(
            evaluate_policy(records, policy, lambda_cost=lambda_cost)
        )
        result["bootstrap"] = bootstrap_policy_evaluation(
            records,
            policy,
            lambda_cost=lambda_cost,
            n_resamples=bootstrap_resamples,
            seed=bootstrap_seed,
            cluster_by=cluster_by,  # type: ignore[arg-type]
        )
        evaluated[name] = result
    return {
        "lambda_cost": lambda_cost,
        "decisions": len(grouped),
        "states": len({record.state_id for record in records}),
        "sources": len({record.source_id for record in records}),
        "images": len({record.image_id for record in records}),
        "candidate_counts": sorted({len(zooms) for zooms in zooms_by_key.values()}),
        "baseline_mean_score": mean(record.correct_before for record in baselines),
        "baseline_wrong_rate": len(wrong_keys) / len(grouped),
        "helpful_state_rate": len(helpful_keys) / len(grouped),
        "harmful_state_rate": len(harmful_keys) / len(grouped),
        "random_crop_rescue_rate_within_helpful_states": (
            mean(
                mean(record.delta_success > 0.0 for record in zooms_by_key[key])
                for key in helpful_keys
            )
            if helpful_keys
            else None
        ),
        "random_crop_harm_rate_within_harmful_states": (
            mean(
                mean(record.delta_success < 0.0 for record in zooms_by_key[key])
                for key in harmful_keys
            )
            if harmful_keys
            else None
        ),
        "rescue_action_rate_on_wrong_baselines": (
            mean(
                mean(record.delta_success > 0.0 for record in zooms_by_key[key])
                for key in wrong_keys
            )
            if wrong_keys
            else None
        ),
        "harm_action_rate_on_correct_baselines": (
            mean(
                mean(record.delta_success < 0.0 for record in zooms_by_key[key])
                for key in correct_keys
            )
            if correct_keys
            else None
        ),
        "policies": evaluated,
    }
