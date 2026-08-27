from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import mean

from beyond_entropy.dataset import group_by_decision, read_jsonl
from beyond_entropy.metrics import (
    bootstrap_policy_evaluation,
    evaluate_policy,
    paired_bootstrap_policy_difference,
)
from beyond_entropy.rescue_gate import (
    PrecomputedActionGatePolicy,
    PrecomputedRescueGatePolicy,
    _grouped_crossfit_records,
)
from beyond_entropy.transfer_gate import (
    score_frozen_factorized_context_model,
    select_frozen_context_quadrant_actions,
)


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compose OOF stopping and crop models")
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--factorized-models", type=Path, required=True)
    parser.add_argument("--action-models", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    args = parser.parse_args()

    records = read_jsonl(args.rollouts)
    grouped = group_by_decision(records)
    factorized = read_json(args.factorized_models)
    action = read_json(args.action_models)
    if (
        factorized.get("model_type") != "nested_oof_factorized_error_and_rescue_gate"
        or action.get("model_type") != "nested_oof_two_stage_state_and_action_gate"
        or action.get("action_feature_mode") != "context-quadrant"
    ):
        raise ValueError("input model types do not match the composed OOF protocol")
    factorized_folds = factorized["fold_models"]
    action_folds = action["fold_models"]
    if not isinstance(factorized_folds, list) or not isinstance(action_folds, list):
        raise ValueError("OOF model files lack fold models")
    if len(factorized_folds) != 5 or len(action_folds) != 5:
        raise ValueError("composed OOF evaluation requires five folds")

    pooled_calls = {}
    pooled_actions = {}
    for fold_index, (_, outer_test) in enumerate(
        _grouped_crossfit_records(
            records,
            split_group="image_id",
            n_folds=5,
            seed=17,
        )
    ):
        factorized_fold = factorized_folds[fold_index]
        action_fold = action_folds[fold_index]
        if not isinstance(factorized_fold, dict) or not isinstance(action_fold, dict):
            raise ValueError("invalid serialized OOF fold model")
        factorized_payload = {
            "model_type": "factorized_context_cross_benchmark_transfer",
            **factorized_fold,
        }
        action_payload = {
            "model_type": "context_quadrant_action_ranker_transfer",
            **action_fold,
        }
        scores = score_frozen_factorized_context_model(
            factorized_payload,
            outer_test,
        )
        top_actions = select_frozen_context_quadrant_actions(
            action_payload,
            outer_test,
        )
        threshold = float(factorized_fold["threshold"])
        for key in sorted(group_by_decision(outer_test)):
            call = scores[key] >= threshold
            if key in pooled_calls:
                raise RuntimeError(f"duplicate OOF decision: {key!r}")
            pooled_calls[key] = float(call)
            pooled_actions[key] = top_actions[key] if call else None
    if set(pooled_calls) != set(grouped):
        raise RuntimeError("composed OOF predictions do not cover all decisions")

    random_policy = PrecomputedRescueGatePolicy(
        pooled_calls,
        threshold=0.5,
        name="factorized_state_uniform_random_expectation",
    )
    composed_policy = PrecomputedActionGatePolicy(
        pooled_actions,
        name="factorized_state_context_quadrant_action",
    )
    policies = {}
    for name, policy in (
        ("factorized_uniform_random", random_policy),
        ("factorized_context_quadrant", composed_policy),
    ):
        result: dict[str, object] = dict(
            evaluate_policy(records, policy, lambda_cost=0.05)
        )
        result["bootstrap"] = bootstrap_policy_evaluation(
            records,
            policy,
            lambda_cost=0.05,
            n_resamples=args.bootstrap_resamples,
            seed=0,
        )
        policies[name] = result
    paired = paired_bootstrap_policy_difference(
        records,
        random_policy,
        records,
        composed_policy,
        lambda_cost=0.05,
        n_resamples=args.bootstrap_resamples,
        seed=0,
    )
    helpful_keys = [
        key
        for key, siblings in grouped.items()
        if any(
            record.action_type == "ZOOM" and record.delta_success > 0.0
            for record in siblings
        )
    ]
    selected_action_by_key = {
        key: action_id
        for key, action_id in pooled_actions.items()
        if action_id is not None
    }
    report: dict[str, object] = {
        "scientific_status": (
            "post-hoc exploratory composition of two leakage-safe OOF components; "
            "intervals do not account for method selection"
        ),
        "run": {
            "rollouts": str(args.rollouts.resolve()),
            "rollouts_sha256": hashlib.sha256(args.rollouts.read_bytes()).hexdigest(),
            "factorized_models": str(args.factorized_models.resolve()),
            "factorized_models_sha256": hashlib.sha256(
                args.factorized_models.read_bytes()
            ).hexdigest(),
            "action_models": str(args.action_models.resolve()),
            "action_models_sha256": hashlib.sha256(
                args.action_models.read_bytes()
            ).hexdigest(),
            "code_revision": os.environ.get("BE_CODE_REVISION"),
            "bootstrap_resamples": args.bootstrap_resamples,
        },
        "n_decisions": len(grouped),
        "policies": policies,
        "paired_context_quadrant_minus_random": paired,
        "diagnostics": {
            "helpful_states": len(helpful_keys),
            "top1_rescue_rate_within_called_helpful_states": mean(
                any(
                    record.action_id == selected_action_by_key.get(key)
                    and record.delta_success > 0.0
                    for record in grouped[key]
                )
                for key in helpful_keys
                if key in selected_action_by_key
            ),
        },
    }
    write_json(report, args.output_dir / "report.json")
    print(json.dumps({"output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
