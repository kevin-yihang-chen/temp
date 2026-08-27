from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from beyond_entropy.dataset import read_jsonl
from beyond_entropy.metrics import bootstrap_policy_evaluation, evaluate_policy
from beyond_entropy.policies import (
    EntropySearchPolicy,
    ExpectedRandomZoomPolicy,
    OracleVOIPolicy,
    Policy,
)
from beyond_entropy.qwen_semantic import (
    load_semantic_feature_dataset,
    validate_semantic_feature_dataset,
)
from beyond_entropy.rescue_gate import (
    fit_nested_oof_entropy_gate,
    fit_nested_oof_rescue_gate,
)
from beyond_entropy.schema import ActionRecord


def write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def evaluated_policy(
    records: Sequence[ActionRecord],
    policy: Policy,
    *,
    lambda_cost: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = dict(
        evaluate_policy(records, policy, lambda_cost=lambda_cost)
    )
    result["bootstrap"] = bootstrap_policy_evaluation(
        records,
        policy,
        lambda_cost=lambda_cost,
        n_resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    return result


def build_markdown(report: dict[str, object]) -> str:
    frontier = report["frontier"]
    assert isinstance(frontier, list)
    lines = [
        "# Nested OOF rescue-gate cost frontier",
        "",
        "> Lambda 0.05 remains the registered primary point; the full curve is exploratory.",
        "> Learned and entropy thresholds exclude every evaluated outer-test image group.",
        "",
        "| Lambda | Context OOF utility [95% CI] | Entropy-gate OOF utility | Always-random utility | Exhaustive-entropy utility | Oracle utility |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frontier:
        assert isinstance(row, dict)
        policies = row["policies"]
        assert isinstance(policies, dict)
        context = policies["context_oof"]
        assert isinstance(context, dict)
        bootstrap = context["bootstrap"]
        assert isinstance(bootstrap, dict)
        metrics = bootstrap["metrics"]
        assert isinstance(metrics, dict)
        interval = metrics["mean_policy_utility"]
        assert isinstance(interval, dict)
        entropy_gate = policies["entropy_gate_oof"]
        always_random = policies["always_random"]
        exhaustive = policies["exhaustive_entropy"]
        oracle = policies["oracle"]
        assert all(
            isinstance(value, dict)
            for value in (entropy_gate, always_random, exhaustive, oracle)
        )
        lines.append(
            "| {:.4f} | {:.4f} [{:.4f}, {:.4f}] | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                row["lambda_cost"],
                context["mean_policy_utility"],
                interval["ci_low"],
                interval["ci_high"],
                entropy_gate["mean_policy_utility"],
                always_random["mean_policy_utility"],
                exhaustive["mean_policy_utility"],
                oracle["mean_policy_utility"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cost frontier for nested OOF rescue gates")
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--lambda-costs",
        type=float,
        nargs="+",
        default=[0.0, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1],
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()

    records = read_jsonl(args.rollouts)
    feature_data = load_semantic_feature_dataset(args.features)
    validate_semantic_feature_dataset(feature_data, records)
    decision_by_key = {
        (str(decision["state_id"]), str(decision["replicate_id"])): decision
        for decision in feature_data["decisions"]
    }
    frontier: list[dict[str, Any]] = []
    for lambda_cost in args.lambda_costs:
        context, _ = fit_nested_oof_rescue_gate(
            records,
            decision_by_key,
            n_outer_folds=args.outer_folds,
            lambda_cost=lambda_cost,
            feature_mode="context",
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
            seed=args.seed,
        )
        entropy_gate = fit_nested_oof_entropy_gate(
            records,
            n_outer_folds=args.outer_folds,
            lambda_cost=lambda_cost,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
            seed=args.seed,
        )
        frontier.append(
            {
                "lambda_cost": lambda_cost,
                "policies": {
                    "context_oof": context["policy_result"],
                    "entropy_gate_oof": entropy_gate["policy_result"],
                    "always_random": evaluated_policy(
                        records,
                        ExpectedRandomZoomPolicy(),
                        lambda_cost=lambda_cost,
                        bootstrap_resamples=args.bootstrap_resamples,
                        bootstrap_seed=args.bootstrap_seed,
                    ),
                    "exhaustive_entropy": evaluated_policy(
                        records,
                        EntropySearchPolicy(),
                        lambda_cost=lambda_cost,
                        bootstrap_resamples=args.bootstrap_resamples,
                        bootstrap_seed=args.bootstrap_seed,
                    ),
                    "oracle": evaluated_policy(
                        records,
                        OracleVOIPolicy(lambda_cost),
                        lambda_cost=lambda_cost,
                        bootstrap_resamples=args.bootstrap_resamples,
                        bootstrap_seed=args.bootstrap_seed,
                    ),
                },
                "context_diagnostic": context,
                "entropy_gate_diagnostic": entropy_gate,
            }
        )
    report: dict[str, object] = {
        "scientific_status": "exploratory cost frontier; lambda=0.05 is primary",
        "run": {
            "rollouts": str(args.rollouts.resolve()),
            "rollouts_sha256": hashlib.sha256(args.rollouts.read_bytes()).hexdigest(),
            "features": str(args.features.resolve()),
            "features_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
            "code_revision": os.environ.get("BE_CODE_REVISION"),
            "lambda_costs": args.lambda_costs,
            "outer_folds": args.outer_folds,
            "seed": args.seed,
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "frontier": frontier,
    }
    json_path = args.output_dir / "report.json"
    markdown_path = args.output_dir / "report.md"
    write_json(report, json_path)
    markdown_path.write_text(build_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, indent=2))


if __name__ == "__main__":
    main()
