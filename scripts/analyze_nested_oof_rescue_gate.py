from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from beyond_entropy.dataset import read_jsonl
from beyond_entropy.qwen_semantic import (
    load_semantic_feature_dataset,
    validate_semantic_feature_dataset,
)
from beyond_entropy.rescue_gate import fit_nested_oof_rescue_gate


def write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_markdown(report: dict[str, object]) -> str:
    evaluations = report["evaluations"]
    assert isinstance(evaluations, dict)
    lines = [
        "# Nested OOF compact rescue-gate diagnostic",
        "",
        "> Each decision is evaluated exactly once by a model that excluded its image group.",
        "> Bootstrap intervals condition on the fitted cross-fold predictions.",
        "",
        "| Features | Accuracy gain | Tool rate | Utility [95% state-bootstrap CI] | Helpful ROC-AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for feature_mode, evaluation in evaluations.items():
        assert isinstance(evaluation, dict)
        result = evaluation["policy_result"]
        assert isinstance(result, dict)
        bootstrap = result["bootstrap"]
        assert isinstance(bootstrap, dict)
        metrics = bootstrap["metrics"]
        assert isinstance(metrics, dict)
        interval = metrics["mean_policy_utility"]
        assert isinstance(interval, dict)
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} [{:.4f}, {:.4f}] | {:.4f} |".format(
                feature_mode,
                result["accuracy_gain"],
                result["tool_use_rate"],
                result["mean_policy_utility"],
                interval["ci_low"],
                interval["ci_high"],
                evaluation["pooled_helpful_roc_auc_of_fold_margin"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Nested grouped OOF rescue-gate evaluation")
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--feature-modes",
        nargs="+",
        choices=("semantic", "context", "semantic-context"),
        default=["semantic", "context", "semantic-context"],
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lambda-cost", type=float, default=0.05)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()

    records = read_jsonl(args.rollouts)
    feature_data = load_semantic_feature_dataset(args.features)
    validate_semantic_feature_dataset(feature_data, records)
    decision_by_key = {
        (str(decision["state_id"]), str(decision["replicate_id"])): decision
        for decision in feature_data["decisions"]
    }
    evaluations = {}
    for feature_mode in args.feature_modes:
        evaluation, models = fit_nested_oof_rescue_gate(
            records,
            decision_by_key,
            n_outer_folds=args.outer_folds,
            lambda_cost=args.lambda_cost,
            feature_mode=feature_mode,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
            seed=args.seed,
        )
        evaluations[feature_mode] = evaluation
        write_json(models, args.output_dir / feature_mode / "models.json")
        write_json(evaluation, args.output_dir / feature_mode / "report.json")
    report = {
        "scientific_status": "exploratory nested grouped OOF diagnostic",
        "run": {
            "rollouts": str(args.rollouts.resolve()),
            "rollouts_sha256": hashlib.sha256(args.rollouts.read_bytes()).hexdigest(),
            "features": str(args.features.resolve()),
            "features_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
            "code_revision": os.environ.get("BE_CODE_REVISION"),
            "feature_modes": args.feature_modes,
            "outer_folds": args.outer_folds,
            "seed": args.seed,
            "lambda_cost": args.lambda_cost,
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "evaluations": evaluations,
    }
    json_path = args.output_dir / "report.json"
    markdown_path = args.output_dir / "report.md"
    write_json(report, json_path)
    markdown_path.write_text(build_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, indent=2))


if __name__ == "__main__":
    main()
