#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from beyond_entropy.action_value import _decision_rows, _ranking_diagnostics
from beyond_entropy.dataset import read_jsonl
from beyond_entropy.metrics import (
    bootstrap_policy_evaluation,
    evaluate_policy,
    paired_bootstrap_policy_difference,
)
from beyond_entropy.policies import ExpectedRandomZoomPolicy
from beyond_entropy.qwen_semantic import (
    load_semantic_feature_dataset,
    validate_semantic_feature_dataset,
)
from beyond_entropy.rescue_gate import PrecomputedActionGatePolicy


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate zero-shot question-to-region attention ranking"
    )
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--lambda-cost", type=float, default=0.05)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260828)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    if args.lambda_cost < 0.0 or args.bootstrap_resamples <= 0:
        raise ValueError("cost must be non-negative and resamples must be positive")
    records = read_jsonl(args.rollouts)
    features = load_semantic_feature_dataset(args.features)
    validate_semantic_feature_dataset(features, records)
    baselines, zooms = _decision_rows(records)
    decision_by_key = {
        (str(decision["state_id"]), str(decision["replicate_id"])): decision
        for decision in features["decisions"]
    }
    selected: dict[tuple[str, str], str] = {}
    for key in sorted(baselines):
        decision = decision_by_key[key]
        attention = decision.get("question_region_attention")
        if attention is None or tuple(attention.shape) != (len(zooms[key]),):
            raise ValueError(f"decision {key!r} lacks complete region attention")
        scored = list(zip(decision["action_ids"], attention.tolist()))
        selected[key] = str(max(scored, key=lambda item: (item[1], item[0]))[0])
    attention_policy = PrecomputedActionGatePolicy(
        selected,
        name="question_region_attention_top1",
    )
    random_policy = ExpectedRandomZoomPolicy()
    result = {
        "scientific_status": (
            "development-only zero-shot ranking diagnostic; always-call policies "
            "are not deployable stopping rules"
        ),
        "domain": args.domain,
        "lambda_cost": args.lambda_cost,
        "ranking": _ranking_diagnostics(
            selected,
            sorted(baselines),
            zooms,
            {key: args.domain for key in baselines},
            lambda_cost=args.lambda_cost,
        ),
        "attention_policy": evaluate_policy(
            records,
            attention_policy,
            lambda_cost=args.lambda_cost,
        ),
        "attention_bootstrap": bootstrap_policy_evaluation(
            records,
            attention_policy,
            lambda_cost=args.lambda_cost,
            n_resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed,
            cluster_by="source_id",
        ),
        "random_policy": evaluate_policy(
            records,
            random_policy,
            lambda_cost=args.lambda_cost,
        ),
        "attention_minus_random": paired_bootstrap_policy_difference(
            records,
            random_policy,
            records,
            attention_policy,
            lambda_cost=args.lambda_cost,
            n_resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed,
            cluster_by="source_id",
        ),
        "run": {
            "features": str(args.features.resolve()),
            "features_sha256": _sha256(args.features),
            "rollouts": str(args.rollouts.resolve()),
            "rollouts_sha256": _sha256(args.rollouts),
            "code_revision": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "formal_outcomes_used": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
