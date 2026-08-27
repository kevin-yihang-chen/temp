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
from beyond_entropy.rescue_gate import (
    aggregate_rescue_gate_splits,
    build_rescue_gate_markdown,
    fit_rescue_gate_split,
)


def write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a compact semantic rescuability gate")
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 11, 17, 29, 47])
    parser.add_argument("--lambda-cost", type=float, default=0.05)
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
    split_reports = []
    for seed in args.seeds:
        split_report, model = fit_rescue_gate_split(
            records,
            decision_by_key,
            lambda_cost=args.lambda_cost,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
            seed=seed,
        )
        split_reports.append(split_report)
        write_json(model, args.output_dir / f"seed-{seed}" / "model.json")
        write_json(split_report, args.output_dir / f"seed-{seed}" / "report.json")
    report = {
        "scientific_status": "exploratory frozen-feature diagnostic; not a benchmark claim",
        "run": {
            "rollouts": str(args.rollouts.resolve()),
            "rollouts_sha256": hashlib.sha256(args.rollouts.read_bytes()).hexdigest(),
            "features": str(args.features.resolve()),
            "features_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
            "code_revision": os.environ.get("BE_CODE_REVISION"),
            "lambda_cost": args.lambda_cost,
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "splits": split_reports,
        "aggregate": aggregate_rescue_gate_splits(split_reports),
    }
    json_path = args.output_dir / "report.json"
    markdown_path = args.output_dir / "report.md"
    write_json(report, json_path)
    markdown_path.write_text(build_rescue_gate_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, indent=2))


if __name__ == "__main__":
    main()
