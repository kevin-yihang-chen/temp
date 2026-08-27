from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from beyond_entropy.candidate_ablation import (
    build_candidate_ablation_markdown,
    compare_candidate_sets,
)
from beyond_entropy.dataset import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare matched rollout candidate sets")
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lambda-cost", type=float, default=0.05)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cluster-by",
        choices=("state_id", "image_id", "source_id"),
        default="state_id",
    )
    args = parser.parse_args()

    report = compare_candidate_sets(
        read_jsonl(args.left),
        read_jsonl(args.right),
        lambda_cost=args.lambda_cost,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
        cluster_by=args.cluster_by,
    )
    report["inputs"] = {
        "left": str(args.left.resolve()),
        "left_sha256": hashlib.sha256(args.left.read_bytes()).hexdigest(),
        "right": str(args.right.resolve()),
        "right_sha256": hashlib.sha256(args.right.read_bytes()).hexdigest(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "candidate_ablation.json"
    markdown_path = args.output_dir / "candidate_ablation.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        build_candidate_ablation_markdown(report),
        encoding="utf-8",
    )
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, indent=2))


if __name__ == "__main__":
    main()
