from __future__ import annotations

import argparse
import json
from pathlib import Path

from beyond_entropy.dataset import read_jsonl
from beyond_entropy.pilot_analysis import (
    analyze_counterfactual_pilot,
    build_pilot_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a frozen counterfactual pilot")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lambda-cost", type=float, default=0.05)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    manifest_rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    state_strata = {
        str(row["state_id"]): str(row.get("stratum", "unknown"))
        for row in manifest_rows
    }
    report = analyze_counterfactual_pilot(
        read_jsonl(args.data),
        state_strata=state_strata,
        lambda_cost=args.lambda_cost,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "pilot_report.json"
    markdown_path = args.output_dir / "pilot_report.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(build_pilot_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, indent=2))


if __name__ == "__main__":
    main()
