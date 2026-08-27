from __future__ import annotations

import argparse
import json
from pathlib import Path

from beyond_entropy.robustness import (
    aggregate_semantic_reports,
    build_robustness_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize semantic split robustness")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lambda-cost", type=float, default=0.05)
    args = parser.parse_args()
    report_paths = sorted(args.input_dir.glob("seed-*/report.json"))
    report = aggregate_semantic_reports(report_paths, lambda_cost=args.lambda_cost)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "robustness_report.json"
    markdown_path = args.output_dir / "robustness_report.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(build_robustness_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, indent=2))


if __name__ == "__main__":
    main()
