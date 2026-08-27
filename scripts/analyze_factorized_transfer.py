from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from beyond_entropy.dataset import read_jsonl
from beyond_entropy.transfer_gate import fit_factorized_context_transfer


def write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_manifest_strata(path: Path) -> dict[str, str]:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            result[str(value["state_id"])] = str(value.get("stratum", "all"))
    return result


def build_markdown(report: dict[str, object]) -> str:
    evaluation = report["evaluation"]
    assert isinstance(evaluation, dict)
    policies = evaluation["policies"]
    assert isinstance(policies, dict)
    lines = [
        "# ChartQA-to-V*Bench factorized-gate transfer",
        "",
        "> Model selection, scaling, and threshold use ChartQA labels only.",
        "> V*Bench labels are consumed only by the final evaluation below.",
        "",
        "| Policy | Accuracy gain | Tool rate | Utility [95% state-bootstrap CI] |",
        "|---|---:|---:|---:|",
    ]
    for name, result in policies.items():
        assert isinstance(result, dict)
        bootstrap = result["bootstrap"]
        assert isinstance(bootstrap, dict)
        metrics = bootstrap["metrics"]
        assert isinstance(metrics, dict)
        interval = metrics["mean_policy_utility"]
        assert isinstance(interval, dict)
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} [{:.4f}, {:.4f}] |".format(
                name,
                result["accuracy_gain"],
                result["tool_use_rate"],
                result["mean_policy_utility"],
                interval["ci_low"],
                interval["ci_high"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Transfer a ChartQA factorized gate")
    parser.add_argument("--source-rollouts", type=Path, required=True)
    parser.add_argument("--target-rollouts", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lambda-cost", type=float, default=0.05)
    parser.add_argument("--error-feature-mode", default="context")
    parser.add_argument("--rescue-feature-mode", default="context")
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    source_records = read_jsonl(args.source_rollouts)
    target_records = read_jsonl(args.target_rollouts)
    target_strata = read_manifest_strata(args.target_manifest)
    evaluation, model = fit_factorized_context_transfer(
        source_records,
        target_records,
        lambda_cost=args.lambda_cost,
        error_feature_mode=args.error_feature_mode,
        rescue_feature_mode=args.rescue_feature_mode,
        target_strata=target_strata,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        seed=args.seed,
    )
    report: dict[str, object] = {
        "scientific_status": "cross-benchmark transfer diagnostic",
        "run": {
            "source_rollouts": str(args.source_rollouts.resolve()),
            "source_rollouts_sha256": hashlib.sha256(
                args.source_rollouts.read_bytes()
            ).hexdigest(),
            "target_rollouts": str(args.target_rollouts.resolve()),
            "target_rollouts_sha256": hashlib.sha256(
                args.target_rollouts.read_bytes()
            ).hexdigest(),
            "target_manifest": str(args.target_manifest.resolve()),
            "target_manifest_sha256": hashlib.sha256(
                args.target_manifest.read_bytes()
            ).hexdigest(),
            "code_revision": os.environ.get("BE_CODE_REVISION"),
            "lambda_cost": args.lambda_cost,
            "error_feature_mode": args.error_feature_mode,
            "rescue_feature_mode": args.rescue_feature_mode,
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed": args.bootstrap_seed,
            "seed": args.seed,
        },
        "evaluation": evaluation,
    }
    write_json(model, args.output_dir / "model.json")
    json_path = args.output_dir / "report.json"
    markdown_path = args.output_dir / "report.md"
    write_json(report, json_path)
    markdown_path.write_text(build_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, indent=2))


if __name__ == "__main__":
    main()
