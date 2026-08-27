from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from beyond_entropy.dataset import read_jsonl
from beyond_entropy.transfer_gate import evaluate_frozen_factorized_context_model


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def read_manifest_strata(path: Path) -> dict[str, str]:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            result[str(value["state_id"])] = str(value["stratum"])
    return result


def write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_markdown(report: dict[str, object]) -> str:
    evaluation = report["evaluation"]
    assert isinstance(evaluation, dict)
    policies = evaluation["policies"]
    assert isinstance(policies, dict)
    criterion = evaluation["primary_confirmation_criterion"]
    assert isinstance(criterion, dict)
    lines = [
        "# Independent ChartQA validation confirmation",
        "",
        f"> Primary criterion passed: **{criterion['passed']}**.",
        "> The model, scaler, regularization, and absolute threshold were frozen before target outcomes were inspected.",
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
    lines.extend(["", "## Criterion", ""])
    for name, passed in criterion.items():
        lines.append(f"- {name}: {passed}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen confirmation model")
    parser.add_argument("--target-rollouts", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--frozen-model", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--expected-source-report-sha256", required=True)
    parser.add_argument("--expected-target-manifest-sha256", required=True)
    parser.add_argument("--lambda-cost", type=float, default=0.05)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()

    actual_hashes = {
        "frozen_model": sha256(args.frozen_model),
        "source_report": sha256(args.source_report),
        "target_manifest": sha256(args.target_manifest),
    }
    expected_hashes = {
        "frozen_model": args.expected_model_sha256,
        "source_report": args.expected_source_report_sha256,
        "target_manifest": args.expected_target_manifest_sha256,
    }
    if actual_hashes != expected_hashes:
        raise ValueError(f"confirmation input hash mismatch: {actual_hashes}")
    model = read_json(args.frozen_model)
    source_report = read_json(args.source_report)
    source_evaluation = source_report["evaluation"]
    if not isinstance(source_evaluation, dict):
        raise ValueError("source report lacks evaluation metadata")
    source_entropy_threshold = float(source_evaluation["source_entropy_threshold"])
    target_records = read_jsonl(args.target_rollouts)
    target_strata = read_manifest_strata(args.target_manifest)
    evaluation = evaluate_frozen_factorized_context_model(
        model,
        target_records,
        source_entropy_threshold=source_entropy_threshold,
        lambda_cost=args.lambda_cost,
        target_strata=target_strata,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    report: dict[str, object] = {
        "scientific_status": "independent confirmation with frozen source model",
        "run": {
            "target_rollouts": str(args.target_rollouts.resolve()),
            "target_rollouts_sha256": sha256(args.target_rollouts),
            "target_manifest": str(args.target_manifest.resolve()),
            "target_manifest_sha256": actual_hashes["target_manifest"],
            "frozen_model": str(args.frozen_model.resolve()),
            "frozen_model_sha256": actual_hashes["frozen_model"],
            "source_report": str(args.source_report.resolve()),
            "source_report_sha256": actual_hashes["source_report"],
            "code_revision": os.environ.get("BE_CODE_REVISION"),
            "lambda_cost": args.lambda_cost,
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "evaluation": evaluation,
    }
    json_path = args.output_dir / "report.json"
    markdown_path = args.output_dir / "report.md"
    write_json(report, json_path)
    markdown_path.write_text(build_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, indent=2))


if __name__ == "__main__":
    main()
