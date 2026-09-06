#!/usr/bin/env python3
"""Evaluate the frozen three-domain Phase-C transaction exactly once."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from beyond_entropy.phase_c_formal_evaluation import evaluate_phase_c_formal
from beyond_entropy.phase_c_formal_transaction import validate_formal_access
from beyond_entropy.phase_c_training import BENCHMARKS, SEEDS
from beyond_entropy.predictability_matrix_artifacts import (
    atomic_json_write_exclusive,
    sha256_file,
)
from beyond_entropy.sequential_schema import SequentialRolloutRecord


def _git_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _read_records(path: Path) -> list[SequentialRolloutRecord]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(SequentialRolloutRecord.from_dict(json.loads(line)))
    if not records:
        raise ValueError(f"empty formal rollout file: {path}")
    return records


def _load_rollouts(plan: dict, benchmark: str) -> tuple[list[SequentialRolloutRecord], dict]:
    spec = plan["benchmarks"][benchmark]
    merged = Path(spec["merged_output"])
    report_path = merged / "report.json"
    rollout_path = merged / "rollouts.jsonl"
    report = json.loads(report_path.read_text())
    required = {
        "schema": "merged_sequential_rollout_bank_v1",
        "completed": True,
        "benchmark": benchmark,
        "dataset_role": "test",
        "test_accessed": True,
        "manifest_sha256": spec["manifest_sha256"],
        "code_revision": plan["code_revision"],
        "shard_count": plan["generation"]["shard_count"],
        "generation_seeds": plan["generation"]["generation_seeds"],
        "states": spec["states"],
        "records": spec["states"],
    }
    if any(report.get(key) != value for key, value in required.items()):
        raise ValueError(f"formal merged rollout contract failed for {benchmark}")
    if report.get("rollouts_sha256") != sha256_file(rollout_path):
        raise ValueError(f"formal merged rollout hash failed for {benchmark}")
    return _read_records(rollout_path), {
        "report": {"path": str(report_path), "sha256": sha256_file(report_path)},
        "rollouts": {"path": str(rollout_path), "sha256": sha256_file(rollout_path)},
    }


def _load_predictions(plan: dict, plan_sha256: str, ledger_sha256: str) -> tuple[list[dict], dict]:
    payloads, evidence = [], {}
    for seed in SEEDS:
        path = Path(plan["predictions"][str(seed)])
        payload = json.loads(path.read_text())
        if (
            payload.get("plan_sha256") != plan_sha256
            or payload.get("access_ledger_sha256") != ledger_sha256
            or payload.get("seed") != seed
            or payload.get("job_id") != os.environ.get("SLURM_JOB_ID")
        ):
            raise ValueError(f"formal prediction provenance failed for seed {seed}")
        payloads.append(payload)
        evidence[str(seed)] = {"path": str(path), "sha256": sha256_file(path)}
    return payloads, evidence


def _render_figures(report: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    display = (
        "factorized_potential_outcomes", "outcome_only", "counterfactual_utility",
        "random_gate", "oracle_gain_rank",
    )
    labels = {
        "factorized_potential_outcomes": "Factorized potential outcomes",
        "outcome_only": "Outcome-only SFT",
        "counterfactual_utility": "Direct counterfactual SFT",
        "random_gate": "Random gate",
        "oracle_gain_rank": "Oracle ranking",
    }
    for benchmark in BENCHMARKS:
        result = report["benchmarks"][benchmark]
        frontier = result["frontier"]
        strongest = result["primary_strongest_uncertainty_baseline"]
        plt.figure(figsize=(7.1, 4.8))
        for name in (*display, strongest):
            shown = "Strongest uncertainty" if name == strongest else labels[name]
            xs = [
                row["policies_by_lambda"]["0.0"][name]["avg_incremental_visual_cost"]
                for row in frontier
            ]
            ys = [
                row["policies_by_lambda"]["0.0"][name]["accuracy"]
                for row in frontier
            ]
            plt.plot(xs, ys, marker="o", linewidth=1.6, label=shown)
        answer_accuracy = result["primary_metrics"]["answer_only"]["stop_accuracy"]
        plt.scatter([0.0], [answer_accuracy], marker="x", s=70, label="Answer only")
        plt.xlabel("Average incremental visual cost / tool calls")
        plt.ylabel("Task score")
        plt.title(f"{benchmark}: accuracy-cost frontier")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(output / f"{benchmark}-accuracy-cost-frontier.png", dpi=200)
        plt.close()

        calibration = result["calibration"]["factorized_potential_outcomes"]
        plt.figure(figsize=(6.4, 4.5))
        for seed, value in calibration.items():
            bins = value["quantile_bins"]
            plt.plot(
                [item["mean_score"] for item in bins],
                [item["mean_gain"] for item in bins],
                marker="o", label=f"seed {seed}", alpha=.8,
            )
        plt.axhline(0.0, color="black", linewidth=.8)
        plt.xlabel("Predicted counterfactual gain (score quantile)")
        plt.ylabel("Observed counterfactual task gain")
        plt.title(f"{benchmark}: utility prediction")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(output / f"{benchmark}-utility-prediction.png", dpi=200)
        plt.close()


def _markdown(report: dict) -> str:
    rows = []
    for benchmark in BENCHMARKS:
        result = report["benchmarks"][benchmark]
        comparison = result["primary_comparisons"][
            "factorized_potential_outcomes_minus_outcome_only"
        ]["accuracy"]
        factorized = result["primary_metrics"]["factorized_potential_outcomes"]
        strongest_name = result["primary_strongest_uncertainty_baseline"]
        strongest = result["primary_metrics"][strongest_name]
        rows.append(
            f"| {benchmark} | {factorized['accuracy']:.6f} | "
            f"{comparison['observed_delta']:+.6f} | "
            f"[{comparison['ci_low']:+.6f}, {comparison['ci_high']:+.6f}] | "
            f"{strongest_name}: {strongest['accuracy']:.6f} |"
        )
    semantic = "\n".join(
        f"- `{mode}`: {'PASS' if value['passed'] else 'FAIL'}; mean accuracy delta "
        f"{value['mean_accuracy_delta_original_minus_ablation']:+.6f}; ranking changed "
        f"on {', '.join(value['domains_with_changed_ranking']) or 'none'}."
        for mode, value in report["semantic_gates"].items()
    )
    go = report["decision"] == "GO"
    direct_positive = sum(
        report["benchmarks"][benchmark]["primary_comparisons"]
        ["factorized_potential_outcomes_minus_counterfactual_utility"]
        ["accuracy"]["observed_delta"] > 0
        for benchmark in BENCHMARKS
    )
    return f"""# Factorized Phase-C Formal GO / NO-GO

Decision: **{report['decision']}**

This is the single pre-registered held-out transaction. Scores are averaged over
three independently trained and independently deployed seeds; no score ensemble
is used to select actions.

| Domain | Factorized task score | vs Outcome-only | 95% source-cluster CI | Strongest uncertainty |
|---|---:|---:|---:|---|
{chr(10).join(rows)}

Mean task-score delta vs Outcome-only: `{report['mean_accuracy_delta_vs_outcome']:+.6f}`.

## Semantic gates

{semantic}

## Four decisions

1. Does Factorized exceed Outcome-only? **{'Yes' if len(report['positive_vs_outcome_domains']) >= 2 else 'No'}** ({len(report['positive_vs_outcome_domains'])}/3 positive domains).
2. Does Factorized exceed direct counterfactual SFT? **{'Yes' if direct_positive >= 2 else 'No'}** ({direct_positive}/3 positive domains).
3. Does the result hold on at least two domains with the frozen safeguards? **{'Yes' if go else 'No'}**.
4. Is escalation to a larger/RL phase justified by this test? **{'Yes' if go else 'No'}**.

The frozen machine checks are recorded in `report.json`. A `NO_GO` is a valid
formal result and does not authorize post-hoc seed, threshold, rate, or method
selection on this held-out set.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--ledger", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("formal evaluation requires the frozen Slurm allocation")
    plan, _ = validate_formal_access(args.plan, args.plan_sha256, args.ledger)
    root = Path(__file__).resolve().parents[1]
    if _git_revision(root) != plan["code_revision"]:
        raise ValueError("formal evaluation code revision drifted")
    for relative, expected in plan["code_hashes"].items():
        if sha256_file(root / relative) != expected:
            raise ValueError(f"formal evaluation code hash drifted: {relative}")

    records_by_benchmark, rollout_evidence = {}, {}
    for benchmark in BENCHMARKS:
        records_by_benchmark[benchmark], rollout_evidence[benchmark] = _load_rollouts(
            plan, benchmark
        )
    ledger_sha256 = sha256_file(args.ledger)
    payloads, prediction_evidence = _load_predictions(
        plan, args.plan_sha256, ledger_sha256
    )
    report = evaluate_phase_c_formal(plan, records_by_benchmark, payloads)
    report["plan_sha256"] = args.plan_sha256
    report["access_ledger_sha256"] = ledger_sha256
    report["code_revision"] = plan["code_revision"]
    report["job_id"] = os.environ["SLURM_JOB_ID"]
    report["rollout_evidence"] = rollout_evidence
    report["prediction_evidence"] = prediction_evidence

    destination = Path(plan["evaluation_output"])
    staging = destination.with_name(
        f"{destination.name}.staging-job-{os.environ['SLURM_JOB_ID']}"
    )
    if destination.exists() or staging.exists():
        raise FileExistsError("formal evaluation output already exists")
    staging.mkdir(parents=True)
    try:
        _render_figures(report, staging)
        atomic_json_write_exclusive(staging / "report.json", report)
        (staging / "GO_NO_GO.md").write_text(_markdown(report))
        os.replace(staging, destination)
    except BaseException:
        # Preserve the staging evidence for postmortem; never overwrite it.
        raise
    print(json.dumps({
        "decision": report["decision"],
        "output": str(destination),
        "report_sha256": sha256_file(destination / "report.json"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
