#!/usr/bin/env python3
"""Run the irreversible Phase-C rollout, scoring, and evaluation transaction."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

from beyond_entropy.phase_c_formal_transaction import (
    load_formal_plan,
    start_formal_access,
)
from beyond_entropy.phase_c_training import BENCHMARKS, SEEDS
from beyond_entropy.predictability_matrix_artifacts import (
    atomic_json_write_exclusive,
    sha256_file,
)


def _run_logged(command: list[str], *, environment: dict[str, str], log: Path) -> subprocess.Popen:
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("x", encoding="utf-8")
    process = subprocess.Popen(
        command, env=environment, stdout=handle, stderr=subprocess.STDOUT, text=True,
    )
    # Keep the handle alive on the Popen object until wait completes.
    process._phase_c_log_handle = handle  # type: ignore[attr-defined]
    return process


def _wait_all(processes: list[tuple[str, subprocess.Popen]]) -> None:
    failures = []
    for label, process in processes:
        return_code = process.wait()
        process._phase_c_log_handle.close()  # type: ignore[attr-defined]
        if return_code:
            failures.append((label, return_code))
    if failures:
        raise RuntimeError(f"formal subprocess failure(s): {failures}")


def _generator_command(plan: dict, plan_path: str, plan_sha: str, ledger: str,
                       benchmark: str, shard: int) -> list[str]:
    generation = plan["generation"]
    command = [
        sys.executable, "scripts/generate_counterfactual_prefixes.py",
        "--manifest", plan["benchmarks"][benchmark]["manifest"],
        "--output", plan["benchmarks"][benchmark]["shards"][str(shard)],
        "--benchmark", benchmark,
        "--dataset-role", "test",
        "--model", plan["model"],
        "--revision", plan["model_revision"],
        "--device-map", "cuda:0",
        "--dtype", generation["dtype"],
        "--attention-implementation", generation["attention_implementation"],
        "--max-new-tokens", str(generation["max_new_tokens"]),
        "--min-pixels", str(generation["min_pixels"]),
        "--max-pixels", str(generation["max_pixels"]),
        "--shard-count", str(generation["shard_count"]),
        "--shard-index", str(shard),
        "--checkpoint-interval", str(generation["checkpoint_interval"]),
        "--formal-plan", plan_path,
        "--formal-plan-sha256", plan_sha,
        "--formal-access-ledger", ledger,
    ]
    for seed in generation["generation_seeds"]:
        command.extend(("--generation-seed", str(seed)))
    return command


def _merge_command(plan: dict, benchmark: str) -> list[str]:
    generation = plan["generation"]
    spec = plan["benchmarks"][benchmark]
    command = [
        sys.executable, "scripts/merge_sequential_rollout_shards.py",
        "--manifest", spec["manifest"],
        "--expected-manifest-sha256", spec["manifest_sha256"],
        "--run-root", spec["rollout_root"],
        "--shard-count", str(generation["shard_count"]),
        "--output-dir", spec["merged_output"],
        "--expected-code-revision", plan["code_revision"],
        "--benchmark", benchmark,
        "--dataset-role", "test",
    ]
    for seed in generation["generation_seeds"]:
        command.extend(("--generation-seed", str(seed)))
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("formal transaction must run under Slurm")
    plan_path = str(Path(args.plan).resolve())
    plan = load_formal_plan(plan_path, args.plan_sha256)
    transaction_root = Path(plan["transaction_root"])
    if transaction_root.exists():
        raise FileExistsError("formal transaction has already been opened")
    visible = [value for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value]
    if len(visible) != int(plan["generation"]["shard_count"]):
        raise RuntimeError("formal transaction requires exactly four visible GPUs")

    ledger = plan["access_ledger"]
    start_formal_access(plan_path, args.plan_sha256, Path(__file__).resolve().parents[1])
    try:
        for benchmark in BENCHMARKS:
            processes = []
            for shard, gpu in enumerate(visible):
                environment = dict(os.environ)
                environment["CUDA_VISIBLE_DEVICES"] = gpu
                log = transaction_root / "logs" / f"{benchmark}-shard-{shard}.log"
                process = _run_logged(
                    _generator_command(
                        plan, plan_path, args.plan_sha256, ledger, benchmark, shard,
                    ),
                    environment=environment,
                    log=log,
                )
                processes.append((f"{benchmark}/shard-{shard}", process))
            _wait_all(processes)
            merge_log = transaction_root / "logs" / f"{benchmark}-merge.log"
            with merge_log.open("x", encoding="utf-8") as handle:
                subprocess.run(
                    _merge_command(plan, benchmark), check=True,
                    stdout=handle, stderr=subprocess.STDOUT, text=True,
                )

        scoring = []
        for index, seed in enumerate(SEEDS):
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = visible[index]
            command = [
                sys.executable, "scripts/score_factorized_phase_c_formal.py",
                "--plan", plan_path,
                "--plan-sha256", args.plan_sha256,
                "--ledger", ledger,
                "--seed", str(seed),
                "--output", plan["predictions"][str(seed)],
            ]
            scoring.append((
                f"score/seed-{seed}",
                _run_logged(
                    command, environment=environment,
                    log=transaction_root / "logs" / f"score-seed-{seed}.log",
                ),
            ))
        _wait_all(scoring)

        evaluation_log = transaction_root / "logs" / "evaluation.log"
        with evaluation_log.open("x", encoding="utf-8") as handle:
            subprocess.run(
                [
                    sys.executable, "scripts/evaluate_factorized_phase_c_formal.py",
                    "--plan", plan_path,
                    "--plan-sha256", args.plan_sha256,
                    "--ledger", ledger,
                ],
                check=True, stdout=handle, stderr=subprocess.STDOUT, text=True,
            )
        report_path = Path(plan["evaluation_output"]) / "report.json"
        report = json.loads(report_path.read_text())
        atomic_json_write_exclusive(transaction_root / "execution-complete.json", {
            "schema": "factorized_phase_c_formal_execution_v1",
            "status": "completed",
            "one_shot": True,
            "test_accessed": True,
            "job_id": os.environ["SLURM_JOB_ID"],
            "plan_sha256": args.plan_sha256,
            "access_ledger_sha256": sha256_file(ledger),
            "evaluation_report": {
                "path": str(report_path), "sha256": sha256_file(report_path),
            },
            "decision": report["decision"],
        })
    except BaseException as exc:
        failure = transaction_root / "execution-failure.json"
        if not failure.exists():
            atomic_json_write_exclusive(failure, {
                "schema": "factorized_phase_c_formal_execution_failure_v1",
                "status": "failed_after_irreversible_access",
                "one_shot": True,
                "test_accessed": True,
                "job_id": os.environ["SLURM_JOB_ID"],
                "plan_sha256": args.plan_sha256,
                "access_ledger_sha256": sha256_file(ledger),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
        raise


if __name__ == "__main__":
    main()
