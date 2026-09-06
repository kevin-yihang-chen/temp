"""Freeze and enforce the one-shot Factorized Phase-C held-out transaction."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .phase_c_training import BENCHMARKS, METHODS, SEEDS
from .predictability_matrix_artifacts import atomic_json_write_exclusive, sha256_file
from .sequential_rollout_shards import shard_directory_name


FORMAL_MODES = ("original", "question_shuffle", "image_shuffle", "region_shuffle")
PLAN_FIELDS = frozenset({
    "schema", "one_shot", "test_authorized", "deadline_hkt", "created_at_utc",
    "code_revision", "code_hashes", "config", "allocation_report", "training_matrix", "model",
    "model_revision", "methods", "seeds", "benchmarks", "generation", "policy",
    "baselines", "ablations", "go_rule", "selectors", "transaction_root", "access_ledger",
    "predictions", "evaluation_output", "runtime_smoke", "post_access_changes_forbidden",
})


def _git_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _resolved(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("formal transaction path escapes repository") from exc
    return path


def _hash_spec(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def validate_formal_config(config: Mapping[str, Any]) -> None:
    if (
        config.get("schema") != "factorized_phase_c_formal_config_v1"
        or config.get("deadline_hkt") != "2026-09-13T23:59:00+08:00"
        or tuple(config.get("methods", ())) != METHODS
        or tuple(config.get("seeds", ())) != SEEDS
        or tuple(sorted(config.get("benchmarks", {}))) != BENCHMARKS
        or config.get("one_shot") is not True
        or config.get("test_authorized") is not True
        or config.get("allow_method_selection_after_access") is not False
        or config.get("allow_seed_selection_after_access") is not False
        or config.get("allow_threshold_selection_after_access") is not False
    ):
        raise ValueError("invalid Phase-C formal configuration")
    generation = config.get("generation", {})
    policy = config.get("policy", {})
    ablations = config.get("ablations", {})
    baselines = config.get("baselines", {})
    go_rule = config.get("go_rule", {})
    if (
        generation.get("generation_seeds") != [0]
        or generation.get("proposer") != "sequential-opposite-ug-v1"
        or generation.get("candidate_count") != 4
        or generation.get("visual_cost_per_crop") != 1
        or generation.get("shard_count") != 4
        or policy.get("primary_call_rate") != 0.25
        or policy.get("primary_lambda") != 0.05
        or policy.get("bootstrap_samples", 0) < 20_000
        or tuple(baselines.get("uncertainty", ())) != (
            "entropy", "confidence", "margin"
        )
        or not isinstance(baselines.get("random_seed"), int)
        or any(
            baselines.get(name) is not True
            for name in (
                "include_answer_only", "include_random_gate", "include_oracle"
            )
        )
        or tuple(ablations.get("methods", ())) != ("factorized_potential_outcomes",)
        or tuple(ablations.get("modes", ())) != FORMAL_MODES[1:]
        or ablations.get("permutation") != "sha256-component-derangement-v1"
        or go_rule.get("positive_mean_delta_vs_outcome_domains") != 2
        or go_rule.get(
            "required_domain_source_bootstrap_ci_low_vs_outcome_positive"
        ) != 1
        or go_rule.get("all_semantic_ablations_must_pass") is not True
    ):
        raise ValueError("formal generation, policy, ablation, or GO rule drifted")


def _validate_selector(
    *, root: Path, training_root: Path, method: str, seed: int, job_id: str,
    matrix_sha256: str,
) -> dict[str, Any]:
    method_dir = method.replace("_", "-")
    run = training_root / method_dir / f"job-{job_id}"
    report_path = run / "report.json"
    selector_path = run / "selector.pt"
    report = json.loads(report_path.read_text())
    if (
        report.get("schema") != "cv_method_post_training_report_v1"
        or report.get("stage") != "phase_c_training"
        or report.get("method") != method
        or report.get("test_accessed") is not False
        or report.get("formal_claim_eligible") is not False
        or report.get("provenance", {}).get("config", {}).get("seed") != seed
        or report.get("provenance", {}).get("config", {}).get(
            "matrix_config_sha256"
        ) != matrix_sha256
        or report.get("selector_sha256") != sha256_file(selector_path)
    ):
        raise ValueError(f"invalid frozen selector for {method} seed {seed}")
    required_checks = (
        "paired_reward_gain_contract", "binary_action_support_valid",
        "finite_trace", "all_trainable_groups_received_gradient",
        "all_trainable_groups_updated", "no_proposed_crop_execution",
        "finite_nonconstant_validation_scores",
    )
    if not all(report.get("checks", {}).get(name) is True for name in required_checks):
        raise ValueError(f"selector engineering gate failed for {method} seed {seed}")
    return {
        "job_id": str(job_id),
        "selector": _hash_spec(selector_path),
        "training_report": _hash_spec(report_path),
        "schedule_sha256": report["schedule_sha256"],
    }


def build_phase_c_formal_plan(
    *, config_path: str | Path, repository_root: str | Path,
    seed_jobs: Mapping[int, str], transaction_id: str, output_path: str | Path,
    runtime_smoke_path: str | Path,
) -> dict[str, Any]:
    """Freeze selectors and output paths without reading held-out manifest bytes."""

    root = Path(repository_root).resolve()
    config_source = _resolved(root, str(config_path))
    plan_output = _resolved(root, str(output_path))
    config = json.loads(config_source.read_text())
    validate_formal_config(config)
    if (
        set(seed_jobs) != set(SEEDS)
        or not transaction_id
        or Path(transaction_id).name != transaction_id
        or transaction_id in {".", ".."}
    ):
        raise ValueError("formal plan requires one job ID for every frozen seed")
    if subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
        check=True, capture_output=True, text=True,
    ).stdout.strip():
        raise ValueError("tracked worktree must be clean before formal freeze")

    allocation_path = _resolved(root, config["allocation_report"]["path"])
    training_matrix_path = _resolved(root, config["training_matrix"]["path"])
    if (
        sha256_file(allocation_path) != config["allocation_report"]["sha256"]
        or sha256_file(training_matrix_path) != config["training_matrix"]["sha256"]
    ):
        raise ValueError("allocation report or training matrix hash mismatch")
    allocation = json.loads(allocation_path.read_text())
    if (
        allocation.get("selection_used_model_outcomes") is not False
        or allocation.get("heldout_sequential_outcomes_opened") is not False
    ):
        raise ValueError("held-out allocation is no longer eligible for one-shot access")

    training_root = _resolved(root, config["training_output_root"])
    selectors: dict[str, dict[str, Any]] = {method: {} for method in METHODS}
    for seed in SEEDS:
        job_id = str(seed_jobs[seed])
        schedules = []
        for method in METHODS:
            selectors[method][str(seed)] = _validate_selector(
                root=root, training_root=training_root, method=method, seed=seed,
                job_id=job_id,
                matrix_sha256=config["training_matrix"]["sha256"],
            )
            schedules.append(selectors[method][str(seed)]["schedule_sha256"])
        if len(set(schedules)) != 1:
            raise ValueError(f"matched arms have different schedules for seed {seed}")
        evaluation_path = training_root / "evaluation" / f"job-{job_id}" / "report.json"
        evaluation = json.loads(evaluation_path.read_text())
        if (
            evaluation.get("stage") != "phase_c_training"
            or evaluation.get("decision") != "PHASE_C_SEED_FROZEN"
            or evaluation.get("test_accessed") is not False
            or evaluation.get("formal_claim_eligible") is not False
        ):
            raise ValueError(f"invalid training monitor evaluation for seed {seed}")
        for method in METHODS:
            selectors[method][str(seed)]["monitor_evaluation"] = _hash_spec(
                evaluation_path
            )

    smoke_path = _resolved(root, str(runtime_smoke_path))
    smoke = json.loads(smoke_path.read_text())
    smoke_seed = int(smoke.get("seed", -1))
    smoke_checks = smoke.get("checks", {})
    if (
        smoke.get("schema") != "factorized_phase_c_formal_runtime_smoke_v1"
        or smoke.get("completed") is not True
        or smoke.get("test_accessed") is not False
        or smoke.get("formal_claim_eligible") is not False
        or smoke.get("code_revision") != _git_revision(root)
        or smoke_seed not in SEEDS
        or smoke.get("selector_training_job_id") != str(seed_jobs[smoke_seed])
        or smoke.get("matrix", {}).get("sha256") != config["training_matrix"]["sha256"]
        or not smoke_checks
        or not all(value is True for value in smoke_checks.values())
    ):
        raise ValueError("formal runtime smoke is invalid or incomplete")
    for method in METHODS:
        observed = smoke.get("selectors", {}).get(method, {}).get(str(smoke_seed), {})
        expected = selectors[method][str(smoke_seed)]
        if any(observed.get(name) != expected[name] for name in ("selector", "training_report")):
            raise ValueError("runtime smoke selector evidence differs from formal selector")

    formal_root = _resolved(root, config["formal_output_root"])
    transaction_root = formal_root / "transactions" / transaction_id
    if transaction_root.exists():
        raise FileExistsError("formal transaction root already exists")
    shard_count = int(config["generation"]["shard_count"])
    benchmark_plan = {}
    for benchmark in BENCHMARKS:
        spec = config["benchmarks"][benchmark]
        allocation_spec = allocation["benchmarks"][benchmark]["heldout"]
        if (
            int(spec["states"]) != int(allocation_spec["states"])
            or spec["manifest_sha256"] != allocation_spec["manifest_sha256"]
        ):
            raise ValueError(f"formal config differs from allocation for {benchmark}")
        manifest = _resolved(root, spec["manifest"])
        # Existence is safe to check; bytes are not read before the access ledger.
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        rollout_root = transaction_root / "rollouts" / benchmark
        benchmark_plan[benchmark] = {
            "states": int(spec["states"]),
            "manifest": str(manifest),
            "manifest_sha256": spec["manifest_sha256"],
            "rollout_root": str(rollout_root),
            "merged_output": str(rollout_root / "merged"),
            "shards": {
                str(index): str(
                    rollout_root / shard_directory_name(index, shard_count) /
                    "rollouts.jsonl"
                )
                for index in range(shard_count)
            },
        }

    script_names = (
        "freeze_factorized_phase_c_formal.py",
        "smoke_factorized_phase_c_formal_runtime.py",
        "generate_counterfactual_prefixes.py", "merge_sequential_rollout_shards.py",
        "score_factorized_phase_c_formal.py", "evaluate_factorized_phase_c_formal.py",
        "execute_factorized_phase_c_formal.py", "slurm_factorized_phase_c_formal.sh",
        "submit_factorized_phase_c_formal.sh",
        "slurm_factorized_phase_c_formal_smoke.sh",
        "submit_factorized_phase_c_formal_smoke.sh",
    )
    code_paths = sorted((root / "src/beyond_entropy").glob("*.py")) + [
        root / "scripts" / name for name in script_names
    ]
    predictions = {
        str(seed): str(transaction_root / "predictions" / f"seed-{seed}.json")
        for seed in SEEDS
    }
    plan = {
        "schema": "factorized_phase_c_formal_plan_v1",
        "one_shot": True,
        "test_authorized": True,
        "deadline_hkt": config["deadline_hkt"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_revision": _git_revision(root),
        "code_hashes": {
            str(path.relative_to(root)): sha256_file(path) for path in code_paths
        },
        "config": _hash_spec(config_source),
        "allocation_report": _hash_spec(allocation_path),
        "training_matrix": _hash_spec(training_matrix_path),
        "model": config["model"],
        "model_revision": config["model_revision"],
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "benchmarks": benchmark_plan,
        "generation": config["generation"],
        "policy": config["policy"],
        "baselines": config["baselines"],
        "ablations": config["ablations"],
        "go_rule": config["go_rule"],
        "selectors": selectors,
        "transaction_root": str(transaction_root),
        "access_ledger": str(transaction_root / "access-ledger.json"),
        "predictions": predictions,
        "evaluation_output": str(transaction_root / "evaluation"),
        "runtime_smoke": _hash_spec(smoke_path),
        "post_access_changes_forbidden": [
            "method", "seed", "threshold", "call_rate", "lambda", "ablation",
            "bootstrap", "go_rule",
        ],
    }
    validate_formal_plan(plan)
    atomic_json_write_exclusive(plan_output, plan)
    return {**plan, "plan_sha256": sha256_file(plan_output)}


def validate_formal_plan(plan: Mapping[str, Any]) -> None:
    if set(plan) != PLAN_FIELDS:
        raise ValueError("formal plan fields differ from exact contract")
    if (
        plan.get("schema") != "factorized_phase_c_formal_plan_v1"
        or plan.get("deadline_hkt") != "2026-09-13T23:59:00+08:00"
        or plan.get("one_shot") is not True
        or plan.get("test_authorized") is not True
        or tuple(plan.get("methods", ())) != METHODS
        or tuple(plan.get("seeds", ())) != SEEDS
        or tuple(sorted(plan.get("benchmarks", {}))) != BENCHMARKS
        or set(plan.get("predictions", {})) != {str(seed) for seed in SEEDS}
        or int(plan.get("generation", {}).get("shard_count", 0)) != 4
        or tuple(plan.get("baselines", {}).get("uncertainty", ())) != (
            "entropy", "confidence", "margin"
        )
        or tuple(plan.get("policy", {}).get("rates", ()))
        != (0, 0.1, 0.25, 0.5, 0.75, 1)
        or tuple(plan.get("policy", {}).get("lambdas", ()))
        != (0, 0.025, 0.05, 0.1, 0.2)
        or not plan.get("post_access_changes_forbidden")
    ):
        raise ValueError("invalid formal plan semantics")
    for benchmark, spec in plan["benchmarks"].items():
        if set(spec["shards"]) != {"0", "1", "2", "3"}:
            raise ValueError(f"formal shard coverage invalid for {benchmark}")


def load_formal_plan(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    source = Path(path).resolve()
    if sha256_file(source) != expected_sha256:
        raise ValueError("formal plan hash mismatch")
    plan = json.loads(source.read_text())
    validate_formal_plan(plan)
    return plan


def start_formal_access(
    plan_path: str | Path, expected_plan_sha256: str, repository_root: str | Path,
) -> dict[str, Any]:
    """Write the irreversible ledger before any held-out manifest byte is read."""

    root = Path(repository_root).resolve()
    plan = load_formal_plan(plan_path, expected_plan_sha256)
    if _git_revision(root) != plan["code_revision"]:
        raise ValueError("code revision changed after formal freeze")
    for relative, expected in plan["code_hashes"].items():
        if sha256_file(root / relative) != expected:
            raise ValueError(f"code changed after formal freeze: {relative}")
    for category in (
        "config", "allocation_report", "training_matrix", "runtime_smoke"
    ):
        if sha256_file(plan[category]["path"]) != plan[category]["sha256"]:
            raise ValueError(f"formal {category} changed after freeze")
    for method in METHODS:
        for seed in SEEDS:
            item = plan["selectors"][method][str(seed)]
            for category in ("selector", "training_report", "monitor_evaluation"):
                if sha256_file(item[category]["path"]) != item[category]["sha256"]:
                    raise ValueError(f"frozen selector evidence drifted: {method}/{seed}")
    ledger = Path(plan["access_ledger"])
    ledger.parent.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema": "factorized_phase_c_formal_access_v1",
        "status": "started_irreversible",
        "plan_path": str(Path(plan_path).resolve()),
        "plan_sha256": expected_plan_sha256,
        "code_revision": plan["code_revision"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "heldout_outcomes_now_considered_opened": True,
    }
    atomic_json_write_exclusive(ledger, payload)
    return payload


def validate_formal_access(
    plan_path: str | Path, expected_plan_sha256: str, ledger_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = load_formal_plan(plan_path, expected_plan_sha256)
    ledger = json.loads(Path(ledger_path).read_text())
    if (
        ledger.get("schema") != "factorized_phase_c_formal_access_v1"
        or ledger.get("status") != "started_irreversible"
        or ledger.get("plan_path") != str(Path(plan_path).resolve())
        or ledger.get("plan_sha256") != expected_plan_sha256
        or ledger.get("code_revision") != plan["code_revision"]
        or ledger.get("heldout_outcomes_now_considered_opened") is not True
        or str(Path(ledger_path).resolve()) != plan["access_ledger"]
    ):
        raise ValueError("invalid or drifted Phase-C formal access ledger")
    return plan, ledger


def authorize_formal_rollout_shard(
    *, plan_path: str | Path, expected_plan_sha256: str,
    ledger_path: str | Path, benchmark: str, manifest_path: str | Path,
    output_path: str | Path, model: str, model_revision: str,
    generation_seeds: list[int], code_revision: str, dtype: str,
    attention_implementation: str, max_new_tokens: int, min_pixels: int,
    max_pixels: int, shard_count: int, shard_index: int,
) -> dict[str, Any]:
    """Authorize exactly one plan-bound shard without reading its manifest."""

    plan, ledger = validate_formal_access(
        plan_path, expected_plan_sha256, ledger_path
    )
    if benchmark not in BENCHMARKS:
        raise ValueError("unsupported formal benchmark")
    generation = plan["generation"]
    expected = {
        "manifest_path": plan["benchmarks"][benchmark]["manifest"],
        "output_path": plan["benchmarks"][benchmark]["shards"][str(shard_index)],
        "model": plan["model"],
        "model_revision": plan["model_revision"],
        "generation_seeds": generation["generation_seeds"],
        "code_revision": plan["code_revision"],
        "dtype": generation["dtype"],
        "attention_implementation": generation["attention_implementation"],
        "max_new_tokens": generation["max_new_tokens"],
        "min_pixels": generation["min_pixels"],
        "max_pixels": generation["max_pixels"],
        "shard_count": generation["shard_count"],
        "shard_index": shard_index,
    }
    actual = {
        "manifest_path": str(Path(manifest_path).resolve()),
        "output_path": str(Path(output_path).resolve()),
        "model": model,
        "model_revision": model_revision,
        "generation_seeds": generation_seeds,
        "code_revision": code_revision,
        "dtype": dtype,
        "attention_implementation": attention_implementation,
        "max_new_tokens": max_new_tokens,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "shard_count": shard_count,
        "shard_index": shard_index,
    }
    if actual != expected:
        raise ValueError("formal rollout shard differs from frozen plan")
    return {
        "plan": plan, "ledger": ledger,
        "expected_manifest_sha256": plan["benchmarks"][benchmark][
            "manifest_sha256"
        ],
    }
