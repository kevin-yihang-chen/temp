from pathlib import Path

import pytest

from beyond_entropy.phase_c_formal_transaction import (
    authorize_formal_rollout_shard,
    build_phase_c_formal_plan,
    start_formal_access,
    validate_formal_config,
)
from beyond_entropy.phase_c_training import BENCHMARKS, METHODS, SEEDS
from beyond_entropy.predictability_matrix_artifacts import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_formal_config_and_slurm_contract() -> None:
    import json

    config = json.loads((ROOT / "configs/factorized_phase_c_formal_v1.json").read_text())
    validate_formal_config(config)
    worker = (ROOT / "scripts/slurm_factorized_phase_c_formal.sh").read_text()
    submitter = (ROOT / "scripts/submit_factorized_phase_c_formal.sh").read_text()
    smoke_worker = (
        ROOT / "scripts/slurm_factorized_phase_c_formal_smoke.sh"
    ).read_text()
    smoke_submitter = (
        ROOT / "scripts/submit_factorized_phase_c_formal_smoke.sh"
    ).read_text()
    assert "#SBATCH --gres=gpu:rtx_4090:4" in worker
    assert "#SBATCH --mail-type=ALL" in worker
    assert "#SBATCH --no-requeue" in worker
    assert "--mail-type=ALL --no-requeue" in submitter
    assert "yihangc@connect.hku.hk" in submitter
    assert "#SBATCH --gres=gpu:rtx_4090:1" in smoke_worker
    assert "#SBATCH --mail-type=ALL" in smoke_worker
    assert "#SBATCH --no-requeue" in smoke_worker
    assert "--mail-type=ALL --no-requeue" in smoke_submitter
    assert "job-${SLURM_JOB_ID}/report.json" in smoke_worker


def test_formal_scorer_recreates_sparse_trainable_topology() -> None:
    source = (ROOT / "scripts/score_factorized_phase_c_formal.py").read_text()
    assert "train_backbone=True" in source
    assert "model.requires_grad_(False)" in source


def _synthetic_plan(tmp_path: Path) -> tuple[Path, dict]:
    import json
    import subprocess

    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}\n")
    hashed = {"path": str(evidence), "sha256": sha256_file(evidence)}
    transaction = tmp_path / "transaction"
    generation = {
        "generation_seeds": [0], "shard_count": 4, "dtype": "bfloat16",
        "attention_implementation": "sdpa", "max_new_tokens": 16,
        "min_pixels": 200704, "max_pixels": 602112,
    }
    plan = {
        "schema": "factorized_phase_c_formal_plan_v1",
        "one_shot": True, "test_authorized": True,
        "deadline_hkt": "2026-09-13T23:59:00+08:00", "created_at_utc": "now",
        "code_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "code_hashes": {}, "config": hashed, "allocation_report": hashed,
        "training_matrix": hashed,
        "model": "model", "model_revision": "revision",
        "methods": list(METHODS), "seeds": list(SEEDS),
        "benchmarks": {
            benchmark: {
                "states": 4, "manifest": str(tmp_path / f"{benchmark}.jsonl"),
                "manifest_sha256": "manifest-hash",
                "rollout_root": str(transaction / "rollouts" / benchmark),
                "merged_output": str(transaction / "rollouts" / benchmark / "merged"),
                "shards": {
                    str(index): str(
                        transaction / "rollouts" / benchmark
                        / f"shard-{index:05d}-of-00004" / "rollouts.jsonl"
                    )
                    for index in range(4)
                },
            }
            for benchmark in BENCHMARKS
        },
        "generation": generation,
        "policy": {
            "rates": [0, .1, .25, .5, .75, 1],
            "lambdas": [0, .025, .05, .1, .2],
        },
        "baselines": {"uncertainty": ["entropy", "confidence", "margin"]},
        "ablations": {}, "go_rule": {},
        "selectors": {
            method: {
                str(seed): {
                    "selector": hashed, "training_report": hashed,
                    "monitor_evaluation": hashed,
                }
                for seed in SEEDS
            }
            for method in METHODS
        },
        "transaction_root": str(transaction),
        "access_ledger": str(transaction / "access-ledger.json"),
        "predictions": {
            str(seed): str(transaction / "predictions" / f"seed-{seed}.json")
            for seed in SEEDS
        },
        "evaluation_output": str(transaction / "evaluation"),
        "runtime_smoke": hashed,
        "post_access_changes_forbidden": ["method", "seed", "threshold"],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan, sort_keys=True) + "\n")
    return path, plan


def test_access_ledger_precedes_and_strictly_authorizes_formal_shard(tmp_path) -> None:
    path, plan = _synthetic_plan(tmp_path)
    digest = sha256_file(path)
    ledger = start_formal_access(path, digest, ROOT)
    assert ledger["heldout_outcomes_now_considered_opened"] is True
    generation = plan["generation"]
    authorized = authorize_formal_rollout_shard(
        plan_path=path, expected_plan_sha256=digest,
        ledger_path=plan["access_ledger"], benchmark="chartqa",
        manifest_path=plan["benchmarks"]["chartqa"]["manifest"],
        output_path=plan["benchmarks"]["chartqa"]["shards"]["0"],
        model=plan["model"], model_revision=plan["model_revision"],
        generation_seeds=[0], code_revision=plan["code_revision"],
        dtype=generation["dtype"],
        attention_implementation=generation["attention_implementation"],
        max_new_tokens=generation["max_new_tokens"],
        min_pixels=generation["min_pixels"], max_pixels=generation["max_pixels"],
        shard_count=4, shard_index=0,
    )
    assert authorized["expected_manifest_sha256"] == "manifest-hash"
    with pytest.raises(ValueError, match="differs from frozen plan"):
        authorize_formal_rollout_shard(
            plan_path=path, expected_plan_sha256=digest,
            ledger_path=plan["access_ledger"], benchmark="chartqa",
            manifest_path=plan["benchmarks"]["chartqa"]["manifest"],
            output_path=plan["benchmarks"]["chartqa"]["shards"]["0"],
            model=plan["model"], model_revision=plan["model_revision"],
            generation_seeds=[0], code_revision=plan["code_revision"],
            dtype=generation["dtype"],
            attention_implementation=generation["attention_implementation"],
            max_new_tokens=generation["max_new_tokens"],
            min_pixels=generation["min_pixels"], max_pixels=generation["max_pixels"],
            shard_count=4, shard_index=1,
        )


def test_formal_access_is_irreversible(tmp_path) -> None:
    path, _ = _synthetic_plan(tmp_path)
    digest = sha256_file(path)
    start_formal_access(path, digest, ROOT)
    with pytest.raises(FileExistsError):
        start_formal_access(path, digest, ROOT)


def test_plan_freeze_does_not_read_or_hash_heldout_manifest_bytes(tmp_path) -> None:
    import json
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=repo, check=True
    )
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
    (repo / "src/beyond_entropy").mkdir(parents=True)
    (repo / "src/beyond_entropy/dummy.py").write_text("VALUE = 1\n")
    (repo / "scripts").mkdir()
    for name in script_names:
        (repo / "scripts" / name).write_text(f"# {name}\n")

    matrix = repo / "configs/training.json"
    matrix.parent.mkdir()
    matrix.write_text("{}\n")
    allocation = repo / "data/allocation.json"
    allocation.parent.mkdir()
    sealed_hash = "sealed-manifest-hash-not-recomputed-before-ledger"
    allocation.write_text(json.dumps({
        "selection_used_model_outcomes": False,
        "heldout_sequential_outcomes_opened": False,
        "benchmarks": {
            name: {"heldout": {"states": 1, "manifest_sha256": sealed_hash}}
            for name in BENCHMARKS
        },
    }))
    manifests = {}
    for name in BENCHMARKS:
        path = repo / f"data/{name}/heldout/manifest.jsonl"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"\xff intentionally sealed and invalid JSON")
        manifests[name] = str(path.relative_to(repo))

    seed_jobs = {17: "101", 29: "102", 47: "103"}
    training_root = repo / "artifacts/training"
    required_checks = {
        name: True for name in (
            "paired_reward_gain_contract", "binary_action_support_valid",
            "finite_trace", "all_trainable_groups_received_gradient",
            "all_trainable_groups_updated", "no_proposed_crop_execution",
            "finite_nonconstant_validation_scores",
        )
    }
    for seed, job in seed_jobs.items():
        for method in METHODS:
            run = training_root / method.replace("_", "-") / f"job-{job}"
            run.mkdir(parents=True)
            selector = run / "selector.pt"
            selector.write_bytes(f"selector-{method}-{seed}".encode())
            report = {
                "schema": "cv_method_post_training_report_v1",
                "stage": "phase_c_training", "method": method,
                "test_accessed": False, "formal_claim_eligible": False,
                "provenance": {"config": {
                    "seed": seed, "matrix_config_sha256": sha256_file(matrix),
                }},
                "selector_sha256": sha256_file(selector),
                "schedule_sha256": f"schedule-{seed}", "checks": required_checks,
            }
            (run / "report.json").write_text(json.dumps(report))
        evaluation = training_root / "evaluation" / f"job-{job}"
        evaluation.mkdir(parents=True)
        (evaluation / "report.json").write_text(json.dumps({
            "stage": "phase_c_training", "decision": "PHASE_C_SEED_FROZEN",
            "test_accessed": False, "formal_claim_eligible": False,
        }))

    config = json.loads((ROOT / "configs/factorized_phase_c_formal_v1.json").read_text())
    config["allocation_report"] = {
        "path": str(allocation.relative_to(repo)), "sha256": sha256_file(allocation),
    }
    config["training_matrix"] = {
        "path": str(matrix.relative_to(repo)), "sha256": sha256_file(matrix),
    }
    config["training_output_root"] = str(training_root.relative_to(repo))
    config["formal_output_root"] = "artifacts/formal"
    config["benchmarks"] = {
        name: {"states": 1, "manifest": manifests[name], "manifest_sha256": sealed_hash}
        for name in BENCHMARKS
    }
    config_path = repo / "configs/formal.json"
    config_path.write_text(json.dumps(config))
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip()

    smoke_path = repo / "artifacts/runtime-smoke/report.json"
    smoke_path.parent.mkdir(parents=True)
    smoke_seed = 17
    smoke_selectors = {}
    for method in METHODS:
        run = training_root / method.replace("_", "-") / f"job-{seed_jobs[smoke_seed]}"
        smoke_selectors[method] = {str(smoke_seed): {
            "selector": {
                "path": str(run / "selector.pt"),
                "sha256": sha256_file(run / "selector.pt"),
            },
            "training_report": {
                "path": str(run / "report.json"),
                "sha256": sha256_file(run / "report.json"),
            },
        }}
    smoke_path.write_text(json.dumps({
        "schema": "factorized_phase_c_formal_runtime_smoke_v1",
        "completed": True, "test_accessed": False,
        "formal_claim_eligible": False, "code_revision": revision,
        "seed": smoke_seed,
        "selector_training_job_id": seed_jobs[smoke_seed],
        "matrix": {"sha256": sha256_file(matrix)},
        "selectors": smoke_selectors,
        "checks": {"real_model_load": True, "no_proposed_crop_execution": True},
    }))
    result = build_phase_c_formal_plan(
        config_path=config_path, repository_root=repo, seed_jobs=seed_jobs,
        transaction_id="one-shot", output_path=repo / "plan.json",
        runtime_smoke_path=smoke_path,
    )
    assert result["code_revision"] == revision
    assert result["benchmarks"]["chartqa"]["manifest_sha256"] == sealed_hash
    assert result["runtime_smoke"]["sha256"] == sha256_file(smoke_path)
