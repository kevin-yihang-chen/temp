from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import ModuleType
from typing import Any

import pytest

from beyond_entropy.refocus_chart_audit import STRUCTURAL_METADATA_FIELDS
from beyond_entropy.counterfactual_action_credit import (
    CounterfactualActionPair,
    CounterfactualArmOutcome,
)
from beyond_entropy.vtool_action_credit import (
    ACTION_CREDIT_KEY,
    ACTION_TOKEN_COUNT_KEY,
    ANSWER_TOKEN_COUNT_KEY,
    OBSERVATION_TOKEN_COUNT_KEY,
    PAIR_VALID_KEY,
    TRAJECTORY_ID_KEY,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {relative}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_dataset_audit_pure_contracts() -> None:
    audit = _load_script(
        "audit_refocus_g1_runtime_dataset",
        "scripts/audit_refocus_g1_runtime_dataset.py",
    )
    assert audit.nearest_rank_percentile([4, 1, 3, 2], 0.5) == 2
    assert audit.nearest_rank_percentile([4, 1, 3, 2], 0.95) == 4
    metadata: dict[str, object] = {field: [] for field in STRUCTURAL_METADATA_FIELDS}
    assert audit.validate_tool_metadata(json.dumps(metadata)) == metadata
    metadata["focus_areas_bbox"] = []
    with pytest.raises(ValueError, match="non-structural"):
        audit.validate_tool_metadata(json.dumps(metadata))


def test_g1_launcher_builds_only_frozen_signed_smoke_command(tmp_path: Path) -> None:
    launcher = _load_script(
        "run_vtool_action_credit_g1",
        "scripts/run_vtool_action_credit_g1.py",
    )
    config = json.loads(
        (ROOT / "configs" / "vtool_action_credit_g1_v1.json").read_text(
            encoding="utf-8"
        )
    )
    arm = config["arms"]["paired_signed_credit"]
    validated = {
        "arm": arm,
        "train_path": ROOT / config["data"]["train"]["paired_path"],
        "curve_path": ROOT / config["data"]["curve_eval"]["paired_path"],
        "model": Path(config["model"]["local_snapshot"]),
        "runtime": Path(config["runtime"]["worktree"]),
    }
    command = launcher.build_command(
        config,
        validated,
        arm_name="paired_signed_credit",
        output_dir=tmp_path,
    )
    overrides = set(command[5:])
    assert "+algorithm.action_credit.enabled=true" in overrides
    assert "+algorithm.action_credit.mode=signed" in overrides
    assert (
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa" in overrides
    )
    assert "actor_rollout_ref.model.use_remove_padding=false" in overrides
    assert "actor_rollout_ref.actor.ppo_mini_batch_size=8" in overrides
    assert "actor_rollout_ref.rollout.n=4" in overrides
    assert "actor_rollout_ref.rollout.max_num_seqs=16" in overrides
    assert "+actor_rollout_ref.rollout.limit_images=2" in overrides
    assert "reward.custom_reward_function.name=compute_score" in overrides
    assert "trainer.n_gpus_per_node=4" in overrides
    assert "trainer.total_training_steps=2" in overrides
    assert "trainer.save_freq=2" in overrides
    assert "trainer.max_actor_ckpt_to_keep=1" in overrides
    assert "trainer.test_freq=-1" in overrides
    assert "trainer.val_before_train=false" in overrides
    assert not any("test.parquet" in value for value in overrides)


def test_g1_launcher_environment_removes_credentials_and_proxies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load_script(
        "run_vtool_action_credit_g1_env",
        "scripts/run_vtool_action_credit_g1.py",
    )
    for name in ("HF_TOKEN", "OPENAI_API_KEY", "HTTP_PROXY", "HTTPS_PROXY"):
        monkeypatch.setenv(name, "must-not-survive")
    runtime = Path("/tmp/runtime-for-environment-test")
    env = launcher.sanitized_environment(runtime)
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["PYTHONHASHSEED"] == "0"
    assert str(runtime) in env["PYTHONPATH"]
    assert not (
        {"HF_TOKEN", "OPENAI_API_KEY", "HTTP_PROXY", "HTTPS_PROXY"} & env.keys()
    )
    assert os.environ["HF_TOKEN"] == "must-not-survive"


def test_g1_rollout_analyzer_preserves_pairs_and_applies_tool_rate_stop(
    tmp_path: Path,
) -> None:
    analyzer = _load_script(
        "analyze_vtool_action_credit_g1",
        "scripts/analyze_vtool_action_credit_g1.py",
    )
    config_path = ROOT / "configs" / "vtool_action_credit_g1_v1.json"

    def digest(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def row(step: int, index: int, *, tool: bool) -> dict[str, Any]:
        trajectory_id = digest(f"trajectory-{step}-{index}")
        if tool:
            shared: dict[str, Any] = {
                "prefix_sha256": digest(f"prefix-{step}-{index}"),
                "action_sha256": digest(f"action-{step}-{index}"),
                "target_sha256": digest("target"),
                "policy_sha256": digest("policy"),
                "decoding_sha256": digest("decoding"),
                "scorer_sha256": digest("scorer"),
                "continuation_seed": step * 1000 + index,
            }
            pair = CounterfactualActionPair(
                trajectory_id=trajectory_id,
                factual=CounterfactualArmOutcome(
                    branch_id="factual",
                    observation_sha256=digest(f"factual-{step}-{index}"),
                    task_score=1.0,
                    action_cost=1.0,
                    **shared,
                ),
                counterfactual=CounterfactualArmOutcome(
                    branch_id="counterfactual",
                    observation_sha256=digest(f"counterfactual-{step}-{index}"),
                    task_score=0.0,
                    action_cost=0.0,
                    **shared,
                ),
            )
            audit = {
                "schema": analyzer.ROLLOUT_AUDIT_SCHEMA,
                "vtool_tool_attempted": True,
                "vtool_tool_success": True,
                "vtool_final_response_text": "FINAL ANSWER: 7 TERMINATE",
                "vtool_counterfactual_response_text": "FINAL ANSWER: 3 TERMINATE",
                TRAJECTORY_ID_KEY: trajectory_id,
                ACTION_TOKEN_COUNT_KEY: 2,
                OBSERVATION_TOKEN_COUNT_KEY: 2,
                ANSWER_TOKEN_COUNT_KEY: 2,
                ACTION_CREDIT_KEY: pair.action_credit,
                PAIR_VALID_KEY: True,
                "vtool_action_credit_pair": pair.to_dict(),
                "vtool_counterfactual_generation_seconds": 1.0,
            }
        else:
            audit = {
                "schema": analyzer.ROLLOUT_AUDIT_SCHEMA,
                "vtool_tool_attempted": False,
                "vtool_tool_success": False,
                "vtool_final_response_text": "FINAL ANSWER: 7 TERMINATE",
                "vtool_counterfactual_response_text": None,
                TRAJECTORY_ID_KEY: trajectory_id,
                ACTION_TOKEN_COUNT_KEY: 0,
                OBSERVATION_TOKEN_COUNT_KEY: 0,
                ANSWER_TOKEN_COUNT_KEY: 1,
                ACTION_CREDIT_KEY: 0.0,
                PAIR_VALID_KEY: False,
                "vtool_action_credit_pair": None,
                "vtool_counterfactual_generation_seconds": 0.0,
            }
        return {
            "input": "question",
            "output": audit["vtool_final_response_text"],
            "gts": "7",
            "score": 1.0,
            "acc": 1.0,
            "step": step,
            analyzer.ROLLOUT_AUDIT_JSON_KEY: json.dumps(audit, sort_keys=True),
        }

    def write_rollouts(directory: Path, *, tools_per_step: int) -> None:
        directory.mkdir()
        for step in (1, 2):
            rows = [
                row(step, index, tool=index < tools_per_step) for index in range(32)
            ]
            (directory / f"{step}.jsonl").write_text(
                "\n".join(json.dumps(value, sort_keys=True) for value in rows) + "\n",
                encoding="utf-8",
            )

    passing_dir = tmp_path / "passing"
    write_rollouts(passing_dir, tools_per_step=1)
    report, exit_code = analyzer.analyze_rollouts(
        rollout_dir=passing_dir,
        config_path=config_path,
        expected_arm="paired_signed_credit",
    )
    assert exit_code == 0
    assert report["decision"] == "paired_signed_g1_smoke_gate_passed"
    assert report["tool_call_rate"] == pytest.approx(2 / 64)
    assert report["mean_signed_action_credit"] == pytest.approx(0.95)
    assert report["pair_mismatch_count"] == 0
    assert report["judge_failure_count"] == 0

    stopped_dir = tmp_path / "stopped"
    write_rollouts(stopped_dir, tools_per_step=0)
    stopped, stopped_exit_code = analyzer.analyze_rollouts(
        rollout_dir=stopped_dir,
        config_path=config_path,
        expected_arm="paired_signed_credit",
    )
    assert stopped_exit_code == 0
    assert stopped["decision"] == "paired_signed_g1_stop_rule_triggered"
    assert stopped["stop_reasons"] == ["tool_call_rate_below_frozen_threshold"]


def test_g1_resolved_config_audit_rejects_scientific_drift(tmp_path: Path) -> None:
    yaml = pytest.importorskip("yaml")
    launcher = _load_script(
        "run_vtool_action_credit_g1_resolved",
        "scripts/run_vtool_action_credit_g1.py",
    )
    config = json.loads(
        (ROOT / "configs" / "vtool_action_credit_g1_v1.json").read_text(
            encoding="utf-8"
        )
    )
    arm = config["arms"]["paired_signed_credit"]
    train_path = ROOT / config["data"]["train"]["paired_path"]
    curve_path = ROOT / config["data"]["curve_eval"]["paired_path"]
    model_path = Path(config["model"]["local_snapshot"])
    resolved: dict[str, Any] = {
        "data": {
            "train_files": str(train_path),
            "val_files": str(curve_path),
            "train_batch_size": 8,
            "max_prompt_length": 4096,
            "max_response_length": 1024,
            "shuffle": True,
            "filter_overlong_prompts": False,
            "truncation": "error",
            "trust_remote_code": False,
        },
        "actor_rollout_ref": {
            "model": {
                "path": str(model_path),
                "use_remove_padding": False,
                "override_config": {"attn_implementation": "sdpa"},
                "enable_gradient_checkpointing": True,
            },
            "actor": {
                "strategy": "fsdp2",
                "fsdp_config": {
                    "strategy": "fsdp2",
                    "dtype": "bfloat16",
                    "param_offload": False,
                    "optimizer_offload": False,
                    "forward_prefetch": True,
                    "use_torch_compile": True,
                },
                "freeze_vision_tower": False,
                "optim": {"lr": 1e-6},
                "ppo_mini_batch_size": 8,
                "ppo_micro_batch_size_per_gpu": 1,
                "ppo_epochs": 1,
                "shuffle": False,
                "use_dynamic_bsz": False,
                "use_kl_loss": False,
            },
            "rollout": {
                "n": 4,
                "dtype": "bfloat16",
                "gpu_memory_utilization": 0.5,
                "enforce_eager": True,
                "max_num_batched_tokens": 8192,
                "max_model_len": 5120,
                "max_num_seqs": 16,
                "limit_images": 2,
                "enable_chunked_prefill": True,
                "enable_prefix_caching": False,
                "calculate_log_probs": True,
                "agent": {"default_agent_loop": "counterfactual_credit_vtool_agent"},
            },
        },
        "reward": {
            "custom_reward_function": {
                "path": "pkg://integrations.vtool_action_credit.paired_vtool",
                "name": "compute_score",
            }
        },
        "algorithm": {
            "adv_estimator": "grpo",
            "norm_adv_by_std_in_grpo": True,
            "use_kl_in_reward": False,
            "action_credit": {"enabled": True, "mode": "signed", "beta": 1.0},
        },
        "trainer": {
            "experiment_name": "paired_signed_credit",
            "n_gpus_per_node": 4,
            "total_training_steps": 2,
            "total_epochs": 1,
            "save_freq": 2,
            "max_actor_ckpt_to_keep": 1,
            "test_freq": -1,
            "val_before_train": False,
            "resume_mode": "disable",
            "default_local_dir": str(tmp_path / "checkpoints"),
            "rollout_data_dir": str(tmp_path / "rollouts"),
            "validation_data_dir": None,
        },
        "ray_kwargs": {"ray_init": {"num_cpus": 48}},
    }
    validated = {
        "arm": arm,
        "train_path": train_path,
        "curve_path": curve_path,
        "model": model_path,
    }
    checks = launcher.audit_resolved_config(
        yaml.safe_dump(resolved),
        config,
        validated,
        arm_name="paired_signed_credit",
        output_dir=tmp_path,
    )
    assert checks and all(checks.values())
    resolved["actor_rollout_ref"]["model"]["override_config"][
        "attn_implementation"
    ] = "flash_attention_2"
    with pytest.raises(ValueError, match="attn_implementation"):
        launcher.audit_resolved_config(
            yaml.safe_dump(resolved),
            config,
            validated,
            arm_name="paired_signed_credit",
            output_dir=tmp_path,
        )
    resolved["actor_rollout_ref"]["model"]["override_config"][
        "attn_implementation"
    ] = "sdpa"
    resolved["trainer"]["total_training_steps"] = 3
    with pytest.raises(ValueError, match="total_training_steps"):
        launcher.audit_resolved_config(
            yaml.safe_dump(resolved),
            config,
            validated,
            arm_name="paired_signed_credit",
            output_dir=tmp_path,
        )


def test_g1_slurm_contract_is_bounded_notified_and_fail_closed() -> None:
    worker_path = ROOT / "scripts" / "slurm_vtool_action_credit_g1_h800.sh"
    submit_path = ROOT / "scripts" / "submit_vtool_action_credit_g1_h800.sh"
    import subprocess

    subprocess.run(["bash", "-n", str(worker_path)], check=True)
    subprocess.run(["bash", "-n", str(submit_path)], check=True)
    worker = worker_path.read_text(encoding="utf-8")
    submit = submit_path.read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:h800:4" in worker
    assert "#SBATCH --cpus-per-task=48" in worker
    assert "#SBATCH --time=02:00:00" in worker
    assert "#SBATCH --mail-user=yihangc@connect.hku.hk" in worker
    assert "#SBATCH --mail-type=ALL" in worker
    assert ".resources.minimum_free_persistent_disk_gib == 64" in worker
    assert worker.count("minimum_free_kb=$((minimum_free_gib * 1024 * 1024))") == 1
    assert submit.count("minimum_free_kb=$((minimum_free_gib * 1024 * 1024))") == 1
    assert "requires at least ${minimum_free_gib} GiB free persistent disk" in worker
    assert "requires at least ${minimum_free_gib} GiB free persistent disk" in submit
    assert "paired_signed_credit" in worker
    assert "runtime dataset audit contract failed" in worker
    assert "runtime audit provenance contract failed" in worker
    assert "exactly four visible H800 GPUs" in worker
    assert "latest_checkpointed_iteration.txt" in worker
    assert "analyze_vtool_action_credit_g1.py" in worker
    assert "rollout-analysis.json" in worker
    assert "paired_signed_g1_stop_rule_triggered" in worker
    assert "smoke_vtool_action_credit_dataproto.py" in worker
    assert "vtool_action_credit_dataproto_chunk_passed" in worker
    assert '.training.actor_attention_implementation == "sdpa"' in worker
    assert ".training.actor_use_remove_padding == false" in worker
    assert "unset HF_TOKEN HUGGINGFACE_HUB_TOKEN" in worker
    assert "git status --porcelain --untracked-files=all" in worker
    assert "jq_bin=/userhome/cs3/yihangc/anaconda3/bin/jq" in worker
    assert "sbatch --test-only --export=NONE" in submit
    assert "show-cpu-gpu-quota" in submit
    assert "a paired-signed G1 job is already queued or running" in submit
    assert "smoke_vtool_action_credit_dataproto.py" in submit
    assert "vtool_action_credit_dataproto_chunk_passed" in submit
    assert "jq_bin=/userhome/cs3/yihangc/anaconda3/bin/jq" in submit


def test_g1_dataproto_chunk_smoke_uses_pinned_runtime() -> None:
    python_bin = Path(
        "/userhome/cs3/yihangc/anaconda3/envs/beyond-entropy-vtool-g1/bin/python"
    )
    runtime = Path("/userhome/cs3/yihangc/Documents/runtime/vtool-action-credit-g1")
    if not python_bin.is_file() or not runtime.is_dir():
        pytest.skip("pinned VTool G1 runtime is unavailable")

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(runtime), str(ROOT), str(ROOT / "src")])
    completed = subprocess.run(
        [
            str(python_bin),
            str(ROOT / "scripts" / "smoke_vtool_action_credit_dataproto.py"),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    report = json.loads(completed.stdout)
    assert report["decision"] == "vtool_action_credit_dataproto_chunk_passed"
    assert report["chunks"] == 4
    assert all(report["checks"].values())
    assert report["protected_split_contents_accessed"] is False
    assert report["model_weights_loaded"] is False


def test_g1_worker_jq_all_checks_predicate_handles_object_values() -> None:
    jq_bin = Path("/userhome/cs3/yihangc/anaconda3/bin/jq")
    if not jq_bin.is_file():
        pytest.skip("frozen jq executable is unavailable")

    worker = (ROOT / "scripts" / "slurm_vtool_action_credit_g1_h800.sh").read_text(
        encoding="utf-8"
    )
    predicate = ".checks | all(.[]; . == true)"
    assert worker.count(f"({predicate})") == 3

    for checks, expected_exit_code in (
        ({"first": True, "second": True}, 0),
        ({"first": True, "second": False}, 1),
    ):
        completed = subprocess.run(
            [str(jq_bin), "-e", f"({predicate})"],
            input=json.dumps({"checks": checks}),
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == expected_exit_code, completed.stderr
