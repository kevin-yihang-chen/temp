from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from beyond_entropy.refocus_chart_audit import STRUCTURAL_METADATA_FIELDS


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
    assert "paired_signed_credit" in worker
    assert "runtime dataset audit contract failed" in worker
    assert "runtime audit provenance contract failed" in worker
    assert "exactly four visible H800 GPUs" in worker
    assert "latest_checkpointed_iteration.txt" in worker
    assert "unset HF_TOKEN HUGGINGFACE_HUB_TOKEN" in worker
    assert "git status --porcelain --untracked-files=all" in worker
    assert "jq_bin=/userhome/cs3/yihangc/anaconda3/bin/jq" in worker
    assert "sbatch --test-only --export=NONE" in submit
    assert "show-cpu-gpu-quota" in submit
    assert "a paired-signed G1 job is already queued or running" in submit
    assert "jq_bin=/userhome/cs3/yihangc/anaconda3/bin/jq" in submit
