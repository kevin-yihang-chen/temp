from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_script() -> ModuleType:
    path = ROOT / "scripts" / "smoke_vtool_hf_actor_load.py"
    spec = importlib.util.spec_from_file_location("smoke_vtool_hf_actor_load", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_actor_load_runtime_config_freezes_sdpa_without_remove_padding() -> None:
    smoke = _load_script()
    config = json.loads(
        (ROOT / "configs" / "vtool_action_credit_g1_v1.json").read_text(
            encoding="utf-8"
        )
    )
    frozen = smoke.validate_actor_runtime_config(config)
    assert frozen["attention_implementation"] == "sdpa"
    assert frozen["use_remove_padding"] is False
    assert frozen["dtype"] == "bfloat16"

    config["training"]["actor_attention_implementation"] = "flash_attention_2"
    with pytest.raises(ValueError, match="must be sdpa"):
        smoke.validate_actor_runtime_config(config)
    config["training"]["actor_attention_implementation"] = "sdpa"
    config["training"]["actor_use_remove_padding"] = True
    with pytest.raises(ValueError, match="remove-padding"):
        smoke.validate_actor_runtime_config(config)


def test_actor_load_slurm_contract_is_single_h800_notified_and_fail_closed() -> None:
    worker_path = ROOT / "scripts" / "slurm_vtool_hf_actor_load_h800.sh"
    submit_path = ROOT / "scripts" / "submit_vtool_hf_actor_load_h800.sh"
    import subprocess

    subprocess.run(["bash", "-n", str(worker_path)], check=True)
    subprocess.run(["bash", "-n", str(submit_path)], check=True)
    worker = worker_path.read_text(encoding="utf-8")
    submit = submit_path.read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:h800:1" in worker
    assert "#SBATCH --time=00:30:00" in worker
    assert "#SBATCH --mail-user=yihangc@connect.hku.hk" in worker
    assert "#SBATCH --mail-type=ALL" in worker
    assert 'actor_attention_implementation == "sdpa"' in worker
    assert "actor_use_remove_padding == false" in worker
    assert "vtool_hf_actor_gpu_forward_smoke_passed" in worker
    assert "model_weights_loaded == true" in worker
    assert "optimizer_step_performed == false" in worker
    assert "trap write_worker_status EXIT" in worker
    assert "unset HF_TOKEN HUGGINGFACE_HUB_TOKEN" in worker
    assert "git status --porcelain --untracked-files=all" in worker
    assert 'git -C "${runtime}" status --short' in worker
    assert "sbatch --test-only --export=NONE" in submit
    assert "show-cpu-gpu-quota" in submit
    assert "already queued or running" in submit
