from __future__ import annotations

import ast
from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess

import pytest

from beyond_entropy.vtool_action_credit import (
    ACTION_CREDIT_KEY,
    ACTION_TOKEN_COUNT_KEY,
    ANSWER_TOKEN_COUNT_KEY,
    OBSERVATION_TOKEN_COUNT_KEY,
    PAIR_VALID_KEY,
    TRAJECTORY_ID_KEY,
    ActionCreditTrajectory,
    deterministic_rollout_seeds,
    extract_vtool_answer,
    inject_token_local_action_credit,
    prepare_action_credit_batch,
    wrap_verl_compute_advantage,
)


def test_answer_extraction_and_rollout_seeds_are_frozen_and_deterministic() -> None:
    assert extract_vtool_answer("FINAL ANSWER: 17 TERMINATE") == "17"
    assert extract_vtool_answer("ANSWER: blue | rationale") == "blue"
    trajectory = {"step": 3, "sample_index": 8, "rollout_n": 1, "validate": False}
    assert deterministic_rollout_seeds(trajectory) == deterministic_rollout_seeds(
        trajectory
    )
    assert deterministic_rollout_seeds(trajectory)[1] == (
        deterministic_rollout_seeds(trajectory)[0] + 1
    )


def _tool(trajectory_id: str, credit: float) -> ActionCreditTrajectory:
    return ActionCreditTrajectory(
        trajectory_id=trajectory_id,
        action_token_count=2,
        observation_token_count=2,
        answer_token_count=3,
        action_credit=credit,
        pair_valid=True,
    )


def _no_tool(trajectory_id: str) -> ActionCreditTrajectory:
    return ActionCreditTrajectory(
        trajectory_id=trajectory_id,
        action_token_count=0,
        observation_token_count=0,
        answer_token_count=4,
        action_credit=0.0,
        pair_valid=False,
    )


def test_prepared_signed_batch_localizes_action_and_outcome_advantages() -> None:
    prepared = prepare_action_credit_batch(
        trajectories=(_tool("tool-a", 0.95), _no_tool("direct-b")),
        outcome_advantages=(-0.25, 0.5),
        response_length=8,
        mode="signed",
    )
    assert prepared.advantages[0] == pytest.approx(
        (0.95, 0.95, 0.0, 0.0, -0.25, -0.25, -0.25, 0.0)
    )
    assert prepared.advantages[1] == pytest.approx(
        (0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0)
    )
    assert prepared.policy_masks[0] == (1, 1, 0, 0, 1, 1, 1, 0)


def test_zero_and_shuffled_controls_change_only_action_credit() -> None:
    trajectories = (_tool("tool-b", -1.05), _tool("tool-a", 0.95))
    zero = prepare_action_credit_batch(
        trajectories=trajectories,
        outcome_advantages=(0.25, -0.5),
        response_length=7,
        mode="zero",
    )
    shuffled = prepare_action_credit_batch(
        trajectories=trajectories,
        outcome_advantages=(0.25, -0.5),
        response_length=7,
        mode="shuffled",
    )
    assert zero.advantages[0][:2] == (0.0, 0.0)
    assert zero.advantages[0][4:] == pytest.approx((0.25, 0.25, 0.25))
    assert shuffled.applied_action_credits == pytest.approx((0.95, -1.05))
    assert shuffled.donor_trajectory_ids == ("tool-a", "tool-b")
    assert shuffled.advantages[0][4:] == zero.advantages[0][4:]
    assert shuffled.advantages[1][4:] == zero.advantages[1][4:]


@pytest.mark.parametrize(
    "trajectories",
    [(_no_tool("direct-only"),), (_tool("single-tool", 0.95),)],
)
def test_shuffled_control_skips_action_loss_with_fewer_than_two_tool_pairs(
    trajectories: tuple[ActionCreditTrajectory, ...],
) -> None:
    prepared = prepare_action_credit_batch(
        trajectories=trajectories,
        outcome_advantages=(0.25,),
        response_length=8,
        mode="shuffled",
    )
    assert prepared.applied_action_credits == (0.0,)
    assert prepared.donor_trajectory_ids == (None,)


def test_runtime_reports_shuffled_batch_skip() -> None:
    data = _runtime_batch()
    keep = [0]
    data.batch = {key: value[keep] for key, value in data.batch.items()}
    data.non_tensor_batch = {
        key: [value[index] for index in keep]
        for key, value in data.non_tensor_batch.items()
    }
    _, metrics = inject_token_local_action_credit(data, mode="shuffled")
    assert metrics["action_credit/shuffled_batch_skipped"] == 1.0


def test_trajectory_contract_rejects_unpaired_tool_and_credited_direct_answer() -> None:
    with pytest.raises(ValueError, match="valid counterfactual pair"):
        ActionCreditTrajectory("bad-tool", 2, 1, 3, 0.5, False)
    with pytest.raises(ValueError, match="zero action credit"):
        ActionCreditTrajectory("bad-direct", 0, 0, 3, 0.5, False)


@dataclass
class _FakeData:
    batch: dict
    non_tensor_batch: dict
    meta_info: dict = field(default_factory=dict)


def _runtime_batch():
    torch = pytest.importorskip("torch")
    response_mask = torch.tensor([[1, 1, 0, 0, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0, 0, 0]])
    return _FakeData(
        batch={
            "advantages": torch.tensor(
                [
                    [-0.25, -0.25, 0.0, 0.0, -0.25, -0.25, -0.25, 0.0],
                    [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
            "returns": torch.zeros(2, 8),
            "response_mask": response_mask,
        },
        non_tensor_batch={
            TRAJECTORY_ID_KEY: ["tool-a", "direct-b"],
            ACTION_TOKEN_COUNT_KEY: [2, 0],
            OBSERVATION_TOKEN_COUNT_KEY: [2, 0],
            ANSWER_TOKEN_COUNT_KEY: [3, 4],
            ACTION_CREDIT_KEY: [0.95, 0.0],
            PAIR_VALID_KEY: [True, False],
        },
    )


def test_runtime_injection_creates_real_action_token_gradient() -> None:
    torch = pytest.importorskip("torch")
    data, metrics = inject_token_local_action_credit(_runtime_batch(), mode="signed")
    logits = torch.zeros(2, 8, requires_grad=True)
    log_probs = torch.nn.functional.logsigmoid(logits)
    loss = -(log_probs * data.batch["advantages"] * data.batch["response_mask"]).sum()
    loss.backward()
    gradient = logits.grad
    assert gradient is not None
    assert torch.all(gradient[0, :2] != 0)
    assert torch.all(gradient[0, 2:4] == 0)
    assert torch.all(gradient[0, 4:7] != 0)
    assert gradient[0, 7] == 0
    assert metrics["action_credit/tool_trajectory_rate"] == pytest.approx(0.5)


def test_runtime_injection_fails_if_upstream_still_masks_action_tokens() -> None:
    data = _runtime_batch()
    data.batch["response_mask"][0, :2] = 0
    with pytest.raises(ValueError, match="action tokens are still masked"):
        inject_token_local_action_credit(data, mode="signed")


def test_wrapper_runs_original_grpo_before_credit_injection() -> None:
    data = _runtime_batch()
    calls: list[str] = []

    def original(batch, *_args, **_kwargs):
        calls.append("grpo")
        return batch

    wrapped = wrap_verl_compute_advantage(original, mode="zero")
    result = wrapped(data)
    assert result is data
    assert calls == ["grpo"]
    assert (
        result.meta_info["action_credit_metrics"]["action_credit/tool_trajectory_count"]
        == 1.0
    )
    assert result.batch["advantages"][0, :2].tolist() == [0.0, 0.0]


def test_paired_vtool_overlay_freezes_same_prefix_and_union_mask_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    integration_path = root / "integrations" / "vtool_action_credit" / "paired_vtool.py"
    source = integration_path.read_text(encoding="utf-8")
    ast.parse(source)
    assert "factual_observation_ids != counterfactual_observation_ids" in source
    assert 'branch_sampling["seed"] = continuation_seed' in source
    assert "prompt_ids=branch_prompt_ids" in source
    assert "[1] * len(action_ids)" in source
    assert "[0] * len(observation_ids)" in source
    assert "action_cost=1.0" in source
    assert "action_cost=0.0" in source
    assert "def compute_score(" in source
    assert 'return {"score": score, "acc": score}' in source
    assert "SUCCESS_OBSERVATION" not in source
    assert "FAILURE_OBSERVATION" not in source

    patch = (
        root
        / "integrations"
        / "vtool_action_credit"
        / "vtool-training-v2-d2aa283.patch"
    ).read_text(encoding="utf-8")
    assert "_trajectory_info=trajectory" in patch
    assert "inject_token_local_action_credit" in patch
    assert "metrics.update(action_credit_metrics)" in patch


def test_g1_config_freezes_matched_controls_and_stop_rules() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs" / "vtool_action_credit_g1_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["schema"] == "vtool_action_credit_g1_config_v1"
    assert config["frozen_before_g1_results"] is True
    assert config["data"]["protected_split_contents_accessed"] is False
    assert config["data"]["train_curve_structural_overlap"] == 0
    assert config["controls"]["exact_compute_controls"] == [
        "paired_zero_credit",
        "paired_shuffled_credit",
    ]
    assert set(config["arms"]) == {
        "upstream_outcome_only",
        "paired_zero_credit",
        "paired_shuffled_credit",
        "paired_signed_credit",
    }
    assert config["arms"]["paired_signed_credit"]["action_credit"] == {
        "beta": 1.0,
        "enabled": True,
        "mode": "signed",
    }
    assert config["arms"]["paired_zero_credit"]["dataset_family"] == "paired"
    assert config["arms"]["paired_shuffled_credit"]["dataset_family"] == "paired"
    assert config["arms"]["upstream_outcome_only"]["agent_name"] == "vtool_agent"
    assert config["training"]["validation_before_train"] is False
    assert config["training"]["test_frequency_steps"] == -1
    assert config["training"]["total_optimizer_steps"] == 2
    assert config["training"]["actor_ppo_mini_batch_size"] == 8
    assert config["training"]["actor_ppo_micro_batch_size_per_gpu"] == 1
    assert config["training"]["actor_use_dynamic_batching"] is False
    assert config["training"]["actor_learning_rate"] == pytest.approx(1e-6)
    assert config["training"]["actor_ppo_epochs"] == 1
    assert config["training"]["actor_use_kl_loss"] is False
    assert config["training"]["actor_fsdp_strategy"] == "fsdp2"
    assert config["training"]["actor_shuffle"] is False
    assert config["training"]["data_shuffle"] is True
    assert config["training"]["grpo_normalize_outcome_advantage_by_std"] is True
    assert config["training"]["trainer_resume_mode"] == "disable"
    assert config["training"]["save_frequency_steps"] == 2
    assert config["training"]["max_actor_checkpoints_to_keep"] == 1
    assert config["training"]["rollout_max_num_seqs"] == 16
    assert config["training"]["rollout_limit_images"] == 2
    assert (
        config["training"]["actor_ppo_mini_batch_size"]
        * config["training"]["rollout_n"]
        // config["resources"]["gpu_count"]
        == config["training"]["data_train_batch_size"]
    )
    assert config["resources"]["gpu_count"] == 4
    assert config["resources"]["slurm_mail_type"] == "ALL"
    assert config["stop_rules"]["tool_call_rate_below"] == pytest.approx(0.01)
    assert config["preflight"]["action_gradient_report_sha256"] == (
        "3fa21b9e5343348cf4619066c08b08049b63fe96fa11a9151d3723fd5238d99b"
    )
    assert config["preflight"]["full_train_runtime_audit_report_sha256"] == (
        "e468c36719ee3e303ff84a98f6feac1b88ebee9f0fb2731be2a129204671c1ed"
    )
    assert config["preflight"]["model_load_report_sha256"] == (
        "1a67365b1fc1fbf76ad84c7954cdeeeb73f1fd1eeb52dd7d94971dd06469009a"
    )


def test_model_load_slurm_contract_binds_runtime_model_data_and_notifications() -> None:
    root = Path(__file__).resolve().parents[1]
    worker_path = root / "scripts" / "slurm_vtool_vllm_model_load_h800.sh"
    submit_path = root / "scripts" / "submit_vtool_vllm_model_load_h800.sh"
    subprocess.run(["bash", "-n", str(worker_path)], check=True)
    subprocess.run(["bash", "-n", str(submit_path)], check=True)
    worker = worker_path.read_text(encoding="utf-8")
    submit = submit_path.read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:h800:1" in worker
    assert "#SBATCH --mail-user=yihangc@connect.hku.hk" in worker
    assert "#SBATCH --mail-type=ALL" in worker
    assert "unset HF_TOKEN HUGGINGFACE_HUB_TOKEN" in worker
    assert 'require_hash "${config}" "${expected_config_sha256}" config' in worker
    assert 'git -C "${runtime}" rev-parse HEAD' in worker
    assert 'git -C "${runtime}" diff --unified=0' in worker
    assert "model-00001-of-00002.safetensors" in worker
    assert "model-00002-of-00002.safetensors" in worker
    assert 'config_sha256=$(sha256sum "${config}"' in submit
    assert "sbatch --test-only --export=NONE" in submit
