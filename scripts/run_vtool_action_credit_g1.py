#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "vtool_action_credit_g1_v1.json"
PROTOCOL_PATH = (
    REPO_ROOT
    / "artifacts"
    / "docvqa-train-factorized-v2"
    / "ops"
    / "vtool-counterfactual-action-credit-protocol-20260902-v1.md"
)
AMENDMENT_PATH = (
    REPO_ROOT
    / "artifacts"
    / "docvqa-train-factorized-v2"
    / "ops"
    / "vtool-action-credit-g1-amendment-20260902-v1.md"
)
RUNTIME_PATCH_RELATIVE = Path(
    "integrations/vtool_action_credit/vtool-training-v2-d2aa283.patch"
)
RUNTIME_PATCHED_FILES = (
    "verl/experimental/agent_loop/agent_loop.py",
    "verl/trainer/ppo/ray_trainer.py",
)
IMPLEMENTATION_PATHS = (
    Path("scripts/run_vtool_action_credit_g1.py"),
    Path("scripts/audit_refocus_g1_runtime_dataset.py"),
    Path("scripts/analyze_vtool_action_credit_g1.py"),
    Path("scripts/smoke_vtool_action_credit_gradient.py"),
    Path("src/beyond_entropy/vtool_action_credit.py"),
    Path("integrations/vtool_action_credit/paired_vtool.py"),
    Path("integrations/vtool_action_credit/agent.yaml"),
    RUNTIME_PATCH_RELATIVE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and launch one frozen action-credit G1 arm."
    )
    parser.add_argument(
        "--arm",
        required=True,
        choices=(
            "upstream_outcome_only",
            "paired_zero_credit",
            "paired_shuffled_credit",
            "paired_signed_credit",
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--mode", choices=("print", "hydra-dry-run", "execute"), required=True
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(runtime: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=runtime,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_status_lines(worktree: Path) -> list[str]:
    return subprocess.run(
        ["git", "status", "--short"],
        cwd=worktree,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()


def _require_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_hash(path: Path, expected: str, *, name: str) -> str:
    resolved = path.resolve(strict=True)
    actual = sha256_file(resolved)
    if actual != expected:
        raise ValueError(f"{name} SHA-256 mismatch: {actual} != {expected}")
    return actual


def _all_true_checks(payload: Mapping[str, Any]) -> bool:
    checks = payload.get("checks")
    return (
        isinstance(checks, Mapping)
        and bool(checks)
        and all(value is True for value in checks.values())
    )


def _require_preflight_report(
    preflight: Mapping[str, Any],
    *,
    path_key: str,
    hash_key: str,
    name: str,
) -> tuple[Mapping[str, Any], str]:
    path = _repo_path(preflight[path_key])
    digest = _require_hash(path, str(preflight[hash_key]), name=name)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must contain a JSON mapping")
    return payload, digest


def _repo_path(relative: object) -> Path:
    path = (REPO_ROOT / str(relative)).resolve(strict=True)
    if not path.is_relative_to(REPO_ROOT):
        raise ValueError(f"frozen path escapes repository: {path}")
    return path


def validate_frozen_inputs(
    config: Mapping[str, Any], *, arm_name: str
) -> dict[str, Any]:
    if config.get("schema") != "vtool_action_credit_g1_config_v1":
        raise ValueError("unsupported G1 config schema")
    if config.get("frozen_before_g1_results") is not True:
        raise ValueError("G1 config must attest it was frozen before results")

    arms = _require_mapping(config.get("arms"), name="arms")
    arm = _require_mapping(arms.get(arm_name), name=f"arms.{arm_name}")
    training = _require_mapping(config.get("training"), name="training")
    resources = _require_mapping(config.get("resources"), name="resources")
    data = _require_mapping(config.get("data"), name="data")
    train = _require_mapping(data.get("train"), name="data.train")
    curve_eval = _require_mapping(data.get("curve_eval"), name="data.curve_eval")
    runtime_config = _require_mapping(config.get("runtime"), name="runtime")
    protocol = _require_mapping(config.get("protocol"), name="protocol")
    model_config = _require_mapping(config.get("model"), name="model")
    preflight = _require_mapping(config.get("preflight"), name="preflight")

    gpu_count = int(resources["gpu_count"])
    train_batch_size = int(training["data_train_batch_size"])
    rollout_n = int(training["rollout_n"])
    mini_batch_size = int(training["actor_ppo_mini_batch_size"])
    if gpu_count != 4 or int(training["total_optimizer_steps"]) != 2:
        raise ValueError("G1 is restricted to four GPUs and two optimizer steps")
    if mini_batch_size > train_batch_size:
        raise ValueError("PPO mini-batch is prompt-count and cannot exceed train batch")
    if (train_batch_size * rollout_n) % gpu_count:
        raise ValueError("trajectory batch must divide evenly across G1 GPUs")
    local_trajectory_batch = train_batch_size * rollout_n // gpu_count
    local_mini_batch = mini_batch_size * rollout_n // gpu_count
    if local_mini_batch != local_trajectory_batch:
        raise ValueError("G1 must use one complete local trajectory mini-batch per GPU")
    rollout_max_num_seqs = int(training["rollout_max_num_seqs"])
    if not local_trajectory_batch <= rollout_max_num_seqs <= 32:
        raise ValueError(
            "G1 rollout concurrency must cover one local batch and stay bounded"
        )
    if int(training["rollout_limit_images"]) != 2:
        raise ValueError("paired G1 requires exactly two images per model request")

    dataset_family = str(arm["dataset_family"])
    if dataset_family not in {"paired", "outcome_only"}:
        raise ValueError("unsupported G1 dataset family")
    path_key = f"{dataset_family}_path"
    hash_key = f"{dataset_family}_sha256"
    train_path = _repo_path(train[path_key])
    curve_path = _repo_path(curve_eval[path_key])
    train_sha256 = _require_hash(
        train_path, str(train[hash_key]), name=f"{arm_name} train data"
    )
    curve_sha256 = _require_hash(
        curve_path, str(curve_eval[hash_key]), name=f"{arm_name} curve data"
    )

    runtime = Path(str(runtime_config["worktree"])).resolve(strict=True)
    if _git(runtime, "rev-parse", "HEAD") != str(runtime_config["upstream_commit"]):
        raise ValueError("pinned runtime commit mismatch")
    patch_path = _repo_path(RUNTIME_PATCH_RELATIVE)
    patch_sha256 = _require_hash(
        patch_path, str(runtime_config["patch_sha256"]), name="runtime patch"
    )
    runtime_diff = subprocess.run(
        ["git", "diff", "--unified=0", "--", *RUNTIME_PATCHED_FILES],
        cwd=runtime,
        check=True,
        capture_output=True,
    ).stdout
    runtime_diff_sha256 = hashlib.sha256(runtime_diff).hexdigest()
    if runtime_diff_sha256 != patch_sha256:
        raise ValueError("runtime worktree does not exactly match frozen patch")
    status = _git_status_lines(runtime)
    expected_status = [f" M {path}" for path in RUNTIME_PATCHED_FILES]
    if status != expected_status:
        raise ValueError(f"runtime has unexpected modifications: {status}")

    _require_hash(
        PROTOCOL_PATH,
        str(protocol["method_protocol_sha256"]),
        name="method protocol",
    )
    _require_hash(
        AMENDMENT_PATH,
        str(protocol["g1_amendment_sha256"]),
        name="G1 amendment",
    )
    model = Path(str(model_config["local_snapshot"])).resolve(strict=True)
    if model.name != str(model_config["revision"]):
        raise ValueError("local model snapshot revision mismatch")
    expected_weight_hashes = model_config.get("weight_blob_sha256")
    if not isinstance(expected_weight_hashes, list) or not all(
        isinstance(value, str) for value in expected_weight_hashes
    ):
        raise ValueError("model weight hashes must be a list of SHA-256 strings")
    weight_paths = sorted(model.glob("model-*-of-*.safetensors"))
    if len(weight_paths) != len(expected_weight_hashes):
        raise ValueError("model weight shard count does not match frozen config")
    actual_weight_hashes = [
        _require_hash(path, expected, name=f"model weight {path.name}")
        for path, expected in zip(weight_paths, expected_weight_hashes, strict=True)
    ]
    expected_environment = Path(str(runtime_config["environment"])).resolve(strict=True)
    if Path(sys.prefix).resolve() != expected_environment:
        raise ValueError(
            f"launcher must run inside frozen environment {expected_environment}"
        )
    environment_payload, environment_report_sha256 = _require_preflight_report(
        preflight,
        path_key="environment_report",
        hash_key="environment_report_sha256",
        name="environment report",
    )
    if not (
        environment_payload.get("decision")
        == "vtool_action_credit_g1_import_gate_passed"
        and _all_true_checks(environment_payload)
    ):
        raise ValueError("environment report semantic contract failed")
    fake_server_payload, fake_server_report_sha256 = _require_preflight_report(
        preflight,
        path_key="fake_server_report",
        hash_key="fake_server_report_sha256",
        name="fake-server report",
    )
    if not (
        fake_server_payload.get("decision")
        == "paired_vtool_fake_server_contract_passed"
        and fake_server_payload.get("protected_split_contents_accessed") is False
        and _all_true_checks(fake_server_payload)
    ):
        raise ValueError("fake-server report semantic contract failed")
    processor_payload, processor_report_sha256 = _require_preflight_report(
        preflight,
        path_key="processor_report",
        hash_key="processor_report_sha256",
        name="processor report",
    )
    if not (
        processor_payload.get("decision") == "refocus_g1_dataset_runtime_smoke_passed"
        and processor_payload.get("protected_split_contents_accessed") is False
        and processor_payload.get("model_weights_loaded") is False
        and _all_true_checks(processor_payload)
    ):
        raise ValueError("processor report semantic contract failed")
    model_load_payload, model_load_report_sha256 = _require_preflight_report(
        preflight,
        path_key="model_load_report",
        hash_key="model_load_report_sha256",
        name="model-load report",
    )
    if not (
        model_load_payload.get("decision") == "vtool_vllm_model_load_smoke_passed"
        and model_load_payload.get("model_revision") == str(model_config["revision"])
        and model_load_payload.get("dataset_sha256")
        == str(preflight["one_row_dataset_sha256"])
        and model_load_payload.get("protected_split_contents_accessed") is False
        and model_load_payload.get("optimizer_step_performed") is False
        and int(model_load_payload.get("prompt_tokens", 0)) > 0
        and int(model_load_payload.get("completion_tokens", 0)) > 0
    ):
        raise ValueError("model-load report semantic contract failed")
    full_train_payload, full_train_audit_sha256 = _require_preflight_report(
        preflight,
        path_key="full_train_runtime_audit_report",
        hash_key="full_train_runtime_audit_report_sha256",
        name="full-train runtime audit",
    )
    prompt_token_summary = full_train_payload.get("prompt_tokens")
    if not (
        full_train_payload.get("decision") == "refocus_g1_runtime_dataset_audit_passed"
        and full_train_payload.get("dataset_sha256") == train_sha256
        and full_train_payload.get("dataset_rows") == int(train["rows"])
        and full_train_payload.get("structural_groups")
        == int(train["structural_groups"])
        and full_train_payload.get("row_id_manifest_sha256")
        == str(train["row_id_manifest_sha256"])
        and full_train_payload.get("model_revision") == str(model_config["revision"])
        and full_train_payload.get("audit_script_sha256")
        == sha256_file(_repo_path("scripts/audit_refocus_g1_runtime_dataset.py"))
        and isinstance(prompt_token_summary, Mapping)
        and int(prompt_token_summary.get("max", 0))
        <= int(training["data_max_prompt_length"])
        and full_train_payload.get("protected_split_contents_accessed") is False
        and full_train_payload.get("model_weights_loaded") is False
        and _all_true_checks(full_train_payload)
    ):
        raise ValueError("full-train runtime audit semantic contract failed")
    action_gradient_payload, action_gradient_report_sha256 = _require_preflight_report(
        preflight,
        path_key="action_gradient_report",
        hash_key="action_gradient_report_sha256",
        name="action gradient report",
    )
    if not (
        action_gradient_payload.get("decision")
        == "token_local_action_credit_gradient_smoke_passed"
        and _all_true_checks(action_gradient_payload)
    ):
        raise ValueError("action gradient report semantic contract failed")

    return {
        "arm": dict(arm),
        "curve_path": curve_path,
        "curve_sha256": curve_sha256,
        "local_mini_batch": local_mini_batch,
        "local_trajectory_batch": local_trajectory_batch,
        "model": model,
        "model_weight_sha256": actual_weight_hashes,
        "full_train_runtime_audit_sha256": full_train_audit_sha256,
        "action_gradient_report_sha256": action_gradient_report_sha256,
        "environment_report_sha256": environment_report_sha256,
        "fake_server_report_sha256": fake_server_report_sha256,
        "model_load_report_sha256": model_load_report_sha256,
        "processor_report_sha256": processor_report_sha256,
        "runtime": runtime,
        "runtime_diff_sha256": runtime_diff_sha256,
        "train_path": train_path,
        "train_sha256": train_sha256,
    }


def _bool(value: object) -> str:
    if not isinstance(value, bool):
        raise ValueError(f"expected boolean config value, got {value!r}")
    return str(value).lower()


def build_command(
    config: Mapping[str, Any],
    validated: Mapping[str, Any],
    *,
    arm_name: str,
    output_dir: Path,
) -> list[str]:
    training = _require_mapping(config["training"], name="training")
    resources = _require_mapping(config["resources"], name="resources")
    seeds = _require_mapping(config["seeds"], name="seeds")
    arm = _require_mapping(validated["arm"], name="validated.arm")
    credit = _require_mapping(arm["action_credit"], name="arm.action_credit")
    runtime = Path(validated["runtime"])
    paired = str(arm["dataset_family"]) == "paired"
    agent_config = (
        REPO_ROOT / "integrations" / "vtool_action_credit" / "agent.yaml"
        if paired
        else runtime / "recipe" / "vtool" / "agent.yaml"
    ).resolve(strict=True)

    overrides = [
        f"data.train_files={validated['train_path']}",
        f"data.val_files={validated['curve_path']}",
        f"data.train_batch_size={training['data_train_batch_size']}",
        f"data.max_prompt_length={training['data_max_prompt_length']}",
        f"data.max_response_length={training['data_max_response_length']}",
        f"data.shuffle={_bool(training['data_shuffle'])}",
        f"data.seed={seeds['dataset_shuffle']}",
        "data.validation_shuffle=false",
        "data.filter_overlong_prompts=false",
        "data.truncation=error",
        "data.image_key=images",
        "data.trust_remote_code=false",
        "data.return_raw_chat=true",
        "data.return_multi_modal_inputs=false",
        f"actor_rollout_ref.model.path={validated['model']}",
        "actor_rollout_ref.model.use_remove_padding=true",
        f"actor_rollout_ref.model.enable_gradient_checkpointing={_bool(training['actor_gradient_checkpointing'])}",
        f"actor_rollout_ref.actor.strategy={training['actor_fsdp_strategy']}",
        f"actor_rollout_ref.actor.fsdp_config.strategy={training['actor_fsdp_strategy']}",
        f"actor_rollout_ref.actor.fsdp_config.dtype={training['actor_dtype']}",
        f"actor_rollout_ref.actor.fsdp_config.seed={seeds['global']}",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={_bool(training['actor_fsdp_param_offload'])}",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={_bool(training['actor_fsdp_optimizer_offload'])}",
        f"actor_rollout_ref.actor.fsdp_config.forward_prefetch={_bool(training['actor_fsdp_forward_prefetch'])}",
        f"actor_rollout_ref.actor.fsdp_config.use_torch_compile={_bool(training['actor_fsdp_use_torch_compile'])}",
        f"actor_rollout_ref.actor.freeze_vision_tower={_bool(training['actor_freeze_vision_tower'])}",
        f"actor_rollout_ref.actor.optim.lr={training['actor_learning_rate']}",
        f"actor_rollout_ref.actor.optim.weight_decay={training['actor_weight_decay']}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={training['actor_ppo_mini_batch_size']}",
        "actor_rollout_ref.actor.ppo_micro_batch_size=null",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={training['actor_ppo_micro_batch_size_per_gpu']}",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={training['actor_ppo_max_token_len_per_gpu']}",
        f"actor_rollout_ref.actor.ppo_epochs={training['actor_ppo_epochs']}",
        f"actor_rollout_ref.actor.shuffle={_bool(training['actor_shuffle'])}",
        f"actor_rollout_ref.actor.data_loader_seed={seeds['actor_data_loader']}",
        f"actor_rollout_ref.actor.use_dynamic_bsz={_bool(training['actor_use_dynamic_batching'])}",
        f"actor_rollout_ref.actor.use_kl_loss={_bool(training['actor_use_kl_loss'])}",
        f"actor_rollout_ref.actor.clip_ratio={training['actor_clip_ratio']}",
        f"actor_rollout_ref.actor.entropy_coeff={training['actor_entropy_coeff']}",
        "actor_rollout_ref.ref.log_prob_micro_batch_size=null",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={training['ref_log_prob_micro_batch_size_per_gpu']}",
        f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={training['ref_log_prob_max_token_len_per_gpu']}",
        "actor_rollout_ref.ref.fsdp_config.dtype=bfloat16",
        "actor_rollout_ref.ref.fsdp_config.param_offload=false",
        f"actor_rollout_ref.rollout.mode=async",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.dtype={training['actor_dtype']}",
        f"actor_rollout_ref.rollout.n={training['rollout_n']}",
        f"actor_rollout_ref.rollout.temperature={training['rollout_temperature']}",
        f"actor_rollout_ref.rollout.top_p={training['rollout_top_p']}",
        f"actor_rollout_ref.rollout.top_k={training['rollout_top_k']}",
        f"actor_rollout_ref.rollout.do_sample={_bool(training['rollout_do_sample'])}",
        f"actor_rollout_ref.rollout.ignore_eos={_bool(training['rollout_ignore_eos'])}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={training['rollout_tensor_model_parallel_size']}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={training['rollout_gpu_memory_utilization']}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={training['rollout_max_num_batched_tokens']}",
        f"actor_rollout_ref.rollout.max_model_len={training['rollout_max_model_len']}",
        f"actor_rollout_ref.rollout.max_num_seqs={training['rollout_max_num_seqs']}",
        f"+actor_rollout_ref.rollout.limit_images={training['rollout_limit_images']}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={training['rollout_log_prob_micro_batch_size_per_gpu']}",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size=null",
        f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={training['rollout_log_prob_max_token_len_per_gpu']}",
        f"actor_rollout_ref.rollout.calculate_log_probs={_bool(training['rollout_calculate_log_probs'])}",
        f"actor_rollout_ref.rollout.enforce_eager={_bool(training['rollout_enforce_eager'])}",
        f"actor_rollout_ref.rollout.enable_chunked_prefill={_bool(training['rollout_enable_chunked_prefill'])}",
        f"actor_rollout_ref.rollout.enable_prefix_caching={_bool(training['rollout_enable_prefix_caching'])}",
        "actor_rollout_ref.rollout.free_cache_engine=true",
        "actor_rollout_ref.rollout.load_format=dummy",
        "actor_rollout_ref.rollout.skip_tokenizer_init=false",
        f"actor_rollout_ref.rollout.agent.num_workers={training['rollout_agent_workers']}",
        f"actor_rollout_ref.rollout.agent.default_agent_loop={arm['agent_name']}",
        f"actor_rollout_ref.rollout.agent.agent_loop_config_path={agent_config}",
        "reward.custom_reward_function.path=pkg://integrations.vtool_action_credit.paired_vtool",
        "reward.custom_reward_function.name=compute_score",
        "algorithm.adv_estimator=grpo",
        f"algorithm.norm_adv_by_std_in_grpo={_bool(training['grpo_normalize_outcome_advantage_by_std'])}",
        "algorithm.use_kl_in_reward=false",
        "+algorithm.action_credit.enabled=" + _bool(credit["enabled"]),
        f"+algorithm.action_credit.mode={credit['mode']}",
        f"+algorithm.action_credit.beta={credit['beta']}",
        f"trainer.balance_batch={_bool(training['trainer_balance_batch'])}",
        "trainer.critic_warmup=0",
        'trainer.logger=["console"]',
        "trainer.project_name=beyond_entropy_vtool_g1",
        f"trainer.experiment_name={arm_name}",
        f"trainer.n_gpus_per_node={resources['gpu_count']}",
        "trainer.nnodes=1",
        "trainer.use_legacy_worker_impl=auto",
        f"trainer.save_freq={training['save_frequency_steps']}",
        f"trainer.max_actor_ckpt_to_keep={training['max_actor_checkpoints_to_keep']}",
        f"trainer.test_freq={training['test_frequency_steps']}",
        f"trainer.total_training_steps={training['total_optimizer_steps']}",
        "trainer.total_epochs=1",
        f"trainer.val_before_train={_bool(training['validation_before_train'])}",
        f"trainer.resume_mode={training['trainer_resume_mode']}",
        f"trainer.default_local_dir={output_dir / 'checkpoints'}",
        f"trainer.rollout_data_dir={output_dir / 'rollouts'}",
        "trainer.validation_data_dir=null",
        "trainer.log_val_generations=0",
        "trainer.ray_wait_register_center_timeout=600",
        "ray_kwargs.ray_init.num_cpus=48",
        f"hydra.run.dir={output_dir / 'hydra'}",
        "hydra.output_subdir=null",
    ]
    return [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        f"--config-path={runtime / 'recipe' / 'vtool'}",
        "--config-name=refocus_multiturn_grpo",
        *overrides,
    ]


def sanitized_environment(runtime: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": f"{runtime}:{REPO_ROOT}:{REPO_ROOT / 'src'}",
            "HF_HOME": "/userhome/cs3/yihangc/Data/hf_cache",
            "HF_HUB_CACHE": "/userhome/cs3/yihangc/Data/hf_cache",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_USE_V1": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "HYDRA_FULL_ERROR": "1",
            "NCCL_DEBUG": "WARN",
            "RAY_memory_usage_threshold": "0.98",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "PYTHONHASHSEED": "0",
        }
    )
    for name in (
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "OPENAI_API_KEY",
        "OPENAI_API_BASE",
        "OPENAI_BASE_URL",
        "VTOOL_JUDGE_API_BASE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        env.pop(name, None)
    return env


def audit_resolved_config(
    rendered: str,
    config: Mapping[str, Any],
    validated: Mapping[str, Any],
    *,
    arm_name: str,
    output_dir: Path,
) -> dict[str, bool]:
    import yaml  # type: ignore[import-untyped]

    resolved = yaml.safe_load(rendered)
    if not isinstance(resolved, Mapping):
        raise ValueError("Hydra resolved config must be a mapping")
    training = _require_mapping(config["training"], name="training")
    resources = _require_mapping(config["resources"], name="resources")
    arm = _require_mapping(validated["arm"], name="validated.arm")
    credit = _require_mapping(arm["action_credit"], name="arm.action_credit")

    def get(*keys: str) -> object:
        value: object = resolved
        for key in keys:
            value = _require_mapping(value, name=".".join(keys))[key]
        return value

    expected = {
        "data.train_files": str(validated["train_path"]),
        "data.val_files": str(validated["curve_path"]),
        "data.train_batch_size": int(training["data_train_batch_size"]),
        "data.max_prompt_length": int(training["data_max_prompt_length"]),
        "data.max_response_length": int(training["data_max_response_length"]),
        "data.shuffle": bool(training["data_shuffle"]),
        "data.filter_overlong_prompts": False,
        "data.truncation": "error",
        "data.trust_remote_code": False,
        "actor_rollout_ref.model.path": str(validated["model"]),
        "actor_rollout_ref.model.enable_gradient_checkpointing": bool(
            training["actor_gradient_checkpointing"]
        ),
        "actor_rollout_ref.actor.strategy": str(training["actor_fsdp_strategy"]),
        "actor_rollout_ref.actor.fsdp_config.strategy": str(
            training["actor_fsdp_strategy"]
        ),
        "actor_rollout_ref.actor.fsdp_config.dtype": str(training["actor_dtype"]),
        "actor_rollout_ref.actor.fsdp_config.param_offload": bool(
            training["actor_fsdp_param_offload"]
        ),
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": bool(
            training["actor_fsdp_optimizer_offload"]
        ),
        "actor_rollout_ref.actor.fsdp_config.forward_prefetch": bool(
            training["actor_fsdp_forward_prefetch"]
        ),
        "actor_rollout_ref.actor.fsdp_config.use_torch_compile": bool(
            training["actor_fsdp_use_torch_compile"]
        ),
        "actor_rollout_ref.actor.freeze_vision_tower": bool(
            training["actor_freeze_vision_tower"]
        ),
        "actor_rollout_ref.actor.optim.lr": float(training["actor_learning_rate"]),
        "actor_rollout_ref.actor.ppo_mini_batch_size": int(
            training["actor_ppo_mini_batch_size"]
        ),
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": int(
            training["actor_ppo_micro_batch_size_per_gpu"]
        ),
        "actor_rollout_ref.actor.ppo_epochs": int(training["actor_ppo_epochs"]),
        "actor_rollout_ref.actor.shuffle": bool(training["actor_shuffle"]),
        "actor_rollout_ref.actor.use_dynamic_bsz": bool(
            training["actor_use_dynamic_batching"]
        ),
        "actor_rollout_ref.actor.use_kl_loss": False,
        "actor_rollout_ref.rollout.n": int(training["rollout_n"]),
        "actor_rollout_ref.rollout.dtype": str(training["actor_dtype"]),
        "actor_rollout_ref.rollout.gpu_memory_utilization": float(
            training["rollout_gpu_memory_utilization"]
        ),
        "actor_rollout_ref.rollout.enforce_eager": bool(
            training["rollout_enforce_eager"]
        ),
        "actor_rollout_ref.rollout.max_num_batched_tokens": int(
            training["rollout_max_num_batched_tokens"]
        ),
        "actor_rollout_ref.rollout.max_model_len": int(
            training["rollout_max_model_len"]
        ),
        "actor_rollout_ref.rollout.max_num_seqs": int(training["rollout_max_num_seqs"]),
        "actor_rollout_ref.rollout.limit_images": int(training["rollout_limit_images"]),
        "actor_rollout_ref.rollout.enable_chunked_prefill": bool(
            training["rollout_enable_chunked_prefill"]
        ),
        "actor_rollout_ref.rollout.enable_prefix_caching": bool(
            training["rollout_enable_prefix_caching"]
        ),
        "actor_rollout_ref.rollout.calculate_log_probs": bool(
            training["rollout_calculate_log_probs"]
        ),
        "actor_rollout_ref.rollout.agent.default_agent_loop": str(arm["agent_name"]),
        "reward.custom_reward_function.path": (
            "pkg://integrations.vtool_action_credit.paired_vtool"
        ),
        "reward.custom_reward_function.name": "compute_score",
        "algorithm.adv_estimator": "grpo",
        "algorithm.norm_adv_by_std_in_grpo": bool(
            training["grpo_normalize_outcome_advantage_by_std"]
        ),
        "algorithm.use_kl_in_reward": False,
        "algorithm.action_credit.enabled": bool(credit["enabled"]),
        "algorithm.action_credit.mode": str(credit["mode"]),
        "algorithm.action_credit.beta": float(credit["beta"]),
        "trainer.experiment_name": arm_name,
        "trainer.n_gpus_per_node": int(resources["gpu_count"]),
        "trainer.total_training_steps": int(training["total_optimizer_steps"]),
        "trainer.total_epochs": 1,
        "trainer.save_freq": int(training["save_frequency_steps"]),
        "trainer.max_actor_ckpt_to_keep": int(
            training["max_actor_checkpoints_to_keep"]
        ),
        "trainer.test_freq": -1,
        "trainer.val_before_train": False,
        "trainer.resume_mode": "disable",
        "trainer.default_local_dir": str(output_dir / "checkpoints"),
        "trainer.rollout_data_dir": str(output_dir / "rollouts"),
        "trainer.validation_data_dir": None,
        "ray_kwargs.ray_init.num_cpus": 48,
    }
    checks: dict[str, bool] = {}
    mismatches: dict[str, dict[str, object]] = {}
    for dotted, expected_value in expected.items():
        actual_value = get(*dotted.split("."))
        passed = actual_value == expected_value
        checks[dotted] = passed
        if not passed:
            mismatches[dotted] = {
                "actual": actual_value,
                "expected": expected_value,
            }
    if mismatches:
        raise ValueError(
            "Hydra resolved config violated frozen G1 contract: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return checks


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validated = validate_frozen_inputs(config, arm_name=args.arm)
    output_dir = args.output_dir.resolve()
    artifact_root = (REPO_ROOT / "artifacts").resolve(strict=True)
    if not output_dir.is_relative_to(artifact_root):
        raise ValueError("G1 output directory must stay under repository artifacts")
    command = build_command(config, validated, arm_name=args.arm, output_dir=output_dir)
    code_worktree_status = _git_status_lines(REPO_ROOT)
    implementation_sha256 = {
        str(relative): sha256_file(_repo_path(relative))
        for relative in IMPLEMENTATION_PATHS
    }
    manifest = {
        "schema": "vtool_action_credit_g1_launch_manifest_v1",
        "arm": args.arm,
        "command": command,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "code_revision": _git(REPO_ROOT, "rev-parse", "HEAD"),
        "code_worktree_status": code_worktree_status,
        "curve_data_sha256": validated["curve_sha256"],
        "method_protocol_sha256": config["protocol"]["method_protocol_sha256"],
        "model_revision": config["model"]["revision"],
        "model_weight_sha256": validated["model_weight_sha256"],
        "full_train_runtime_audit_sha256": validated["full_train_runtime_audit_sha256"],
        "action_gradient_report_sha256": validated["action_gradient_report_sha256"],
        "environment_report_sha256": validated["environment_report_sha256"],
        "fake_server_report_sha256": validated["fake_server_report_sha256"],
        "model_load_report_sha256": validated["model_load_report_sha256"],
        "processor_report_sha256": validated["processor_report_sha256"],
        "implementation_sha256": implementation_sha256,
        "protected_split_contents_accessed": False,
        "runtime_commit": config["runtime"]["upstream_commit"],
        "runtime_diff_sha256": validated["runtime_diff_sha256"],
        "train_data_sha256": validated["train_sha256"],
    }
    if args.mode == "print":
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    if args.mode == "execute" and code_worktree_status:
        raise RuntimeError(
            "G1 execute mode requires a clean repository worktree; commit locally first"
        )
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite G1 output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    _write_json(output_dir / "launch-manifest.json", manifest)
    env = sanitized_environment(Path(validated["runtime"]))
    if args.mode == "hydra-dry-run":
        result = subprocess.run(
            [*command[:5], "--cfg", "job", *command[5:]],
            cwd=validated["runtime"],
            env=env,
            text=True,
            capture_output=True,
        )
        (output_dir / "resolved-config.yaml").write_text(
            result.stdout, encoding="utf-8"
        )
        (output_dir / "hydra-stderr.txt").write_text(result.stderr, encoding="utf-8")
        resolved_checks: dict[str, bool] = {}
        resolved_error: str | None = None
        if result.returncode == 0:
            try:
                resolved_checks = audit_resolved_config(
                    result.stdout,
                    config,
                    validated,
                    arm_name=args.arm,
                    output_dir=output_dir,
                )
            except (KeyError, TypeError, ValueError) as exc:
                resolved_error = str(exc)
        effective_exit_code = result.returncode or (2 if resolved_error else 0)
        report = {
            "schema": "vtool_action_credit_g1_hydra_dry_run_v1",
            "arm": args.arm,
            "decision": (
                "vtool_action_credit_g1_hydra_dry_run_passed"
                if effective_exit_code == 0
                else "vtool_action_credit_g1_hydra_dry_run_failed"
            ),
            "exit_code": effective_exit_code,
            "launch_manifest_sha256": sha256_file(output_dir / "launch-manifest.json"),
            "resolved_config_sha256": (
                sha256_file(output_dir / "resolved-config.yaml")
                if result.stdout
                else None
            ),
            "resolved_config_checks": resolved_checks,
            "resolved_config_error": resolved_error,
        }
        _write_json(output_dir / "report.json", report)
        print(json.dumps(report, sort_keys=True))
        return effective_exit_code

    visible_devices = [
        item for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item
    ]
    if len(visible_devices) != int(config["resources"]["gpu_count"]):
        raise RuntimeError("G1 execute mode requires exactly four visible GPUs")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("G1 execute mode is restricted to a Slurm allocation")
    started = time.time()
    execution_result = subprocess.run(command, cwd=validated["runtime"], env=env)
    _write_json(
        output_dir / "execution.json",
        {
            "schema": "vtool_action_credit_g1_execution_v1",
            "arm": args.arm,
            "elapsed_seconds": time.time() - started,
            "exit_code": execution_result.returncode,
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
        },
    )
    return execution_result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
