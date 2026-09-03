#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from beyond_entropy.refocus_chart_audit import canonical_sha256, sha256_file
from beyond_entropy.refocus_g1_dataset import ACTION_SYSTEM_PROMPT_V2
from beyond_entropy.refocus_typed_action_evaluation import (
    TypedActionResponseDiagnostics,
    analyze_typed_action_response,
)
from beyond_entropy.refocus_typed_action_runtime import (
    execute_renderer_owned_action,
    image_sha256,
    load_refocus_runtime,
    tensor_sha256,
)


REPORT_SCHEMA = "refocus_typed_action_b0_generation_smoke_v1"
REPORT_DECISION = "refocus_typed_action_b0_generation_smoke_completed"
EXPECTED_METRICS = (
    "tool_intent",
    "complete_python_fence",
    "python_syntax_valid",
    "argument_contract_valid",
    "parser_valid",
    "execution_success",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _require_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _require_sequence(value: object, *, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a sequence")
    return value


def _resolve_repo_path(repo: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value)
    return (path if path.is_absolute() else repo / path).resolve(strict=True)


def _load_json_mapping(path: Path, *, field: str) -> Mapping[str, Any]:
    return _require_mapping(
        json.loads(path.read_text(encoding="utf-8")),
        field=field,
    )


def _diagnostic_payload(
    diagnostics: TypedActionResponseDiagnostics,
) -> dict[str, Any]:
    return {
        "tool_intent": diagnostics.tool_intent,
        "complete_python_fence": diagnostics.complete_python_fence,
        "python_syntax_valid": diagnostics.python_syntax_valid,
        "argument_contract_valid": diagnostics.argument_contract_valid,
        "parser_valid": diagnostics.parser_valid,
        "contract_error": diagnostics.contract_error,
        "parser_error": diagnostics.parser_error,
    }


def _conditional_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _scientific_decision(
    *,
    intent_count: int,
    parser_valid_count: int,
    execution_count: int,
    minimum_intents: int,
    minimum_conditional_execution_rate: float,
) -> str:
    if intent_count < minimum_intents:
        return "typed_action_b0_insufficient_tool_intent_support"
    parser_rate = parser_valid_count / intent_count
    execution_rate = execution_count / intent_count
    if (
        parser_rate >= minimum_conditional_execution_rate
        and execution_rate >= minimum_conditional_execution_rate
    ):
        return "typed_action_b0_format_gate_passed"
    return "typed_action_b0_malformed_tool_intent"


def _validate_protocol(config: Mapping[str, Any]) -> None:
    analysis = _require_mapping(config.get("analysis"), field="analysis")
    data = _require_mapping(config.get("data"), field="data")
    resources = _require_mapping(config.get("resources"), field="resources")
    sampling = _require_mapping(config.get("sampling"), field="sampling")
    if config.get("schema") != "refocus_typed_action_b0_generation_protocol_v1":
        raise ValueError("B0 generation protocol schema mismatch")
    if config.get("study_role") != "baseline_correctness_only":
        raise ValueError("B0 generation study role mismatch")
    if config.get("uses_reward_target") is not False:
        raise ValueError("B0 generation must not use reward targets")
    if tuple(analysis.get("nested_metrics", ())) != EXPECTED_METRICS:
        raise ValueError("B0 nested metrics changed")
    if analysis.get("raw_model_text_execution_allowed") is not False:
        raise ValueError("raw model text execution must remain forbidden")
    if data.get("protected_split_contents_accessed") is not False:
        raise ValueError("protected split must remain closed")
    if data.get("row_count") != 1 or data.get("development_split") != "b0_smoke":
        raise ValueError("B0 generation must use the frozen one-row smoke split")
    if resources.get("gpu_count") != 1 or resources.get("gpu_type") != "H800":
        raise ValueError("B0 generation requires exactly one H800")
    if (
        resources.get("optimizer_steps") != 0
        or resources.get("checkpoints_written") != 0
    ):
        raise ValueError("B0 generation must not optimize or checkpoint")
    generation_count = sampling.get("generation_count")
    seeds = _require_sequence(sampling.get("seeds"), field="sampling seeds")
    if generation_count != 16 or len(seeds) != generation_count:
        raise ValueError("B0 generation count must remain exactly 16")
    if any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("B0 generation seeds must be unique integers")
    if sampling.get("n") != 1 or sampling.get("max_tokens") != 128:
        raise ValueError("B0 per-seed sampling contract changed")


def main() -> None:
    args = parse_args()
    if len(args.code_revision) != 40:
        raise ValueError("code revision must be a full commit")
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite B0 generation report: {args.output}"
        )

    import torch
    import vllm
    from omegaconf import OmegaConf
    from transformers import AutoProcessor
    from verl.utils.chat_template import apply_chat_template
    from verl.utils.dataset.rl_dataset import RLHFDataset
    from verl.utils.tokenizer import normalize_token_ids
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    repo = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve(strict=True)
    config = _load_json_mapping(config_path, field="B0 generation protocol")
    _validate_protocol(config)
    analysis = _require_mapping(config["analysis"], field="analysis")
    data = _require_mapping(config["data"], field="data")
    model = _require_mapping(config["model"], field="model")
    prompt_config = _require_mapping(config["prompt"], field="prompt")
    runtime_config = _require_mapping(config["runtime"], field="runtime")
    sampling = _require_mapping(config["sampling"], field="sampling")

    dataset_path = _resolve_repo_path(repo, data["dataset"], field="dataset")
    converter_report_path = _resolve_repo_path(
        repo, data["converter_report"], field="converter report"
    )
    processor_report_path = _resolve_repo_path(
        repo,
        data["processor_executor_report"],
        field="processor/executor report",
    )
    model_path = _resolve_repo_path(repo, model["local_snapshot"], field="model")
    runtime_root = _resolve_repo_path(
        repo, runtime_config["worktree"], field="runtime worktree"
    )
    runtime_tools_path = (
        runtime_root / "recipe" / "vtool" / "refocus_tools.py"
    ).resolve(strict=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("B0 generation requires exactly one visible CUDA device")

    converter_report = _load_json_mapping(
        converter_report_path, field="converter report"
    )
    processor_report = _load_json_mapping(
        processor_report_path, field="processor/executor report"
    )
    artifact_checks = {
        "dataset_sha256_matches": sha256_file(dataset_path) == data["dataset_sha256"],
        "converter_report_sha256_matches": sha256_file(converter_report_path)
        == data["converter_report_sha256"],
        "processor_report_sha256_matches": sha256_file(processor_report_path)
        == data["processor_executor_report_sha256"],
        "converter_decision_matches": converter_report.get("decision")
        == "refocus_official_typed_action_b0_converter_passed",
        "processor_decision_matches": processor_report.get("decision")
        == "refocus_typed_action_b0_real_runtime_smoke_passed",
        "processor_checks_all_true": all(
            _require_mapping(
                processor_report.get("checks"), field="processor checks"
            ).values()
        ),
        "runtime_tools_sha256_matches": sha256_file(runtime_tools_path)
        == runtime_config["refocus_tools_sha256"],
        "model_revision_matches": model_path.name == model["revision"],
        "system_prompt_sha256_matches": canonical_sha256(ACTION_SYSTEM_PROMPT_V2)
        == prompt_config["system_prompt_sha256"],
    }
    if not all(artifact_checks.values()):
        raise ValueError("B0 frozen artifact contract failed")

    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        use_fast=True,
    )
    data_config = OmegaConf.create(
        {
            "filter_overlong_prompts": False,
            "return_raw_chat": True,
            "return_multi_modal_inputs": False,
            "image_key": "images",
            "need_tools_kwargs": True,
            "max_prompt_length": 4096,
        }
    )
    dataset = RLHFDataset(
        str(dataset_path),
        processor.tokenizer,
        data_config,
        processor=processor,
    )
    if len(dataset) != 1:
        raise ValueError("B0 generation dataset must contain exactly one row")
    raw_row = dataset.dataframe[0]
    raw_prompt = raw_row["prompt"]
    extra_info = _require_mapping(raw_row.get("extra_info"), field="row extra_info")
    item = dataset[0]
    images, videos = asyncio.run(
        RLHFDataset.process_vision_info(
            item["raw_prompt"],
            image_patch_size=processor.image_processor.patch_size,
            config=data_config,
        )
    )
    if images is None or len(images) != 1 or videos:
        raise ValueError("B0 generation expects exactly one image and no videos")
    rendered_prompt = apply_chat_template(
        processor,
        item["raw_prompt"],
        add_generation_prompt=True,
        tokenize=False,
    )
    model_inputs = processor(
        text=[rendered_prompt],
        images=images,
        return_tensors="pt",
    )
    prompt_ids = normalize_token_ids(model_inputs["input_ids"])
    if len(prompt_ids) > 4096:
        raise ValueError("B0 generation prompt exceeds 4096 tokens")
    metadata = _require_mapping(
        json.loads(item["tools_kwargs"]["metadata"]),
        field="tools metadata",
    )
    available_labels = {
        "x": tuple(metadata.get("x_values", ())),
        "y": tuple(metadata.get("y_values", ())),
    }

    free_before, total_memory = torch.cuda.mem_get_info()
    load_started = time.perf_counter()
    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=1,
        dtype=str(sampling["dtype"]),
        seed=int(sampling["engine_seed"]),
        gpu_memory_utilization=float(sampling["gpu_memory_utilization"]),
        max_model_len=int(sampling["max_model_len"]),
        max_num_seqs=int(sampling["max_num_seqs"]),
        enforce_eager=bool(sampling["enforce_eager"]),
        enable_prefix_caching=bool(sampling["enable_prefix_caching"]),
        trust_remote_code=False,
        limit_mm_per_prompt={"image": 1},
        disable_log_stats=True,
    )
    load_seconds = time.perf_counter() - load_started
    free_after_load, _ = torch.cuda.mem_get_info()

    seeds = [int(value) for value in sampling["seeds"]]
    prompts = [
        TokensPrompt(
            prompt_token_ids=prompt_ids,
            multi_modal_data={"image": [images[0].copy()]},
        )
        for _ in seeds
    ]
    sampling_params = [
        SamplingParams(
            n=1,
            temperature=float(sampling["temperature"]),
            top_p=float(sampling["top_p"]),
            top_k=int(sampling["top_k"]),
            seed=seed,
            max_tokens=int(sampling["max_tokens"]),
            logprobs=1,
        )
        for seed in seeds
    ]
    generation_started = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    generation_seconds = time.perf_counter() - generation_started
    if len(outputs) != len(seeds):
        raise RuntimeError("vLLM returned the wrong number of B0 requests")

    runtime = load_refocus_runtime(runtime_tools_path)
    intent_substrings = tuple(str(value) for value in analysis["intent_substrings"])
    completions: list[dict[str, Any]] = []
    for index, (seed, request_output) in enumerate(zip(seeds, outputs, strict=True)):
        if len(request_output.outputs) != 1:
            raise RuntimeError("vLLM B0 request returned an unexpected output count")
        completion = request_output.outputs[0]
        output_ids = list(completion.token_ids)
        output_text = str(completion.text)
        diagnostics = analyze_typed_action_response(
            output_text,
            available_labels=available_labels,
            intent_substrings=intent_substrings,
        )
        finish_reason = getattr(completion, "finish_reason", None)
        stop_reason = getattr(completion, "stop_reason", None)
        execution_success = False
        execution_error: str | None = None
        output_changed: bool | None = None
        output_image_sha256: str | None = None
        action_payload: dict[str, Any] | None = None
        if diagnostics.action is not None:
            action = diagnostics.action
            action_payload = {
                "axis": action.axis,
                "mode": action.mode,
                "label_count": len(action.labels),
                "labels_sha256": canonical_sha256(list(action.labels)),
            }
            try:
                _, output_image, output_changed = execute_renderer_owned_action(
                    runtime=runtime,
                    image=images[0],
                    metadata=metadata,
                    action=action,
                )
                execution_success = True
                output_image_sha256 = image_sha256(output_image)
            except Exception as exc:
                execution_error = f"{type(exc).__name__}: {exc}"
        completions.append(
            {
                "index": index,
                "seed": seed,
                "completion_tokens": len(output_ids),
                "completion_token_ids_sha256": canonical_sha256(output_ids),
                "completion_text": output_text,
                "completion_text_sha256": canonical_sha256(output_text),
                "finish_reason": None if finish_reason is None else str(finish_reason),
                "stop_reason": None if stop_reason is None else str(stop_reason),
                **_diagnostic_payload(diagnostics),
                "action": action_payload,
                "execution_success": execution_success,
                "execution_error": execution_error,
                "output_image_changed": output_changed,
                "output_image_sha256": output_image_sha256,
                "raw_model_text_executed": False,
            }
        )

    metric_counts = {
        metric: sum(bool(row[metric]) for row in completions)
        for metric in EXPECTED_METRICS
    }
    generation_count = len(completions)
    metric_rates = {
        metric: metric_counts[metric] / generation_count for metric in EXPECTED_METRICS
    }
    intent_count = metric_counts["tool_intent"]
    conditional_on_intent = {
        metric: _conditional_rate(metric_counts[metric], intent_count)
        for metric in EXPECTED_METRICS[1:]
    }
    scientific_decision = _scientific_decision(
        intent_count=intent_count,
        parser_valid_count=metric_counts["parser_valid"],
        execution_count=metric_counts["execution_success"],
        minimum_intents=int(analysis["minimum_tool_intents"]),
        minimum_conditional_execution_rate=float(
            analysis["minimum_conditional_execution_rate"]
        ),
    )
    hierarchy_valid = all(
        row["execution_success"]
        <= row["parser_valid"]
        <= row["argument_contract_valid"]
        <= row["python_syntax_valid"]
        <= row["complete_python_fence"]
        and (not row["parser_valid"] or row["tool_intent"])
        for row in completions
    )
    row_checks = {
        "dataset_has_one_row": len(dataset) == 1,
        "row_is_independent_b0": item["split"] == "b0_smoke"
        and item["agent_name"] == "vtool_agent",
        "row_prompt_version_matches": extra_info.get("action_prompt_version")
        == "typed_action_v2",
        "row_prompt_hash_matches": extra_info.get("prompt_sha256")
        == canonical_sha256(raw_prompt),
        "row_system_prompt_is_exact_v2": raw_prompt[0]["content"]
        == ACTION_SYSTEM_PROMPT_V2,
        "one_image_no_video": len(images) == 1 and not videos,
        "focus_area_absent": "focus_areas_bbox" not in metadata,
        "prompt_fits_4096": len(prompt_ids) <= 4096,
        "generation_count_matches": generation_count
        == int(sampling["generation_count"]),
        "seed_order_matches": [row["seed"] for row in completions] == seeds,
        "nested_metric_hierarchy_valid": hierarchy_valid,
        "no_raw_model_text_executed": all(
            row["raw_model_text_executed"] is False for row in completions
        ),
    }
    checks = {**artifact_checks, **row_checks}
    decision = (
        REPORT_DECISION
        if all(checks.values())
        else ("refocus_typed_action_b0_generation_smoke_failed")
    )
    free_after_generation, _ = torch.cuda.mem_get_info()
    report = {
        "schema": REPORT_SCHEMA,
        "decision": decision,
        "scientific_decision": scientific_decision,
        "format_gate_qualified": scientific_decision
        == "typed_action_b0_format_gate_passed",
        "checks": checks,
        "code_revision": args.code_revision,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "runner_sha256": sha256_file(Path(__file__).resolve(strict=True)),
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "row_id_sha256": canonical_sha256(str(item["id"])),
        "structural_chart_sha256": str(extra_info["structural_chart_sha256"]),
        "source": str(metadata["source"]),
        "model_path": str(model_path),
        "model_revision": model_path.name,
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_bytes": int(total_memory),
        "gpu_free_before_bytes": int(free_before),
        "gpu_free_after_load_bytes": int(free_after_load),
        "gpu_free_after_generation_bytes": int(free_after_generation),
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "prompt_tokens": len(prompt_ids),
        "input_ids_sha256": tensor_sha256(model_inputs["input_ids"]),
        "pixel_values_sha256": tensor_sha256(model_inputs["pixel_values"]),
        "original_image_sha256": image_sha256(images[0]),
        "rendered_prompt_sha256": canonical_sha256(rendered_prompt),
        "sampling": dict(sampling),
        "sampling_sha256": canonical_sha256(sampling),
        "generation_count": generation_count,
        "metric_counts": metric_counts,
        "metric_rates": metric_rates,
        "conditional_on_intent_rates": conditional_on_intent,
        "minimum_tool_intents": int(analysis["minimum_tool_intents"]),
        "minimum_conditional_execution_rate": float(
            analysis["minimum_conditional_execution_rate"]
        ),
        "completions": completions,
        "reward_target_used": False,
        "raw_model_text_executed": False,
        "model_weights_loaded": True,
        "optimizer_steps": 0,
        "checkpoints_written": 0,
        "protected_split_contents_accessed": False,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "scientific_decision": scientific_decision,
                "metric_counts": metric_counts,
                "generation_count": generation_count,
            },
            sort_keys=True,
        )
    )
    if decision != REPORT_DECISION:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
