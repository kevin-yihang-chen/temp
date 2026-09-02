#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import time

from beyond_entropy.refocus_chart_audit import canonical_sha256, sha256_file
from beyond_entropy.vtool_action_credit import deterministic_rollout_seeds


REPORT_SCHEMA = "vtool_vllm_model_load_smoke_v1"
SMOKE_TRAJECTORY = {"stage": "model_load", "index": 0, "rollout": 0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load the pinned Qwen2.5-VL model in vLLM and generate one first-turn "
            "response from a converted official-train row."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    import vllm
    from omegaconf import OmegaConf
    from transformers import AutoProcessor
    from verl.utils.chat_template import apply_chat_template
    from verl.utils.dataset.rl_dataset import RLHFDataset
    from verl.utils.tokenizer import normalize_token_ids
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from recipe.vtool.refocus_tools import RefocusCodeParser

    dataset_path = args.dataset.resolve(strict=True)
    model_path = args.model.resolve(strict=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the vLLM model-load smoke")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("model-load smoke requires exactly one visible GPU")

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
        raise ValueError("model-load smoke dataset must contain exactly one row")
    item = dataset[0]
    images, videos = asyncio.run(
        RLHFDataset.process_vision_info(
            item["raw_prompt"],
            image_patch_size=processor.image_processor.patch_size,
            config=data_config,
        )
    )
    if images is None or len(images) != 1 or videos:
        raise ValueError("model-load smoke expects exactly one image and no videos")
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
        raise ValueError("model-load smoke prompt exceeds frozen maximum")

    free_before, total_memory = torch.cuda.mem_get_info()
    load_started = time.perf_counter()
    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=1,
        dtype="bfloat16",
        seed=20260902,
        gpu_memory_utilization=0.45,
        max_model_len=5120,
        max_num_seqs=4,
        enforce_eager=True,
        enable_prefix_caching=False,
        trust_remote_code=False,
        limit_mm_per_prompt={"image": 2},
        disable_log_stats=True,
    )
    load_seconds = time.perf_counter() - load_started
    free_after_load, _ = torch.cuda.mem_get_info()

    action_seed, continuation_seed = deterministic_rollout_seeds(SMOKE_TRAJECTORY)
    sampling = SamplingParams(
        n=1,
        temperature=0.7,
        top_p=0.9,
        top_k=-1,
        seed=action_seed,
        max_tokens=128,
        logprobs=1,
    )
    prompt = TokensPrompt(
        prompt_token_ids=prompt_ids,
        multi_modal_data={"image": images},
    )
    generation_started = time.perf_counter()
    outputs = llm.generate([prompt], sampling, use_tqdm=False)
    generation_seconds = time.perf_counter() - generation_started
    if len(outputs) != 1 or len(outputs[0].outputs) != 1:
        raise RuntimeError("vLLM model-load smoke returned unexpected output count")
    completion = outputs[0].outputs[0]
    output_ids = list(completion.token_ids)
    output_text = str(completion.text)
    if not output_ids or not output_text.strip():
        raise RuntimeError("vLLM model-load smoke returned an empty completion")
    parse_result = RefocusCodeParser().parse(output_text)
    if parse_result.error_code == "unknown":
        raise RuntimeError("first response produced a malformed tool action")

    free_after_generation, _ = torch.cuda.mem_get_info()
    report = {
        "schema": REPORT_SCHEMA,
        "decision": "vtool_vllm_model_load_smoke_passed",
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "row_id_sha256": canonical_sha256(str(item["id"])),
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
        "completion_tokens": len(output_ids),
        "completion_token_ids_sha256": canonical_sha256(output_ids),
        "completion_text": output_text,
        "completion_text_sha256": canonical_sha256(output_text),
        "parse_error_code": parse_result.error_code,
        "tool_attempted": parse_result.error_code != "NOTOOL",
        "action_seed": action_seed,
        "reserved_continuation_seed": continuation_seed,
        "sampling_sha256": canonical_sha256(
            {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": -1,
                "max_tokens": 128,
                "seed": action_seed,
            }
        ),
        "protected_split_contents_accessed": False,
        "optimizer_step_performed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
