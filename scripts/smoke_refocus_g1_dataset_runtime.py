#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from beyond_entropy.refocus_chart_audit import canonical_sha256, sha256_file
from beyond_entropy.refocus_g1_dataset import AGENT_NAME, DATA_SOURCE


REPORT_SCHEMA = "refocus_g1_dataset_runtime_smoke_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load one converted official-train row through the pinned verl "
            "RLHFDataset and Qwen2.5-VL processor without loading model weights."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _tensor_sha256(tensor: Any) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    from omegaconf import OmegaConf
    from transformers import AutoProcessor
    from verl.utils.dataset.rl_dataset import RLHFDataset

    args = parse_args()
    dataset_path = args.dataset.resolve(strict=True)
    model_path = args.model.resolve(strict=True)
    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        use_fast=True,
    )
    config = OmegaConf.create(
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
        config,
        processor=processor,
    )
    if len(dataset) != 1:
        raise ValueError("runtime smoke requires exactly one converted row")
    item = dataset[0]
    images, videos = asyncio.run(
        RLHFDataset.process_vision_info(
            item["raw_prompt"],
            image_patch_size=processor.image_processor.patch_size,
            config=config,
        )
    )
    if images is None or len(images) != 1:
        raise ValueError("runtime smoke must decode exactly one original image")
    if videos:
        raise ValueError("runtime smoke unexpectedly decoded a video")

    rendered_prompt = processor.apply_chat_template(
        item["raw_prompt"], add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        text=[rendered_prompt],
        images=images,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]
    metadata = json.loads(item["tools_kwargs"]["metadata"])
    checks = {
        "agent_name_matches": item["agent_name"] == AGENT_NAME,
        "data_source_matches": item["data_source"] == DATA_SOURCE,
        "ground_truth_nonempty": bool(item["reward_model"]["ground_truth"]),
        "raw_prompt_roles_match": [message["role"] for message in item["raw_prompt"]]
        == ["system", "user"],
        "one_image_decoded": len(images) == 1,
        "no_video_decoded": not videos,
        "focus_area_absent": "focus_areas_bbox" not in metadata,
        "tool_name_matches": item["tools_kwargs"]["name"] == "refocus",
        "prompt_fits_4096": int(input_ids.shape[-1]) <= 4096,
    }
    decision = (
        "refocus_g1_dataset_runtime_smoke_passed"
        if all(checks.values())
        else "refocus_g1_dataset_runtime_smoke_failed"
    )
    report = {
        "schema": REPORT_SCHEMA,
        "decision": decision,
        "checks": checks,
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "dataset_rows": len(dataset),
        "model_path": str(model_path),
        "processor_class": type(processor).__name__,
        "image_processor_class": type(processor.image_processor).__name__,
        "processor_use_fast": True,
        "prompt_tokens": int(input_ids.shape[-1]),
        "input_ids_sha256": _tensor_sha256(input_ids),
        "pixel_values_shape": list(pixel_values.shape),
        "pixel_values_sha256": _tensor_sha256(pixel_values),
        "rendered_prompt_sha256": canonical_sha256(rendered_prompt),
        "row_id_sha256": canonical_sha256(str(item["id"])),
        "structural_chart_sha256": str(item["extra_info"]["structural_chart_sha256"]),
        "protected_split_contents_accessed": False,
        "model_weights_loaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    if decision != "refocus_g1_dataset_runtime_smoke_passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
