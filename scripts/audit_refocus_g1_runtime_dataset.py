#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from beyond_entropy.refocus_chart_audit import (
    STRUCTURAL_METADATA_FIELDS,
    canonical_sha256,
    sha256_file,
)
from beyond_entropy.refocus_g1_dataset import DATA_SOURCE


REPORT_SCHEMA = "refocus_g1_runtime_dataset_audit_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit every frozen G1 row through the pinned verl dataset and Qwen "
            "processor without loading model weights."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-agent-name", required=True)
    parser.add_argument("--expected-split", required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--expected-structural-groups", type=int, required=True)
    parser.add_argument("--max-prompt-length", type=int, required=True)
    return parser.parse_args()


def nearest_rank_percentile(values: Sequence[int], probability: float) -> int:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    ordered = sorted(values)
    rank = max(1, math.ceil(probability * len(ordered)))
    return ordered[rank - 1]


def validate_tool_metadata(value: object) -> dict[str, Any]:
    if not isinstance(value, str):
        raise ValueError("tools metadata must be serialized JSON")
    metadata = json.loads(value)
    if not isinstance(metadata, dict):
        raise ValueError("tools metadata must decode to a mapping")
    if set(metadata) != set(STRUCTURAL_METADATA_FIELDS):
        raise ValueError("tools metadata contains non-structural or missing fields")
    return metadata


def _require_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def main() -> int:
    from omegaconf import OmegaConf
    import torch
    import transformers
    from transformers import AutoProcessor
    from verl.utils.chat_template import apply_chat_template
    from verl.utils.dataset.rl_dataset import RLHFDataset

    args = parse_args()
    dataset_path = args.dataset.resolve(strict=True)
    model_path = args.model.resolve(strict=True)
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite runtime audit: {output_path}")
    if args.expected_rows <= 0 or args.expected_structural_groups <= 0:
        raise ValueError("expected dataset sizes must be positive")
    if args.max_prompt_length <= 0:
        raise ValueError("max prompt length must be positive")
    actual_dataset_sha256 = sha256_file(dataset_path)
    if actual_dataset_sha256 != args.dataset_sha256:
        raise ValueError("dataset SHA-256 does not match the frozen config")

    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    config = OmegaConf.create(
        {
            "filter_overlong_prompts": False,
            "return_raw_chat": True,
            "return_multi_modal_inputs": False,
            "image_key": "images",
            "need_tools_kwargs": True,
            "max_prompt_length": args.max_prompt_length,
            "truncation": "error",
            "shuffle": False,
        }
    )
    dataset = RLHFDataset(
        str(dataset_path),
        processor.tokenizer,
        config,
        processor=processor,
    )
    checks = {
        "dataset_rows_match": len(dataset) == args.expected_rows,
        "all_agent_names_match": True,
        "all_data_sources_match": True,
        "all_ground_truths_nonempty": True,
        "all_prompt_roles_match": True,
        "all_prompt_hashes_match": True,
        "all_rows_have_one_image": True,
        "no_rows_have_video": True,
        "all_tool_names_match": True,
        "all_tool_metadata_structural_only": True,
        "all_prompts_fit_frozen_limit": True,
        "all_pixel_tensors_nonempty": True,
        "row_ids_unique": True,
        "structural_groups_match": True,
    }
    failures: list[dict[str, Any]] = []
    prompt_lengths: list[int] = []
    pixel_shapes: dict[str, int] = {}
    row_ids: set[str] = set()
    structural_groups: set[str] = set()

    for index in range(len(dataset)):
        raw_row = dataset.dataframe[index]
        item = dataset[index]
        row_failures: list[str] = []
        row_id = str(item.get("id"))
        row_id_sha256 = canonical_sha256(row_id)
        if row_id in row_ids:
            checks["row_ids_unique"] = False
            row_failures.append("duplicate_row_id")
        row_ids.add(row_id)

        if item.get("agent_name") != args.expected_agent_name:
            checks["all_agent_names_match"] = False
            row_failures.append("agent_name")
        if item.get("data_source") != DATA_SOURCE:
            checks["all_data_sources_match"] = False
            row_failures.append("data_source")
        reward_model = _require_mapping(item.get("reward_model"), name="reward_model")
        if not str(reward_model.get("ground_truth", "")).strip():
            checks["all_ground_truths_nonempty"] = False
            row_failures.append("ground_truth")
        extra_info = _require_mapping(item.get("extra_info"), name="extra_info")
        if extra_info.get("split") != args.expected_split:
            row_failures.append("development_split")
        structural_sha256 = str(extra_info.get("structural_chart_sha256", ""))
        if len(structural_sha256) != 64:
            row_failures.append("structural_chart_sha256")
        structural_groups.add(structural_sha256)

        original_prompt = raw_row.get("prompt")
        if not isinstance(original_prompt, list) or [
            message.get("role") for message in original_prompt
        ] != ["system", "user"]:
            checks["all_prompt_roles_match"] = False
            row_failures.append("prompt_roles")
        if canonical_sha256(original_prompt) != str(extra_info.get("prompt_sha256")):
            checks["all_prompt_hashes_match"] = False
            row_failures.append("prompt_hash")
        if [message.get("role") for message in item["raw_prompt"]] != [
            "system",
            "user",
        ]:
            checks["all_prompt_roles_match"] = False
            row_failures.append("raw_prompt_roles")

        tools_kwargs = _require_mapping(item.get("tools_kwargs"), name="tools_kwargs")
        if tools_kwargs.get("name") != "refocus":
            checks["all_tool_names_match"] = False
            row_failures.append("tool_name")
        try:
            validate_tool_metadata(tools_kwargs.get("metadata"))
        except (TypeError, ValueError, json.JSONDecodeError):
            checks["all_tool_metadata_structural_only"] = False
            row_failures.append("tool_metadata")

        images, videos = asyncio.run(
            RLHFDataset.process_vision_info(
                item["raw_prompt"],
                image_patch_size=processor.image_processor.patch_size,
                config=config,
            )
        )
        if images is None or len(images) != 1:
            checks["all_rows_have_one_image"] = False
            row_failures.append("image_count")
        if videos:
            checks["no_rows_have_video"] = False
            row_failures.append("video_count")
        if images is not None and len(images) == 1 and not videos:
            rendered_prompt = apply_chat_template(
                processor,
                item["raw_prompt"],
                add_generation_prompt=True,
                tokenize=False,
            )
            inputs = processor(
                text=[rendered_prompt],
                images=images,
                return_tensors="pt",
            )
            prompt_length = int(inputs["input_ids"].shape[-1])
            prompt_lengths.append(prompt_length)
            if prompt_length > args.max_prompt_length:
                checks["all_prompts_fit_frozen_limit"] = False
                row_failures.append("prompt_overlong")
            pixel_values = inputs.get("pixel_values")
            if pixel_values is None or pixel_values.numel() == 0:
                checks["all_pixel_tensors_nonempty"] = False
                row_failures.append("pixel_values")
            else:
                shape = "x".join(str(value) for value in pixel_values.shape)
                pixel_shapes[shape] = pixel_shapes.get(shape, 0) + 1
        if row_failures:
            failures.append(
                {
                    "row_index": index,
                    "row_id_sha256": row_id_sha256,
                    "failures": sorted(set(row_failures)),
                }
            )
        if (index + 1) % 10 == 0 or index + 1 == len(dataset):
            print(
                json.dumps(
                    {"audited_rows": index + 1, "dataset_rows": len(dataset)},
                    sort_keys=True,
                ),
                flush=True,
            )

    checks["structural_groups_match"] = (
        len(structural_groups) == args.expected_structural_groups
    )
    if not checks["structural_groups_match"]:
        failures.append(
            {
                "failure": "structural_group_count",
                "actual": len(structural_groups),
                "expected": args.expected_structural_groups,
            }
        )
    if len(prompt_lengths) != len(dataset):
        checks["all_prompts_fit_frozen_limit"] = False
    decision = (
        "refocus_g1_runtime_dataset_audit_passed"
        if all(checks.values()) and not failures
        else "refocus_g1_runtime_dataset_audit_failed"
    )
    report = {
        "schema": REPORT_SCHEMA,
        "decision": decision,
        "checks": checks,
        "dataset": str(dataset_path),
        "dataset_sha256": actual_dataset_sha256,
        "dataset_rows": len(dataset),
        "structural_groups": len(structural_groups),
        "agent_name": args.expected_agent_name,
        "development_split": args.expected_split,
        "model_path": str(model_path),
        "model_revision": model_path.name,
        "processor_class": type(processor).__name__,
        "image_processor_class": type(processor.image_processor).__name__,
        "processor_use_fast": True,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "audit_script_sha256": sha256_file(Path(__file__).resolve()),
        "prompt_tokens": {
            "min": min(prompt_lengths) if prompt_lengths else None,
            "median_nearest_rank": (
                nearest_rank_percentile(prompt_lengths, 0.5) if prompt_lengths else None
            ),
            "p95_nearest_rank": (
                nearest_rank_percentile(prompt_lengths, 0.95)
                if prompt_lengths
                else None
            ),
            "max": max(prompt_lengths) if prompt_lengths else None,
            "frozen_limit": args.max_prompt_length,
        },
        "pixel_value_shapes": dict(sorted(pixel_shapes.items())),
        "row_id_manifest_sha256": canonical_sha256(sorted(row_ids)),
        "structural_group_manifest_sha256": canonical_sha256(sorted(structural_groups)),
        "failures": failures,
        "protected_split_contents_accessed": False,
        "model_weights_loaded": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps({"decision": decision, "output": str(output_path)}, sort_keys=True)
    )
    return 0 if decision == "refocus_g1_runtime_dataset_audit_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
