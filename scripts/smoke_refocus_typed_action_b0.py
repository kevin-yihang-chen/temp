#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Mapping

from PIL import Image

import beyond_entropy.refocus_g1_dataset as refocus_g1_dataset_module
import beyond_entropy.refocus_typed_action as refocus_typed_action_module
from beyond_entropy.refocus_chart_audit import canonical_sha256, sha256_file
from beyond_entropy.refocus_g1_dataset import (
    ACTION_SYSTEM_PROMPT_V1,
    ACTION_SYSTEM_PROMPT_V2,
    DATA_SOURCE,
    TYPED_ACTION_AGENT_NAME,
    TYPED_ACTION_CONVERTER_SCHEMA,
    TYPED_ACTION_DEVELOPMENT_SPLIT,
)
from beyond_entropy.refocus_typed_action import (
    Axis,
    RefocusTypedAction,
    parse_refocus_typed_action,
    render_refocus_typed_action,
)


REPORT_SCHEMA = "refocus_typed_action_b0_real_runtime_smoke_v1"
REPORT_DECISION = "refocus_typed_action_b0_real_runtime_smoke_passed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load one independent V2 official-train row through the real pinned "
            "Qwen processor, then execute one renderer-owned typed action in the "
            "pinned VTool context without loading model weights."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--converter-report", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--runtime-refocus-tools", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _require_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _tensor_sha256(tensor: Any) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def _image_sha256(image: Image.Image) -> str:
    normalized = image.convert("RGBA")
    digest = hashlib.sha256()
    digest.update(str(normalized.size).encode("ascii"))
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def _load_runtime_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "typed_action_b0_runtime_refocus_tools", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned runtime module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "build_refocus_context", None)):
        raise ValueError("pinned runtime does not expose build_refocus_context")
    return module


def _select_deterministic_action(metadata: Mapping[str, Any]) -> RefocusTypedAction:
    source = str(metadata.get("source", ""))
    if source == "chartqa_v_bar":
        axis: Axis = "x"
    elif source == "chartqa_h_bar":
        axis = "y"
    else:
        raise ValueError(f"B0 one-row smoke has unsupported source: {source!r}")
    labels = metadata.get(f"{axis}_values")
    bboxes = metadata.get(f"{axis}_values_bbox")
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"B0 smoke requires non-empty {axis}-axis labels")
    if not isinstance(bboxes, Mapping):
        raise ValueError(f"B0 smoke requires {axis}-axis bbox mapping")
    label = labels[0]
    if not isinstance(label, str) or not label:
        raise ValueError("B0 smoke selected label must be non-empty text")
    if label not in bboxes:
        raise ValueError("B0 smoke selected label is absent from bbox mapping")
    return RefocusTypedAction(axis=axis, mode="draw", labels=(label,))


def _execute_renderer_owned_action(
    *,
    runtime: ModuleType,
    image: Image.Image,
    metadata: Mapping[str, Any],
    action: RefocusTypedAction,
) -> tuple[str, Image.Image, bool]:
    available_labels = {
        "x": tuple(metadata.get("x_values", ())),
        "y": tuple(metadata.get("y_values", ())),
    }
    response = render_refocus_typed_action(action)
    parsed = parse_refocus_typed_action(
        response,
        available_labels=available_labels,
    )
    if parsed != action:
        raise AssertionError("typed action renderer/parser round-trip mismatch")

    # Only re-rendered, strictly parsed structured data reaches exec. Raw model
    # text is never executed by this smoke.
    trusted_response = render_refocus_typed_action(parsed)
    if trusted_response != response:
        raise AssertionError("typed action canonicalization is not stable")
    code = trusted_response.removeprefix("```python\n").removesuffix("\n```")
    displayed: list[Image.Image] = []
    execution_image = image.convert("RGB").copy()
    original_sha256 = _image_sha256(execution_image)
    context = runtime.build_refocus_context(
        display_callback=displayed.append,
        image=execution_image,
        metadata=dict(metadata),
    )
    compile_context = {"__builtins__": {}, **context}
    exec(compile(code, "typed_action_b0_renderer_owned.py", "exec"), compile_context)
    if len(displayed) != 1 or not isinstance(displayed[0], Image.Image):
        raise RuntimeError("typed action did not display exactly one PIL image")
    output = displayed[0]
    return response, output, _image_sha256(output) != original_sha256


def _validate_converter_report(
    report: Mapping[str, Any], *, dataset_sha256: str
) -> dict[str, bool]:
    outputs = _require_mapping(report.get("outputs"), field="converter outputs")
    smoke = _require_mapping(
        outputs.get(TYPED_ACTION_DEVELOPMENT_SPLIT),
        field="converter B0 smoke output",
    )
    return {
        "converter_schema_matches": report.get("schema")
        == TYPED_ACTION_CONVERTER_SCHEMA,
        "converter_decision_passed": report.get("decision")
        == "refocus_official_typed_action_b0_converter_passed",
        "converter_prompt_version_matches": report.get("action_prompt_version")
        == "typed_action_v2",
        "converter_system_prompt_matches": report.get("system_prompt_sha256")
        == canonical_sha256(ACTION_SYSTEM_PROMPT_V2),
        "converter_agent_matches": report.get("agent_name") == TYPED_ACTION_AGENT_NAME,
        "converter_source_is_official_train": report.get("source_split") == "train",
        "converter_protected_split_closed": report.get(
            "protected_split_contents_accessed"
        )
        is False,
        "converter_has_one_row": report.get("selected_rows") == 1
        and smoke.get("rows") == 1,
        "converter_dataset_hash_matches": smoke.get("sha256") == dataset_sha256,
    }


def main() -> None:
    from omegaconf import OmegaConf
    from transformers import AutoProcessor
    from verl.utils.dataset.rl_dataset import RLHFDataset

    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite B0 smoke report: {args.output}")
    dataset_path = args.dataset.resolve(strict=True)
    converter_report_path = args.converter_report.resolve(strict=True)
    model_path = args.model.resolve(strict=True)
    runtime_path = args.runtime_refocus_tools.resolve(strict=True)
    dataset_sha256 = sha256_file(dataset_path)
    converter_report = _require_mapping(
        json.loads(converter_report_path.read_text(encoding="utf-8")),
        field="converter report",
    )
    converter_checks = _validate_converter_report(
        converter_report,
        dataset_sha256=dataset_sha256,
    )
    if not all(converter_checks.values()):
        raise ValueError("independent V2 converter report contract failed")

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
        raise ValueError("B0 runtime smoke requires exactly one converted row")
    raw_row = dataset.dataframe[0]
    raw_prompt = raw_row["prompt"]
    extra_info = _require_mapping(raw_row.get("extra_info"), field="row extra_info")
    item = dataset[0]
    images, videos = asyncio.run(
        RLHFDataset.process_vision_info(
            item["raw_prompt"],
            image_patch_size=processor.image_processor.patch_size,
            config=config,
        )
    )
    if images is None or len(images) != 1:
        raise ValueError("B0 runtime smoke must decode exactly one original image")
    if videos:
        raise ValueError("B0 runtime smoke unexpectedly decoded a video")

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
    metadata = _require_mapping(
        json.loads(item["tools_kwargs"]["metadata"]),
        field="tools metadata",
    )
    runtime = _load_runtime_module(runtime_path)
    action = _select_deterministic_action(metadata)
    action_response, output_image, output_changed = _execute_renderer_owned_action(
        runtime=runtime,
        image=images[0],
        metadata=metadata,
        action=action,
    )

    system_message = _require_mapping(raw_prompt[0], field="system message")
    row_checks = {
        "dataset_has_one_row": len(dataset) == 1,
        "row_split_is_independent_b0": item["split"] == TYPED_ACTION_DEVELOPMENT_SPLIT,
        "row_data_source_matches": item["data_source"] == DATA_SOURCE,
        "row_agent_is_outcome_only": item["agent_name"] == TYPED_ACTION_AGENT_NAME,
        "row_prompt_version_matches": extra_info.get("action_prompt_version")
        == "typed_action_v2",
        "row_prompt_hash_matches": extra_info.get("prompt_sha256")
        == canonical_sha256(raw_prompt),
        "row_system_prompt_is_exact_v2": system_message.get("content")
        == ACTION_SYSTEM_PROMPT_V2,
        "frozen_v1_prompt_still_distinct": canonical_sha256(ACTION_SYSTEM_PROMPT_V1)
        != canonical_sha256(ACTION_SYSTEM_PROMPT_V2),
        "raw_prompt_roles_match": [message["role"] for message in raw_prompt]
        == ["system", "user"],
        "one_image_decoded": len(images) == 1,
        "no_video_decoded": not videos,
        "focus_area_absent": "focus_areas_bbox" not in metadata,
        "tool_name_matches": item["tools_kwargs"]["name"] == "refocus",
        "prompt_fits_4096": int(input_ids.shape[-1]) <= 4096,
        "typed_action_round_trip": parse_refocus_typed_action(
            action_response,
            available_labels={
                "x": tuple(metadata.get("x_values", ())),
                "y": tuple(metadata.get("y_values", ())),
            },
        )
        == action,
        "runtime_displayed_one_image": isinstance(output_image, Image.Image),
        "runtime_output_changed": output_changed,
    }
    checks = {**converter_checks, **row_checks}
    decision = (
        REPORT_DECISION
        if all(checks.values())
        else ("refocus_typed_action_b0_real_runtime_smoke_failed")
    )
    report = {
        "schema": REPORT_SCHEMA,
        "decision": decision,
        "checks": checks,
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "dataset_rows": len(dataset),
        "converter_report": str(converter_report_path),
        "converter_report_sha256": sha256_file(converter_report_path),
        "model_path": str(model_path),
        "processor_class": type(processor).__name__,
        "image_processor_class": type(processor.image_processor).__name__,
        "processor_use_fast": True,
        "prompt_tokens": int(input_ids.shape[-1]),
        "input_ids_sha256": _tensor_sha256(input_ids),
        "pixel_values_shape": list(pixel_values.shape),
        "pixel_values_sha256": _tensor_sha256(pixel_values),
        "rendered_prompt_sha256": canonical_sha256(rendered_prompt),
        "system_prompt_sha256": canonical_sha256(ACTION_SYSTEM_PROMPT_V2),
        "row_id_sha256": canonical_sha256(str(item["id"])),
        "structural_chart_sha256": str(extra_info["structural_chart_sha256"]),
        "runtime_refocus_tools": str(runtime_path),
        "runtime_refocus_tools_sha256": sha256_file(runtime_path),
        "implementation_sha256": {
            "smoke_script": sha256_file(Path(__file__).resolve(strict=True)),
            "dataset_module": sha256_file(
                Path(str(refocus_g1_dataset_module.__file__)).resolve(strict=True)
            ),
            "typed_action_module": sha256_file(
                Path(str(refocus_typed_action_module.__file__)).resolve(strict=True)
            ),
        },
        "action": {
            "axis": action.axis,
            "mode": action.mode,
            "label_count": len(action.labels),
            "labels_sha256": canonical_sha256(list(action.labels)),
            "response_sha256": canonical_sha256(action_response),
        },
        "source": str(metadata["source"]),
        "original_image_sha256": _image_sha256(images[0]),
        "output_image_sha256": _image_sha256(output_image),
        "reward_target_used_for_action_selection": False,
        "raw_model_text_executed": False,
        "model_weights_loaded": False,
        "optimizer_steps": 0,
        "checkpoints_written": 0,
        "protected_split_contents_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    if decision != REPORT_DECISION:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
