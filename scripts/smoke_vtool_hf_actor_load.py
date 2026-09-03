#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from typing import Any, Mapping


REPORT_SCHEMA = "vtool_hf_actor_load_smoke_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exercise the pinned verl Hugging Face actor dispatch with the frozen "
            "attention backend, optionally including a real-image GPU forward."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("meta", "gpu-forward"), required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def validate_actor_runtime_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("schema") != "vtool_action_credit_g1_config_v1":
        raise ValueError("unsupported G1 config schema")
    model = _mapping(config.get("model"), name="model")
    training = _mapping(config.get("training"), name="training")
    runtime = _mapping(config.get("runtime"), name="runtime")
    backend = str(training.get("actor_attention_implementation"))
    use_remove_padding = training.get("actor_use_remove_padding")
    dtype = str(training.get("actor_dtype"))
    if backend != "sdpa":
        raise ValueError("actor attention implementation must be sdpa")
    if use_remove_padding is not False:
        raise ValueError(
            "remove-padding must be false because the pinned Qwen patch otherwise "
            "dispatches to FlashAttention"
        )
    if dtype != "bfloat16":
        raise ValueError("actor dtype must remain bfloat16")
    model_path = Path(str(model["local_snapshot"])).resolve(strict=True)
    if model_path.name != str(model["revision"]):
        raise ValueError("model snapshot revision mismatch")
    return {
        "attention_implementation": backend,
        "dtype": dtype,
        "model_path": model_path,
        "model_revision": str(model["revision"]),
        "runtime_commit": str(runtime["upstream_commit"]),
        "use_remove_padding": use_remove_padding,
    }


def _actor_model_class(actor_config: Any) -> Any:
    from transformers import AutoModel, AutoModelForCausalLM
    from transformers import AutoModelForImageTextToText
    from verl.utils.transformers_compat import get_auto_model_for_vision2seq

    auto_vision = get_auto_model_for_vision2seq()
    if type(actor_config) in auto_vision._model_mapping.keys():
        return auto_vision
    if type(actor_config) in AutoModelForCausalLM._model_mapping.keys():
        return AutoModelForCausalLM
    if type(actor_config) in AutoModelForImageTextToText._model_mapping.keys():
        return AutoModelForImageTextToText
    return AutoModel


def _attention_backends(actor_config: Any) -> dict[str, object]:
    def value(config: Any) -> object:
        return getattr(config, "_attn_implementation", None)

    return {
        "model": value(actor_config),
        "text": value(getattr(actor_config, "text_config", object())),
        "vision": value(getattr(actor_config, "vision_config", object())),
    }


def _load_actor_model(
    model_path: Path, *, backend: str, meta: bool
) -> tuple[Any, Any, float]:
    import torch
    from transformers import AutoConfig

    actor_config = AutoConfig.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        attn_implementation=backend,
    )
    actor_class = _actor_model_class(actor_config)
    started = time.perf_counter()
    if meta:
        from accelerate import init_empty_weights

        with init_empty_weights():
            model = actor_class.from_config(
                config=actor_config,
                trust_remote_code=False,
                attn_implementation=backend,
            )
    else:
        model = actor_class.from_pretrained(
            pretrained_model_name_or_path=model_path,
            torch_dtype=torch.float32,
            config=actor_config,
            local_files_only=True,
            trust_remote_code=False,
            attn_implementation=backend,
        )
    return model, actor_config, time.perf_counter() - started


def _apply_pinned_actor_patch(
    model: Any, *, use_remove_padding: bool
) -> dict[str, bool]:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLAttention,
        Qwen2_5_VLForConditionalGeneration,
    )
    from verl.models.transformers.monkey_patch import apply_monkey_patch

    attention_forward_before = Qwen2_5_VLAttention.forward
    model_forward_before = Qwen2_5_VLForConditionalGeneration.forward
    apply_monkey_patch(
        model=model,
        use_remove_padding=use_remove_padding,
        ulysses_sp_size=1,
        use_fused_kernels=False,
        use_prefix_grouper=False,
        use_tiled_mlp=False,
    )
    return {
        "native_attention_forward_preserved": (
            Qwen2_5_VLAttention.forward is attention_forward_before
        ),
        "verl_multimodal_model_forward_applied": (
            Qwen2_5_VLForConditionalGeneration.forward is not model_forward_before
        ),
    }


def _prepare_real_image_inputs(
    dataset_path: Path, model_path: Path
) -> tuple[Any, dict[str, Any], int]:
    from omegaconf import OmegaConf
    from verl.utils import hf_processor
    from verl.utils.chat_template import apply_chat_template
    from verl.utils.dataset.rl_dataset import RLHFDataset

    processor = hf_processor(
        model_path,
        trust_remote_code=False,
        local_files_only=True,
    )
    if processor is None:
        raise RuntimeError("pinned model did not produce a multimodal processor")
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
        raise ValueError("actor-load smoke dataset must contain exactly one row")
    item = dataset[0]
    images, videos = asyncio.run(
        RLHFDataset.process_vision_info(
            item["raw_prompt"],
            image_patch_size=processor.image_processor.patch_size,
            config=data_config,
        )
    )
    if images is None or len(images) != 1 or videos:
        raise ValueError("actor-load smoke expects one image and no videos")
    rendered = apply_chat_template(
        processor,
        item["raw_prompt"],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = processor(text=[rendered], images=images, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    vision_position_ids, _ = processor.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=inputs.get("image_grid_thw"),
        video_grid_thw=inputs.get("video_grid_thw"),
        attention_mask=attention_mask,
    )
    vision_position_ids = vision_position_ids.transpose(0, 1)
    valid_mask = attention_mask[0].bool()
    text_position_ids = attention_mask.new_ones((1, len(input_ids[0])))
    text_position_ids[0, valid_mask] = attention_mask.new_tensor(
        range(int(valid_mask.sum().item()))
    )
    position_ids = torch_cat_position_ids(text_position_ids, vision_position_ids)
    model_inputs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids.transpose(0, 1),
        "pixel_values": inputs["pixel_values"],
        "image_grid_thw": inputs["image_grid_thw"],
        "use_cache": False,
    }
    return item, model_inputs, int(input_ids.shape[-1])


def torch_cat_position_ids(text_position_ids: Any, vision_position_ids: Any) -> Any:
    import torch

    return torch.cat((text_position_ids.unsqueeze(0), vision_position_ids), dim=1)


def _write_report(path: Path, report: Mapping[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite actor-load report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("G1 config must contain a JSON mapping")
    frozen = validate_actor_runtime_config(config)
    model_path = Path(frozen["model_path"])
    backend = str(frozen["attention_implementation"])
    use_remove_padding = bool(frozen["use_remove_padding"])

    import torch
    import transformers

    gpu_forward = args.mode == "gpu-forward"
    if gpu_forward:
        if args.dataset is None:
            raise ValueError("--dataset is required in gpu-forward mode")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("GPU actor-load smoke requires exactly one visible GPU")
        if "H800" not in torch.cuda.get_device_name(0):
            raise RuntimeError("GPU actor-load smoke requires an H800")
        free_before, total_bytes = torch.cuda.mem_get_info()
    else:
        free_before, total_bytes = 0, 0

    model, actor_config, load_seconds = _load_actor_model(
        model_path,
        backend=backend,
        meta=not gpu_forward,
    )
    patch_checks = _apply_pinned_actor_patch(
        model,
        use_remove_padding=use_remove_padding,
    )
    checks = {
        "attention_backend_is_sdpa": all(
            value == "sdpa" for value in _attention_backends(actor_config).values()
        ),
        "flash_attn_not_required": backend == "sdpa" and not use_remove_padding,
        **patch_checks,
    }
    report: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "mode": args.mode,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "smoke_script_sha256": sha256_file(Path(__file__).resolve()),
        "model_path": str(model_path),
        "model_revision": frozen["model_revision"],
        "runtime_commit": frozen["runtime_commit"],
        "actor_class": type(model).__name__,
        "attention_implementation": backend,
        "resolved_attention_backends": _attention_backends(actor_config),
        "use_remove_padding": use_remove_padding,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "flash_attn_installed": importlib.util.find_spec("flash_attn") is not None,
        "load_seconds": load_seconds,
        "checks": checks,
        "model_weights_loaded": gpu_forward,
        "optimizer_step_performed": False,
        "protected_split_contents_accessed": False,
    }

    if gpu_forward:
        dataset_path = args.dataset.resolve(strict=True)
        _, model_inputs, prompt_tokens = _prepare_real_image_inputs(
            dataset_path,
            model_path,
        )
        model = model.to(device="cuda", dtype=torch.bfloat16)
        model.eval()
        moved_inputs = {
            key: (value.to("cuda") if isinstance(value, torch.Tensor) else value)
            for key, value in model_inputs.items()
        }
        torch.cuda.reset_peak_memory_stats()
        forward_started = time.perf_counter()
        with torch.no_grad():
            output = model(**moved_inputs)
        torch.cuda.synchronize()
        forward_seconds = time.perf_counter() - forward_started
        last_logits = output.logits[:, -1, :].float()
        finite = bool(torch.isfinite(last_logits).all().item())
        checks["real_image_forward_completed"] = True
        checks["last_token_logits_finite"] = finite
        report.update(
            {
                "dataset": str(dataset_path),
                "dataset_sha256": sha256_file(dataset_path),
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_total_bytes": int(total_bytes),
                "gpu_free_before_bytes": int(free_before),
                "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "forward_seconds": forward_seconds,
                "prompt_tokens": prompt_tokens,
                "logits_shape": list(output.logits.shape),
                "last_token_logits_sha256": hashlib.sha256(
                    last_logits.cpu().numpy().tobytes()
                ).hexdigest(),
            }
        )

    passed = bool(checks) and all(value is True for value in checks.values())
    report["decision"] = (
        "vtool_hf_actor_gpu_forward_smoke_passed"
        if gpu_forward and passed
        else (
            "vtool_hf_actor_meta_dispatch_passed"
            if passed
            else "vtool_hf_actor_load_smoke_failed"
        )
    )
    _write_report(args.output.resolve(), report)
    print(json.dumps(report, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
