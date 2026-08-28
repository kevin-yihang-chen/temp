from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

from .dataset import read_jsonl
from .question_reembed import (
    last_subsequence_span,
    question_token_ids_from_prompt,
)
from .qwen_semantic import (
    load_semantic_feature_dataset,
    reshape_merged_visual_tokens,
    validate_semantic_feature_dataset,
)
from .semantic import require_torch, roi_pool_spatial_tokens


def normalized_question_region_attention(attention_grid: Any, bboxes: Any) -> Any:
    """Pool a question-to-image attention map into candidate relevance shares."""

    require_torch()
    import torch  # type: ignore[import-not-found]

    if attention_grid.ndim != 2:
        raise ValueError("attention grid must have shape [height, width]")
    if bboxes.ndim != 2 or bboxes.shape[-1] != 4 or bboxes.shape[0] == 0:
        raise ValueError("bboxes must have shape [candidates, 4]")
    nonnegative = attention_grid.float().clamp_min(0.0)
    total = nonnegative.sum()
    if not bool(torch.isfinite(total)) or float(total) <= 0.0:
        raise ValueError("attention grid must have positive finite mass")
    normalized_grid = nonnegative / total
    densities = roi_pool_spatial_tokens(
        normalized_grid[None, :, :, None],
        bboxes.float()[None, :, :],
    )[0, :, 0]
    density_total = densities.sum()
    if not bool(torch.isfinite(density_total)) or float(density_total) <= 0.0:
        raise ValueError("candidate regions received no finite attention")
    return densities / density_total


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _atomic_torch_save(value: object, destination: Path) -> None:
    import torch  # type: ignore[import-not-found]

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(destination)


def augment_question_region_attention(
    *,
    source_features_path: str | Path,
    rollouts_path: str | Path,
    output_path: str | Path,
    model_name_or_path: str,
    revision: str,
    device_map: str = "cuda:0",
    dtype: str = "bfloat16",
    top_layers: int = 4,
    checkpoint_interval: int = 32,
    local_files_only: bool = True,
    resume: bool = False,
) -> dict[str, Any]:
    """Add zero-shot question-to-region attention without executing crops."""

    require_torch()
    import torch  # type: ignore[import-not-found]
    from qwen_vl_utils import process_vision_info  # type: ignore[import-not-found]
    from transformers import (  # type: ignore[import-not-found]
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
    )

    if top_layers <= 0 or checkpoint_interval <= 0:
        raise ValueError("top_layers and checkpoint_interval must be positive")
    if not hasattr(torch, dtype):
        raise ValueError(f"unsupported torch dtype: {dtype}")
    source_path = Path(source_features_path).resolve()
    records_path = Path(rollouts_path).resolve()
    destination = Path(output_path).resolve()
    if source_path == destination:
        raise ValueError("attention augmentation requires a distinct output path")
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    rollouts_sha256 = hashlib.sha256(records_path.read_bytes()).hexdigest()
    records = read_jsonl(records_path)
    source = load_semantic_feature_dataset(source_path)
    validate_semantic_feature_dataset(source, records)
    total = len(source["decisions"])
    if destination.exists():
        if not resume:
            raise FileExistsError(f"output already exists: {destination}")
        result = load_semantic_feature_dataset(destination)
        augmentation = result["metadata"].get("question_region_attention")
        if not isinstance(augmentation, dict):
            raise ValueError("resume output lacks attention metadata")
        if augmentation.get("source_features_sha256") != source_sha256:
            raise ValueError("resume output was initialized from different features")
        if int(augmentation.get("top_layers", 0)) != top_layers:
            raise ValueError("resume output uses different attention layers")
        completed = int(augmentation.get("completed_decisions", 0))
        if len(result["decisions"]) != total or not 0 <= completed <= total:
            raise ValueError("resume output has inconsistent decision counts")
    else:
        completed = 0
        metadata = dict(source["metadata"])
        metadata["question_region_attention"] = {
            "scientific_status": "development-only pre-action feature",
            "source_features": str(source_path),
            "source_features_sha256": source_sha256,
            "source_rollouts": str(records_path),
            "source_rollouts_sha256": rollouts_sha256,
            "model": model_name_or_path,
            "model_revision": revision,
            "device_map": device_map,
            "dtype": dtype,
            "attention_implementation": "eager",
            "top_layers": top_layers,
            "head_pooling": "mean",
            "question_token_pooling": "mean",
            "candidate_pooling": "ROI mean then normalize across candidates",
            "candidate_actions_executed": False,
            "code_revision": os.environ.get("BE_CODE_REVISION"),
            "completed_decisions": 0,
            "total_decisions": total,
            "packages": {
                "torch": _package_version("torch"),
                "transformers": _package_version("transformers"),
            },
        }
        result = {
            "format_version": source["format_version"],
            "metadata": metadata,
            "decisions": [dict(decision) for decision in source["decisions"]],
        }
    if completed == total:
        validate_semantic_feature_dataset(result, records)
        return result

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        dtype=getattr(torch, dtype),
        device_map=device_map,
        local_files_only=local_files_only,
        revision=revision,
        attn_implementation="eager",
    ).eval()
    min_pixels = int(source["metadata"].get("min_pixels", 256 * 28 * 28))
    max_pixels = int(source["metadata"].get("max_pixels", 768 * 28 * 28))
    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        local_files_only=local_files_only,
        revision=revision,
    )
    target_device = next(model.parameters()).device
    image_token_id = int(model.config.image_token_id)
    merge_size = int(model.model.visual.spatial_merge_size)
    next_checkpoint = min(total, completed + checkpoint_interval)
    for index in range(completed, total):
        decision = result["decisions"][index]
        question = str(decision["question"])
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": str(Path(decision["original_image"]).resolve()),
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                    },
                    {"type": "text", "text": question},
                ],
            },
        ]
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        exact_question_ids = question_token_ids_from_prompt(
            processor,
            prompt,
            question,
        )
        images, videos = process_vision_info(messages)
        if videos or len(images) != 1:
            raise ValueError("each attention decision requires exactly one image")
        inputs = processor(
            text=[prompt],
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(target_device)
        with torch.inference_mode():
            outputs = model.model(
                **inputs,
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )
        attentions = outputs.attentions
        if attentions is None or len(attentions) < top_layers:
            raise RuntimeError("Qwen did not return the requested attention layers")
        token_start, token_stop = last_subsequence_span(
            [int(value) for value in inputs.input_ids[0].detach().cpu().tolist()],
            exact_question_ids,
        )
        selected_layers = attentions[-top_layers:]
        if any(layer is None for layer in selected_layers):
            raise RuntimeError("Qwen returned an empty attention layer")
        stacked = torch.stack([layer[0].float() for layer in selected_layers])
        key_attention = stacked[:, :, token_start:token_stop, :].mean(dim=(0, 1, 2))
        image_positions = torch.nonzero(
            inputs.input_ids[0] == image_token_id,
            as_tuple=False,
        ).flatten()
        image_attention = key_attention[image_positions]
        grid = reshape_merged_visual_tokens(
            image_attention[:, None],
            inputs.image_grid_thw[0],
            spatial_merge_size=merge_size,
        )[:, :, 0]
        region_attention = normalized_question_region_attention(
            grid,
            decision["bboxes"].to(target_device),
        )
        decision["question_region_attention"] = (
            region_attention.detach().to(torch.float32).cpu()
        )
        decision["question_image_attention_mass"] = float(
            image_attention.sum().detach().cpu()
        )
        completed = index + 1
        if completed >= next_checkpoint or completed == total:
            result["metadata"]["question_region_attention"][
                "completed_decisions"
            ] = completed
            _atomic_torch_save(result, destination)
            print(
                json.dumps(
                    {
                        "checkpoint": str(destination),
                        "total_completed": completed,
                        "total_decisions": total,
                    }
                ),
                flush=True,
            )
            next_checkpoint = min(total, completed + checkpoint_interval)
    validate_semantic_feature_dataset(result, records)
    return result
