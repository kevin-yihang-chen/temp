from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .attention_features import _atomic_torch_save, _package_version
from .dataset import read_jsonl
from .infographicvqa_literature_attention_where import (
    VICROP_ANSWER_SUFFIX,
    VICROP_GENERIC_QUESTION,
    VICROP_QWEN25_LAYER_INDEX,
    image_attention_entropy,
    laser_all_head_candidate_scores,
    vicrop_relative_candidate_scores,
)
from .qwen_semantic import (
    load_semantic_feature_dataset,
    reshape_merged_visual_tokens,
    validate_semantic_feature_dataset,
)
from .semantic import require_torch

LITERATURE_ATTENTION_METADATA_KEY = "literature_attention_where"
LITERATURE_ATTENTION_FORMAT_VERSION = 1
LITERATURE_ATTENTION_SYSTEM_PROMPT = "You are a helpful assistant."
LITERATURE_ATTENTION_ENCORE_LAYERS = (0, 1)


def literature_prefill_texts(question: str) -> tuple[str, str, None]:
    if not question:
        raise ValueError("literature attention requires a non-empty question")
    return (
        question + VICROP_ANSWER_SUFFIX,
        VICROP_GENERIC_QUESTION + VICROP_ANSWER_SUFFIX,
        None,
    )


def literature_messages(
    *,
    image_path: str | Path,
    text: str | None,
    min_pixels: int,
    max_pixels: int,
) -> list[dict[str, Any]]:
    if min_pixels <= 0 or max_pixels < min_pixels:
        raise ValueError("literature attention pixel bounds are invalid")
    content: list[dict[str, Any]] = [
        {
            "type": "image",
            "image": str(Path(image_path).resolve()),
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        }
    ]
    if text is not None:
        content.append({"type": "text", "text": text})
    return [
        {"role": "system", "content": LITERATURE_ATTENTION_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def visual_attention_grids(
    attentions: Sequence[Any],
    input_ids: Any,
    image_grid_thw: Any,
    *,
    image_token_id: int,
    spatial_merge_size: int,
) -> Any:
    """Restore final-prefill image attention as [layers, heads, height, width]."""

    require_torch()
    import torch  # type: ignore[import-not-found]

    if not attentions or spatial_merge_size <= 0:
        raise ValueError("literature attention output is empty or malformed")
    if input_ids.ndim != 1:
        raise ValueError("literature attention input IDs must be one-dimensional")
    image_positions = torch.nonzero(
        input_ids == image_token_id, as_tuple=False
    ).flatten()
    if image_positions.numel() == 0:
        raise ValueError("literature prefill contains no image tokens")
    rows: list[Any] = []
    expected_heads: int | None = None
    for layer in attentions:
        if (
            layer is None
            or layer.ndim != 4
            or layer.shape[0] != 1
            or layer.shape[2] != input_ids.numel()
            or layer.shape[3] != input_ids.numel()
        ):
            raise ValueError("literature attention layer shape changed")
        head_count = int(layer.shape[1])
        if expected_heads is None:
            expected_heads = head_count
        elif head_count != expected_heads:
            raise ValueError("literature attention head count changed across layers")
        rows.append(layer[0, :, -1, image_positions].float())
    stacked = torch.stack(rows)
    if not bool(torch.isfinite(stacked).all()) or bool((stacked < 0.0).any()):
        raise ValueError("literature image attention is nonfinite or negative")
    layers, heads, tokens = stacked.shape
    restored = reshape_merged_visual_tokens(
        stacked.permute(2, 0, 1).reshape(tokens, layers * heads),
        image_grid_thw,
        spatial_merge_size=spatial_merge_size,
    )
    height, width, _ = restored.shape
    return restored.reshape(height, width, layers, heads).permute(2, 3, 0, 1)


def _extract_prefill_grids(
    *,
    model: Any,
    processor: Any,
    process_vision_info: Callable[[Any], tuple[Any, Any]],
    messages: list[dict[str, Any]],
    target_device: Any,
    image_token_id: int,
    spatial_merge_size: int,
) -> tuple[Any, tuple[int, int, int]]:
    require_torch()
    import torch  # type: ignore[import-not-found]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    images, videos = process_vision_info(messages)
    if videos or len(images) != 1:
        raise ValueError("each literature prefill requires exactly one image")
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
    if outputs.attentions is None:
        raise RuntimeError("Qwen returned no attention for literature prefill")
    grids = visual_attention_grids(
        outputs.attentions,
        inputs.input_ids[0],
        inputs.image_grid_thw[0],
        image_token_id=image_token_id,
        spatial_merge_size=spatial_merge_size,
    )
    raw_grid_values = [int(value) for value in inputs.image_grid_thw[0].tolist()]
    if len(raw_grid_values) != 3:
        raise ValueError("literature prefill image grid must have three dimensions")
    raw_grid = (
        raw_grid_values[0],
        raw_grid_values[1],
        raw_grid_values[2],
    )
    del outputs, inputs
    return grids, raw_grid


def _expected_resume_metadata(
    *,
    source_sha256: str,
    rollouts_sha256: str,
    model_name_or_path: str,
    revision: str,
    device_map: str,
    dtype: str,
) -> dict[str, Any]:
    return {
        "format_version": LITERATURE_ATTENTION_FORMAT_VERSION,
        "source_features_sha256": source_sha256,
        "source_rollouts_sha256": rollouts_sha256,
        "model": model_name_or_path,
        "model_revision": revision,
        "device_map": device_map,
        "dtype": dtype,
        "attention_implementation": "eager",
        "prefill_query_position": "final assistant-prefix token",
        "query_suffix": VICROP_ANSWER_SUFFIX,
        "generic_question": VICROP_GENERIC_QUESTION,
        "no_query_text_content": False,
        "vicrop_layer_index": VICROP_QWEN25_LAYER_INDEX,
        "vicrop_head_pooling": "mean all returned heads",
        "vicrop_formula": "query_attention / generic_attention; no epsilon",
        "laser_head_pooling": "all heads",
        "laser_layer_selection": "mean head L2 norm of positive query-minus-no-query contrast",
        "encore_layers": list(LITERATURE_ATTENTION_ENCORE_LAYERS),
        "encore_head_pooling": "mean all returned heads before normalized Shannon entropy",
        "candidate_pooling": "ROI mean then normalize across candidates",
        "candidate_actions_executed": False,
        "outcomes_included": False,
        "validation_or_test_inputs_used": False,
        "code_revision": os.environ.get("BE_CODE_REVISION"),
    }


def augment_literature_attention_where(
    *,
    source_features_path: str | Path,
    rollouts_path: str | Path,
    output_path: str | Path,
    model_name_or_path: str,
    revision: str,
    device_map: str = "cuda:0",
    dtype: str = "bfloat16",
    checkpoint_interval: int = 256,
    local_files_only: bool = True,
    resume: bool = False,
) -> dict[str, Any]:
    """Extract blind ViCrop/LASER-bank scores and ENCORE diagnostics."""

    require_torch()
    import torch  # type: ignore[import-not-found]
    from qwen_vl_utils import process_vision_info  # type: ignore[import-not-found]
    from transformers import AutoProcessor  # type: ignore[import-not-found]
    from transformers import Qwen2_5_VLForConditionalGeneration

    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    if not hasattr(torch, dtype):
        raise ValueError(f"unsupported torch dtype: {dtype}")
    source_path = Path(source_features_path).resolve()
    records_path = Path(rollouts_path).resolve()
    destination = Path(output_path).resolve()
    if source_path == destination:
        raise ValueError("literature attention requires a distinct output path")
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    rollouts_sha256 = hashlib.sha256(records_path.read_bytes()).hexdigest()
    records = read_jsonl(records_path)
    source = load_semantic_feature_dataset(source_path)
    validate_semantic_feature_dataset(source, records, require_outcomes=False)
    if bool(source["metadata"].get("outcomes_included", True)):
        raise ValueError("literature attention source features include outcomes")
    total = len(source["decisions"])
    expected_metadata = _expected_resume_metadata(
        source_sha256=source_sha256,
        rollouts_sha256=rollouts_sha256,
        model_name_or_path=model_name_or_path,
        revision=revision,
        device_map=device_map,
        dtype=dtype,
    )
    if destination.exists():
        if not resume:
            raise FileExistsError(f"output already exists: {destination}")
        result = load_semantic_feature_dataset(destination)
        augmentation = result["metadata"].get(LITERATURE_ATTENTION_METADATA_KEY)
        if not isinstance(augmentation, Mapping):
            raise ValueError("resume output lacks literature attention metadata")
        for name, expected in expected_metadata.items():
            if augmentation.get(name) != expected:
                raise ValueError(f"resume literature attention changed for {name}")
        completed = int(augmentation.get("completed_decisions", -1))
        if len(result["decisions"]) != total or not 0 <= completed <= total:
            raise ValueError("resume output has inconsistent decision counts")
    else:
        completed = 0
        metadata = dict(source["metadata"])
        metadata[LITERATURE_ATTENTION_METADATA_KEY] = {
            **expected_metadata,
            "scientific_status": "frozen outcome-free literature comparator feature",
            "source_features": str(source_path),
            "source_rollouts": str(records_path),
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
        validate_semantic_feature_dataset(result, records, require_outcomes=False)
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
        texts = literature_prefill_texts(str(decision["question"]))
        grids: list[Any] = []
        raw_grids: list[tuple[int, int, int]] = []
        for text in texts:
            messages = literature_messages(
                image_path=str(decision["original_image"]),
                text=text,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            attention, raw_grid = _extract_prefill_grids(
                model=model,
                processor=processor,
                process_vision_info=process_vision_info,
                messages=messages,
                target_device=target_device,
                image_token_id=image_token_id,
                spatial_merge_size=merge_size,
            )
            grids.append(attention)
            raw_grids.append(raw_grid)
        query, generic, no_query = grids
        if query.shape != generic.shape or query.shape != no_query.shape:
            raise ValueError("literature prefill attention shapes differ")
        if raw_grids[0] != raw_grids[1] or raw_grids[0] != raw_grids[2]:
            raise ValueError("literature prefill visual grids differ")
        if query.shape[0] <= max(
            VICROP_QWEN25_LAYER_INDEX, *LITERATURE_ATTENTION_ENCORE_LAYERS
        ):
            raise ValueError("Qwen has fewer layers than the frozen protocol requires")
        bboxes = decision["bboxes"].to(target_device)
        vicrop = vicrop_relative_candidate_scores(
            query[VICROP_QWEN25_LAYER_INDEX].mean(dim=0),
            generic[VICROP_QWEN25_LAYER_INDEX].mean(dim=0),
            bboxes,
        )
        laser = laser_all_head_candidate_scores(query, no_query, bboxes)
        if laser.selected_layer is None or laser.layer_scores is None:
            raise RuntimeError("LASER adaptation returned incomplete diagnostics")
        encore = torch.tensor(
            [
                image_attention_entropy(query[layer].mean(dim=0))
                for layer in LITERATURE_ATTENTION_ENCORE_LAYERS
            ],
            dtype=torch.float32,
        )
        layer = VICROP_QWEN25_LAYER_INDEX
        masses = torch.tensor(
            [
                float(query[layer].mean(dim=0).sum()),
                float(generic[layer].mean(dim=0).sum()),
                float(no_query[layer].mean(dim=0).sum()),
            ],
            dtype=torch.float32,
        )
        decision["vicrop_relative_region_attention"] = (
            vicrop.candidate_scores.detach().to(torch.float32).cpu()
        )
        decision["laser_contrastive_region_attention"] = (
            laser.candidate_scores.detach().to(torch.float32).cpu()
        )
        decision["laser_selected_layer"] = int(laser.selected_layer)
        decision["laser_layer_scores"] = (
            laser.layer_scores.detach().to(torch.float32).cpu()
        )
        decision["laser_zero_map_fallback"] = bool(laser.zero_map_fallback)
        decision["encore_early_entropy"] = encore
        decision["literature_attention_image_mass"] = masses
        decision["literature_attention_grid_thw"] = torch.tensor(
            raw_grids[0], dtype=torch.int32
        )
        decision["literature_attention_layer_count"] = int(query.shape[0])
        decision["literature_attention_head_count"] = int(query.shape[1])
        del grids, query, generic, no_query
        completed = index + 1
        if completed >= next_checkpoint or completed == total:
            result["metadata"][LITERATURE_ATTENTION_METADATA_KEY][
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
    validate_semantic_feature_dataset(result, records, require_outcomes=False)
    return result
