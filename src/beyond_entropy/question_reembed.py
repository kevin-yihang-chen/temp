from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any, Sequence

from .dataset import read_jsonl
from .qwen_semantic import (
    load_semantic_feature_dataset,
    validate_semantic_feature_dataset,
)


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


def masked_hidden_mean(hidden_states: Any, attention_mask: Any) -> Any:
    """Mean-pool non-padding hidden states for a padded text batch."""

    if hidden_states.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("hidden states and attention mask must have [B,T,D] and [B,T]")
    if hidden_states.shape[:2] != attention_mask.shape:
        raise ValueError("hidden states and attention mask shapes do not align")
    mask = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
    counts = mask.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("cannot pool an empty token sequence")
    return (hidden_states * mask).sum(dim=1) / counts


def last_subsequence_span(
    sequence: Sequence[int],
    pattern: Sequence[int],
) -> tuple[int, int]:
    """Return the final exact occurrence of ``pattern`` in ``sequence``."""

    if not pattern:
        raise ValueError("subsequence pattern cannot be empty")
    if len(pattern) > len(sequence):
        raise ValueError("subsequence pattern is absent")
    for start in range(len(sequence) - len(pattern), -1, -1):
        stop = start + len(pattern)
        if list(sequence[start:stop]) == list(pattern):
            return start, stop
    raise ValueError("subsequence pattern is absent")


def _question_token_ids(processor: Any, prompt: str, question: str) -> list[int]:
    """Tokenize the exact question span as it appears inside a chat prompt."""

    character_start = prompt.rfind(question)
    if character_start < 0:
        raise ValueError("question is absent from the rendered chat prompt")
    character_stop = character_start + len(question)
    tokenized = processor.tokenizer(
        prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = tokenized.input_ids
    offsets = tokenized.offset_mapping
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
        offsets = offsets[0]
    selected = [
        int(token_id)
        for token_id, (start, stop) in zip(input_ids, offsets)
        if int(stop) > character_start and int(start) < character_stop
    ]
    if not selected:
        raise ValueError("question span produced no prompt tokens")
    return selected


def reembed_contextual_questions(
    *,
    source_features_path: str | Path,
    rollouts_path: str | Path,
    output_path: str | Path,
    model_name_or_path: str,
    revision: str,
    device_map: str = "cuda:0",
    dtype: str = "bfloat16",
    attention_implementation: str = "sdpa",
    batch_size: int = 64,
    checkpoint_interval: int = 512,
    local_files_only: bool = True,
    resume: bool = False,
) -> dict[str, Any]:
    """Replace only question embeddings with contextual frozen-Qwen text states."""

    import torch  # type: ignore[import-not-found]
    from transformers import (  # type: ignore[import-not-found]
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
    )

    if batch_size <= 0 or checkpoint_interval <= 0:
        raise ValueError("batch_size and checkpoint_interval must be positive")
    if not hasattr(torch, dtype):
        raise ValueError(f"unsupported torch dtype: {dtype}")
    source_path = Path(source_features_path).resolve()
    records_path = Path(rollouts_path).resolve()
    destination = Path(output_path).resolve()
    if source_path == destination:
        raise ValueError("contextual re-embedding requires a distinct output path")
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
        upgrade = result["metadata"].get("question_reembedding")
        if not isinstance(upgrade, dict):
            raise ValueError("resume output lacks question-reembedding metadata")
        if upgrade.get("source_features_sha256") != source_sha256:
            raise ValueError("resume output was initialized from different source features")
        completed = int(upgrade.get("completed_decisions", 0))
        if len(result["decisions"]) != total or not 0 <= completed <= total:
            raise ValueError("resume output has inconsistent decision counts")
    else:
        completed = 0
        metadata = dict(source["metadata"])
        metadata.update(
            {
                "question_feature_mode": "contextual_text_mean_reembedded",
                "question_feature": "mean frozen Qwen language-model hidden state",
                "question_reembedding": {
                    "source_features": str(source_path),
                    "source_features_sha256": source_sha256,
                    "source_rollouts": str(records_path),
                    "source_rollouts_sha256": rollouts_sha256,
                    "model": model_name_or_path,
                    "model_revision": revision,
                    "device_map": device_map,
                    "dtype": dtype,
                    "attention_implementation": attention_implementation,
                    "batch_size": batch_size,
                    "local_files_only": local_files_only,
                    "code_revision": os.environ.get("BE_CODE_REVISION"),
                    "completed_decisions": 0,
                    "total_decisions": total,
                    "packages": {
                        "torch": _package_version("torch"),
                        "transformers": _package_version("transformers"),
                    },
                },
            }
        )
        result = {
            "format_version": source["format_version"],
            "metadata": metadata,
            "decisions": [dict(decision) for decision in source["decisions"]],
        }
    if completed == total:
        validate_semantic_feature_dataset(result, records)
        return result

    model_kwargs: dict[str, Any] = {
        "dtype": getattr(torch, dtype),
        "device_map": device_map,
        "local_files_only": local_files_only,
        "revision": revision,
    }
    if attention_implementation:
        model_kwargs["attn_implementation"] = attention_implementation
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        local_files_only=local_files_only,
        revision=revision,
    )
    target_device = next(model.parameters()).device
    next_checkpoint = min(total, completed + checkpoint_interval)
    for start in range(completed, total, batch_size):
        stop = min(total, start + batch_size)
        questions = [
            str(result["decisions"][index]["question"]) for index in range(start, stop)
        ]
        tokenized = processor.tokenizer(
            questions,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids.to(target_device)
        attention_mask = tokenized.attention_mask.to(target_device)
        with torch.inference_mode():
            outputs = model.model.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            embeddings = masked_hidden_mean(
                outputs.last_hidden_state,
                attention_mask,
            ).to(torch.float32).cpu()
        for offset, embedding in enumerate(embeddings):
            result["decisions"][start + offset]["question_embedding"] = embedding
        completed = stop
        if completed >= next_checkpoint or completed == total:
            result["metadata"]["question_reembedding"][
                "completed_decisions"
            ] = completed
            _atomic_torch_save(result, destination)
            next_checkpoint = min(total, completed + checkpoint_interval)
    validate_semantic_feature_dataset(result, records)
    return result


def reembed_multimodal_questions(
    *,
    source_features_path: str | Path,
    rollouts_path: str | Path,
    output_path: str | Path,
    model_name_or_path: str,
    revision: str,
    device_map: str = "cuda:0",
    dtype: str = "bfloat16",
    attention_implementation: str = "sdpa",
    batch_size: int = 4,
    checkpoint_interval: int = 64,
    local_files_only: bool = True,
    resume: bool = False,
) -> dict[str, Any]:
    """Replace question embeddings with ORIGINAL-image-conditioned states.

    Question tokens occur after the image tokens in the frozen Qwen chat
    sequence, so their final hidden states can attend to the original visual
    observation. Candidate crops are never executed in this feature pass.
    """

    import torch  # type: ignore[import-not-found]
    from qwen_vl_utils import process_vision_info  # type: ignore[import-not-found]
    from transformers import (  # type: ignore[import-not-found]
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
    )

    if batch_size <= 0 or checkpoint_interval <= 0:
        raise ValueError("batch_size and checkpoint_interval must be positive")
    if not hasattr(torch, dtype):
        raise ValueError(f"unsupported torch dtype: {dtype}")
    source_path = Path(source_features_path).resolve()
    records_path = Path(rollouts_path).resolve()
    destination = Path(output_path).resolve()
    if source_path == destination:
        raise ValueError("multimodal re-embedding requires a distinct output path")
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
        upgrade = result["metadata"].get("question_reembedding")
        if not isinstance(upgrade, dict):
            raise ValueError("resume output lacks question-reembedding metadata")
        if upgrade.get("source_features_sha256") != source_sha256:
            raise ValueError("resume output was initialized from different source features")
        if result["metadata"].get("question_feature_mode") != (
            "multimodal_original_question_mean"
        ):
            raise ValueError("resume output is not a multimodal question embedding")
        completed = int(upgrade.get("completed_decisions", 0))
        if len(result["decisions"]) != total or not 0 <= completed <= total:
            raise ValueError("resume output has inconsistent decision counts")
    else:
        completed = 0
        metadata = dict(source["metadata"])
        metadata.update(
            {
                "question_feature_mode": "multimodal_original_question_mean",
                "question_feature": (
                    "mean frozen Qwen final hidden state over exact user question "
                    "tokens conditioned on the ORIGINAL image"
                ),
                "question_reembedding": {
                    "source_features": str(source_path),
                    "source_features_sha256": source_sha256,
                    "source_rollouts": str(records_path),
                    "source_rollouts_sha256": rollouts_sha256,
                    "model": model_name_or_path,
                    "model_revision": revision,
                    "device_map": device_map,
                    "dtype": dtype,
                    "attention_implementation": attention_implementation,
                    "batch_size": batch_size,
                    "local_files_only": local_files_only,
                    "conditioning": "one ORIGINAL image and the user question",
                    "candidate_actions_executed": False,
                    "code_revision": os.environ.get("BE_CODE_REVISION"),
                    "completed_decisions": 0,
                    "total_decisions": total,
                    "packages": {
                        "torch": _package_version("torch"),
                        "transformers": _package_version("transformers"),
                    },
                },
            }
        )
        result = {
            "format_version": source["format_version"],
            "metadata": metadata,
            "decisions": [dict(decision) for decision in source["decisions"]],
        }
    if completed == total:
        validate_semantic_feature_dataset(result, records)
        return result

    model_kwargs: dict[str, Any] = {
        "dtype": getattr(torch, dtype),
        "device_map": device_map,
        "local_files_only": local_files_only,
        "revision": revision,
    }
    if attention_implementation:
        model_kwargs["attn_implementation"] = attention_implementation
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        **model_kwargs,
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
    next_checkpoint = min(total, completed + checkpoint_interval)
    for start in range(completed, total, batch_size):
        stop = min(total, start + batch_size)
        prompts: list[str] = []
        question_token_ids: list[list[int]] = []
        image_inputs: list[Any] = []
        for index in range(start, stop):
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
            images, videos = process_vision_info(messages)
            if videos:
                raise ValueError("question re-embedding only supports images")
            if len(images) != 1:
                raise ValueError("each decision must contain exactly one original image")
            prompts.append(prompt)
            question_token_ids.append(_question_token_ids(processor, prompt, question))
            image_inputs.extend(images)
        inputs = processor(
            text=prompts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(target_device)
        with torch.inference_mode():
            outputs = model.model(
                **inputs,
                use_cache=False,
                return_dict=True,
            )
        hidden_states = outputs.last_hidden_state
        input_ids = inputs.input_ids
        for offset, token_ids in enumerate(question_token_ids):
            token_start, token_stop = last_subsequence_span(
                [int(value) for value in input_ids[offset].detach().cpu().tolist()],
                token_ids,
            )
            embedding = hidden_states[offset, token_start:token_stop].mean(dim=0)
            result["decisions"][start + offset]["question_embedding"] = (
                embedding.detach().to(torch.float32).cpu()
            )
        completed = stop
        print(
            json.dumps(
                {
                    "checkpoint": str(destination),
                    "completed_this_run": completed - start,
                    "total_completed": completed,
                    "total_decisions": total,
                }
            ),
            flush=True,
        )
        if completed >= next_checkpoint or completed == total:
            result["metadata"]["question_reembedding"][
                "completed_decisions"
            ] = completed
            _atomic_torch_save(result, destination)
            next_checkpoint = min(total, completed + checkpoint_interval)
    validate_semantic_feature_dataset(result, records)
    return result
