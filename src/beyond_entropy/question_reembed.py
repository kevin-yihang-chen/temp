from __future__ import annotations

import hashlib
import importlib.metadata
import os
from pathlib import Path
from typing import Any

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
