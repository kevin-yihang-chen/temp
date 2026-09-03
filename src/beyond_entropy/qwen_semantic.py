from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .dataset import group_by_decision, read_jsonl
from .schema import ActionRecord, BBox
from .semantic import require_torch, roi_pool_spatial_tokens


SEMANTIC_FEATURE_FORMAT_VERSION = 1
SEMANTIC_OUTCOME_FIELDS = frozenset(
    {
        "answer_after",
        "correct_after",
        "correct_before",
        "delta_success",
        "entropy_after",
        "success_after",
        "success_before",
    }
)


def reshape_merged_visual_tokens(
    merged_tokens: Any,
    grid_thw: Any,
    *,
    spatial_merge_size: int,
) -> Any:
    """Restore Qwen merged visual tokens to a raster-ordered spatial grid."""

    require_torch()
    if merged_tokens.ndim != 2:
        raise ValueError("merged_tokens must have shape [tokens, visual_dim]")
    if spatial_merge_size <= 0:
        raise ValueError("spatial_merge_size must be positive")
    values = [int(value) for value in grid_thw.detach().cpu().reshape(-1).tolist()]
    if len(values) != 3:
        raise ValueError("grid_thw must contain exactly t, h, and w")
    temporal, height, width = values
    if min(temporal, height, width) <= 0:
        raise ValueError("grid_thw dimensions must be positive")
    if height % spatial_merge_size or width % spatial_merge_size:
        raise ValueError("Qwen visual grid is not divisible by spatial_merge_size")
    merged_height = height // spatial_merge_size
    merged_width = width // spatial_merge_size
    expected = temporal * merged_height * merged_width
    if merged_tokens.shape[0] != expected:
        raise ValueError(
            f"merged token count mismatch: expected {expected}, got {merged_tokens.shape[0]}"
        )
    temporal_grid = merged_tokens.reshape(
        temporal,
        merged_height,
        merged_width,
        merged_tokens.shape[-1],
    )
    return temporal_grid.mean(dim=0)


def pool_multimodal_prompt_states(
    last_hidden_state: Any,
    input_ids: Any,
    attention_mask: Any,
    *,
    image_token_id: int,
) -> dict[str, Any]:
    """Pool frozen pre-generation Qwen states without outcome information."""

    require_torch()
    import torch  # type: ignore[import-not-found]

    if (
        getattr(last_hidden_state, "ndim", None) != 3
        or getattr(input_ids, "ndim", None) != 2
        or getattr(attention_mask, "ndim", None) != 2
        or last_hidden_state.shape[:2] != input_ids.shape
        or input_ids.shape != attention_mask.shape
        or last_hidden_state.shape[0] != 1
    ):
        raise ValueError("multimodal prompt tensors have incompatible shapes")
    attended = attention_mask[0].to(dtype=bool)
    image = attended & (input_ids[0] == image_token_id)
    language = attended & ~image
    if not bool(image.any()) or not bool(language.any()):
        raise ValueError("multimodal prompt must contain image and language tokens")
    attended_positions = attended.nonzero(as_tuple=False).reshape(-1)
    final_position = int(attended_positions[-1].item())
    hidden = last_hidden_state[0]
    return {
        "pooled_language_state": hidden[language]
        .mean(dim=0)
        .detach()
        .to(torch.float32)
        .cpu(),
        "pooled_visual_state": hidden[image]
        .mean(dim=0)
        .detach()
        .to(torch.float32)
        .cpu(),
        "fused_multimodal_state": hidden[final_position]
        .detach()
        .to(torch.float32)
        .cpu(),
        "multimodal_prompt_tokens": int(attended.sum().item()),
        "multimodal_image_tokens": int(image.sum().item()),
        "multimodal_language_tokens": int(language.sum().item()),
    }


class Qwen25VLSemanticExtractor:
    """Extract pre-action Qwen question/image/ROI features with one image pass."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        revision: str = "main",
        device_map: str = "cuda:0",
        dtype: str = "bfloat16",
        attention_implementation: str = "sdpa",
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 768 * 28 * 28,
        local_files_only: bool = True,
        question_feature_mode: str = "input_mean",
    ) -> None:
        require_torch()
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import (  # type: ignore[import-not-found]
                AutoProcessor,
                Qwen2_5_VLForConditionalGeneration,
            )
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError(
                "Qwen semantic extraction requires torch, transformers, and Pillow"
            ) from exc
        if not hasattr(torch, dtype):
            raise ValueError(f"unsupported torch dtype: {dtype}")
        if min_pixels <= 0 or max_pixels < min_pixels:
            raise ValueError("pixel limits must be positive and ordered")
        if question_feature_mode not in ("input_mean", "contextual_text_mean"):
            raise ValueError(
                f"unsupported question_feature_mode: {question_feature_mode}"
            )
        model_kwargs: dict[str, Any] = {
            "dtype": getattr(torch, dtype),
            "device_map": device_map,
            "local_files_only": local_files_only,
            "revision": revision,
        }
        if attention_implementation:
            model_kwargs["attn_implementation"] = attention_implementation
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            local_files_only=local_files_only,
            revision=revision,
        )
        self.model_name_or_path = model_name_or_path
        self.revision = revision
        self.device_map = device_map
        self.dtype = dtype
        self.attention_implementation = attention_implementation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.question_feature_mode = question_feature_mode

    def encode(
        self,
        *,
        image_path: str | Path,
        question: str,
        bboxes: Sequence[BBox],
    ) -> dict[str, Any]:
        import torch  # type: ignore[import-not-found]
        from PIL import Image

        if not question:
            raise ValueError("question must be non-empty")
        if not bboxes:
            raise ValueError("semantic extraction requires at least one candidate bbox")
        target_device = next(self.model.parameters()).device
        with Image.open(image_path) as loaded:
            image = loaded.convert("RGB")
            image_inputs = self.processor(
                images=[image],
                text=[""],
                return_tensors="pt",
            )
        pixel_values = image_inputs.pixel_values.to(target_device)
        grid_thw = image_inputs.image_grid_thw.to(target_device)
        tokenized = self.processor.tokenizer(
            question,
            add_special_tokens=False,
            return_tensors="pt",
        )
        question_ids = tokenized.input_ids.to(target_device)
        question_attention_mask = tokenized.attention_mask.to(target_device)
        if question_ids.shape[1] == 0:
            raise ValueError("question tokenization produced no tokens")
        with torch.inference_mode():
            vision_outputs = self.model.model.get_image_features(
                pixel_values=pixel_values,
                image_grid_thw=grid_thw,
            )
            image_tokens = vision_outputs.pooler_output
            if not isinstance(image_tokens, (tuple, list)) or len(image_tokens) != 1:
                raise RuntimeError("expected exactly one image embedding sequence")
            spatial_tokens = reshape_merged_visual_tokens(
                image_tokens[0],
                grid_thw[0],
                spatial_merge_size=int(self.model.model.visual.spatial_merge_size),
            )
            bbox_tensor = torch.tensor(
                [bbox.to_list() for bbox in bboxes],
                dtype=torch.float32,
                device=target_device,
            )
            regions = roi_pool_spatial_tokens(
                spatial_tokens.unsqueeze(0),
                bbox_tensor.unsqueeze(0),
            )[0]
            if self.question_feature_mode == "input_mean":
                question_embedding = self.model.model.get_input_embeddings()(
                    question_ids
                ).mean(dim=1)[0]
            else:
                question_outputs = self.model.model.language_model(
                    input_ids=question_ids,
                    attention_mask=question_attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                question_embedding = question_outputs.last_hidden_state.mean(dim=1)[0]
        return {
            "question_embedding": question_embedding.detach().to(torch.float32).cpu(),
            "global_visual_embedding": spatial_tokens.mean(dim=(0, 1))
            .detach()
            .to(torch.float32)
            .cpu(),
            "region_embeddings": regions.detach().to(torch.float32).cpu(),
            "bboxes": bbox_tensor.detach().cpu(),
            "visual_grid_hw": [
                int(spatial_tokens.shape[0]),
                int(spatial_tokens.shape[1]),
            ],
        }

    def encode_multimodal_states(
        self,
        *,
        image_path: str | Path,
        model_prompt: str,
        system_prompt: str,
    ) -> dict[str, Any]:
        """Extract L3 states from one original-image prompt before generation."""

        import torch  # type: ignore[import-not-found]
        from PIL import Image

        if not model_prompt or not system_prompt:
            raise ValueError("model_prompt and system_prompt must be non-empty")
        with Image.open(image_path) as loaded:
            image = loaded.convert("RGB")
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                            "min_pixels": self.min_pixels,
                            "max_pixels": self.max_pixels,
                        },
                        {"type": "text", "text": model_prompt},
                    ],
                },
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        target_device = next(self.model.parameters()).device
        inputs = inputs.to(target_device)
        with torch.inference_mode():
            outputs = self.model(
                **inputs,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
                logits_to_keep=1,
            )
        if not outputs.hidden_states:
            raise RuntimeError("Qwen did not return multimodal hidden states")
        return pool_multimodal_prompt_states(
            outputs.hidden_states[-1],
            inputs.input_ids,
            inputs.attention_mask,
            image_token_id=int(self.model.config.image_token_id),
        )


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def semantic_decision_from_records(
    siblings: Sequence[ActionRecord],
    encoded: Mapping[str, Any],
    *,
    include_outcomes: bool = True,
) -> dict[str, Any]:
    require_torch()
    import torch  # type: ignore[import-not-found]

    answers = [record for record in siblings if record.action_type == "ANSWER"]
    zooms = sorted(
        (record for record in siblings if record.action_type == "ZOOM"),
        key=lambda record: record.action_id,
    )
    if len(answers) != 1 or not zooms:
        raise ValueError("semantic decision requires one ANSWER and at least one ZOOM")
    baseline = answers[0]
    if encoded["region_embeddings"].shape[0] != len(zooms):
        raise ValueError("region embedding count does not match ZOOM actions")
    expected_bboxes = torch.tensor(
        [record.candidate_bbox.to_list() for record in zooms if record.candidate_bbox],
        dtype=torch.float32,
    )
    if not torch.allclose(encoded["bboxes"], expected_bboxes, atol=1e-7, rtol=0.0):
        raise ValueError("encoded bbox order does not match sorted ZOOM actions")
    decision = {
        "state_id": baseline.state_id,
        "image_id": baseline.image_id,
        "source_id": baseline.source_id,
        "replicate_id": baseline.replicate_id,
        "question": baseline.question,
        "original_image": baseline.original_image,
        "action_ids": [record.action_id for record in zooms],
        "question_embedding": encoded["question_embedding"],
        "global_visual_embedding": encoded["global_visual_embedding"],
        "region_embeddings": encoded["region_embeddings"],
        "bboxes": encoded["bboxes"],
        "state_signals": torch.tensor([baseline.entropy_before], dtype=torch.float32),
        "tool_costs": torch.tensor(
            [record.tool_cost for record in zooms], dtype=torch.float32
        ),
        "visual_grid_hw": list(encoded["visual_grid_hw"]),
    }
    if include_outcomes:
        decision.update(
            {
                "success_before": float(baseline.correct_before),
                "success_after": torch.tensor(
                    [record.correct_after for record in zooms], dtype=torch.float32
                ),
            }
        )
    return decision


def _atomic_torch_save(value: object, destination: Path) -> None:
    import torch  # type: ignore[import-not-found]

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(destination)


def load_semantic_feature_dataset(path: str | Path) -> dict[str, Any]:
    require_torch()
    import torch  # type: ignore[import-not-found]

    value = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError("semantic feature dataset must be a mapping")
    if value.get("format_version") != SEMANTIC_FEATURE_FORMAT_VERSION:
        raise ValueError("unsupported semantic feature dataset format")
    if not isinstance(value.get("metadata"), dict) or not isinstance(
        value.get("decisions"), list
    ):
        raise ValueError("semantic feature dataset is missing metadata or decisions")
    return value


def validate_semantic_feature_dataset(
    value: Mapping[str, Any],
    records: Sequence[ActionRecord],
    *,
    allow_partial: bool = False,
    require_outcomes: bool | None = None,
) -> None:
    require_torch()
    import torch  # type: ignore[import-not-found]

    grouped = group_by_decision(records)
    decisions = value.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("semantic feature decisions must be a list")
    metadata = value.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("semantic feature metadata must be a mapping")
    if require_outcomes is None:
        require_outcomes = bool(metadata.get("outcomes_included", True))
    seen: set[tuple[str, str]] = set()
    for decision in decisions:
        key = (str(decision["state_id"]), str(decision["replicate_id"]))
        if key in seen:
            raise ValueError(f"duplicate semantic decision {key!r}")
        seen.add(key)
        if key not in grouped:
            raise ValueError(f"semantic decision {key!r} is absent from rollouts")
        zooms = sorted(
            (record for record in grouped[key] if record.action_type == "ZOOM"),
            key=lambda record: record.action_id,
        )
        baseline = next(
            record for record in grouped[key] if record.action_type == "ANSWER"
        )
        for name, expected in (
            ("source_id", baseline.source_id),
            ("image_id", baseline.image_id),
            ("question", baseline.question),
        ):
            if decision.get(name) != expected:
                raise ValueError(f"semantic {name} differs for decision {key!r}")
        if list(decision["action_ids"]) != [record.action_id for record in zooms]:
            raise ValueError(f"semantic action IDs differ for decision {key!r}")
        expected_costs = torch.tensor(
            [record.tool_cost for record in zooms], dtype=torch.float32
        )
        stored_costs = decision.get("tool_costs")
        if not isinstance(stored_costs, torch.Tensor) or not torch.equal(
            stored_costs, expected_costs
        ):
            raise ValueError(f"semantic tool costs differ for decision {key!r}")
        bbox_values: list[list[float]] = []
        for record in zooms:
            if record.candidate_bbox is None:
                raise ValueError(
                    f"semantic candidate bbox is missing for decision {key!r}"
                )
            bbox_values.append(record.candidate_bbox.to_list())
        expected_bboxes = torch.tensor(
            bbox_values,
            dtype=torch.float32,
        )
        stored_bboxes = decision.get("bboxes")
        if not isinstance(stored_bboxes, torch.Tensor) or not torch.equal(
            stored_bboxes, expected_bboxes
        ):
            raise ValueError(f"semantic bounding boxes differ for decision {key!r}")
        expected_state_signals = torch.tensor(
            [baseline.entropy_before], dtype=torch.float32
        )
        stored_state_signals = decision.get("state_signals")
        if not isinstance(stored_state_signals, torch.Tensor) or not torch.equal(
            stored_state_signals, expected_state_signals
        ):
            raise ValueError(f"semantic state signals differ for decision {key!r}")
        outcome_fields = SEMANTIC_OUTCOME_FIELDS & set(decision)
        if require_outcomes:
            if "success_before" not in decision or "success_after" not in decision:
                raise ValueError(f"semantic labels are missing for decision {key!r}")
            expected_after = torch.tensor(
                [record.correct_after for record in zooms], dtype=torch.float32
            )
            if float(decision["success_before"]) != float(
                baseline.correct_before
            ) or not torch.equal(decision["success_after"], expected_after):
                raise ValueError(f"semantic labels differ for decision {key!r}")
        elif outcome_fields:
            raise ValueError(
                f"outcome-free features contain labels for decision {key!r}: "
                f"{sorted(outcome_fields)}"
            )
    if not allow_partial and seen != set(grouped):
        missing = sorted(set(grouped) - seen)
        raise ValueError(
            f"semantic features are missing rollout decisions: {missing[:5]}"
        )


def initialize_semantic_feature_checkpoint(
    *,
    source_feature_path: str | Path,
    target_rollouts_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Rebase a validated partial feature set onto a larger rollout superset."""

    source_path = Path(source_feature_path).resolve()
    target_path = Path(target_rollouts_path).resolve()
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
    source = load_semantic_feature_dataset(source_path)
    target_records = read_jsonl(target_path)
    validate_semantic_feature_dataset(source, target_records, allow_partial=True)
    source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    target_digest = hashlib.sha256(target_path.read_bytes()).hexdigest()
    metadata = dict(source["metadata"])
    previous_rollouts = metadata.get("source_rollouts")
    previous_rollouts_sha256 = metadata.get("source_rollouts_sha256")
    metadata.update(
        {
            "source_rollouts": str(target_path),
            "source_rollouts_sha256": target_digest,
            "checkpoint_initialization": {
                "source_features": str(source_path),
                "source_features_sha256": source_digest,
                "source_rollouts": previous_rollouts,
                "source_rollouts_sha256": previous_rollouts_sha256,
                "initialized_decisions": len(source["decisions"]),
                "target_decisions": len(group_by_decision(target_records)),
                "code_revision": os.environ.get("BE_CODE_REVISION"),
            },
        }
    )
    result = {
        "format_version": SEMANTIC_FEATURE_FORMAT_VERSION,
        "metadata": metadata,
        "decisions": list(source["decisions"]),
    }
    _atomic_torch_save(result, destination)
    return result


def extract_qwen_semantic_dataset(
    *,
    rollouts_path: str | Path,
    output_path: str | Path,
    model_name_or_path: str,
    revision: str,
    device_map: str = "cuda:0",
    dtype: str = "bfloat16",
    attention_implementation: str = "sdpa",
    min_pixels: int = 256 * 28 * 28,
    max_pixels: int = 768 * 28 * 28,
    local_files_only: bool = True,
    question_feature_mode: str = "input_mean",
    include_outcomes: bool = True,
    checkpoint_interval: int = 1,
    resume: bool = False,
) -> dict[str, Any]:
    """Checkpoint frozen Qwen semantic features for every rollout decision."""

    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    records_path = Path(rollouts_path).resolve()
    destination = Path(output_path)
    records = read_jsonl(records_path)
    grouped = group_by_decision(records)
    rollouts_sha256 = hashlib.sha256(records_path.read_bytes()).hexdigest()
    metadata = {
        "scientific_status": "diagnostic; not a benchmark claim",
        "source_rollouts": str(records_path),
        "source_rollouts_sha256": rollouts_sha256,
        "model": model_name_or_path,
        "model_revision": revision,
        "device_map": device_map,
        "dtype": dtype,
        "attention_implementation": attention_implementation,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "local_files_only": local_files_only,
        "code_revision": os.environ.get("BE_CODE_REVISION"),
        "question_feature_mode": question_feature_mode,
        "outcomes_included": include_outcomes,
        "checkpoint_interval": checkpoint_interval,
        "question_feature": (
            "mean frozen Qwen input-token embedding"
            if question_feature_mode == "input_mean"
            else "mean final hidden state from frozen Qwen text-only contextualization"
        ),
        "visual_feature": "raster-restored Qwen vision merger output",
        "region_feature": "mean ROI pool from the single full-image token grid",
        "packages": {
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "Pillow": _package_version("Pillow"),
        },
    }
    decisions: list[dict[str, Any]] = []
    if destination.exists():
        if not resume:
            raise FileExistsError(
                f"output already exists: {destination}; pass resume=True to continue"
            )
        existing = load_semantic_feature_dataset(destination)
        existing_metadata = existing["metadata"]
        for name in (
            "source_rollouts_sha256",
            "model",
            "model_revision",
            "dtype",
            "attention_implementation",
            "question_feature_mode",
            "min_pixels",
            "max_pixels",
            "outcomes_included",
            "checkpoint_interval",
        ):
            if existing_metadata.get(name) != metadata[name]:
                raise ValueError(f"resume metadata mismatch for {name}")
        metadata = dict(existing_metadata)
        decisions = list(existing["decisions"])
    completed = {
        (str(decision["state_id"]), str(decision["replicate_id"]))
        for decision in decisions
    }
    unexpected = completed - set(grouped)
    if unexpected:
        raise ValueError(
            f"checkpoint has unexpected decisions: {sorted(unexpected)[:5]}"
        )
    pending = [(key, grouped[key]) for key in sorted(grouped) if key not in completed]
    if pending:
        current_revision = os.environ.get("BE_CODE_REVISION")
        raw_revisions = metadata.get("extraction_code_revisions", [])
        if not isinstance(raw_revisions, list):
            raise ValueError("extraction_code_revisions metadata must be a list")
        extraction_revisions = [str(value) for value in raw_revisions]
        original_revision = metadata.get("code_revision")
        if original_revision:
            original_revision_text = str(original_revision)
            if original_revision_text not in extraction_revisions:
                extraction_revisions.append(original_revision_text)
        if current_revision and current_revision not in extraction_revisions:
            extraction_revisions.append(current_revision)
        metadata["extraction_code_revisions"] = extraction_revisions
        extractor = Qwen25VLSemanticExtractor(
            model_name_or_path,
            revision=revision,
            device_map=device_map,
            dtype=dtype,
            attention_implementation=attention_implementation,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            local_files_only=local_files_only,
            question_feature_mode=question_feature_mode,
        )
        state_cache: dict[str, dict[str, Any]] = {}
        for position, (key, siblings) in enumerate(pending, start=1):
            zooms = sorted(
                (record for record in siblings if record.action_type == "ZOOM"),
                key=lambda record: record.action_id,
            )
            exemplar = siblings[0]
            if exemplar.state_id not in state_cache:
                state_cache[exemplar.state_id] = extractor.encode(
                    image_path=exemplar.original_image,
                    question=exemplar.question,
                    bboxes=[
                        record.candidate_bbox
                        for record in zooms
                        if record.candidate_bbox
                    ],
                )
            decisions.append(
                semantic_decision_from_records(
                    siblings,
                    state_cache[exemplar.state_id],
                    include_outcomes=include_outcomes,
                )
            )
            checkpoint_due = position % checkpoint_interval == 0 or position == len(
                pending
            )
            if checkpoint_due:
                payload = {
                    "format_version": SEMANTIC_FEATURE_FORMAT_VERSION,
                    "metadata": metadata,
                    "decisions": decisions,
                }
                _atomic_torch_save(payload, destination)
            print(
                json.dumps(
                    {
                        "checkpoint": str(destination),
                        "checkpoint_interval": checkpoint_interval,
                        "checkpoint_written": checkpoint_due,
                        "completed_this_run": position,
                        "pending_this_run": len(pending) - position,
                        "total_completed": len(completed) + position,
                        "total_decisions": len(grouped),
                        "decision": key,
                    }
                ),
                flush=True,
            )
    result = {
        "format_version": SEMANTIC_FEATURE_FORMAT_VERSION,
        "metadata": metadata,
        "decisions": decisions,
    }
    validate_semantic_feature_dataset(result, records)
    _atomic_torch_save(result, destination)
    return result
