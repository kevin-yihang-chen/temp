from __future__ import annotations

import math
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from .image_ops import normalized_crop_resized_to_source
from .rollout import AgentState, ModelOutput, VisualObservation


_PEAK_MEMORY_FIELDS = ("peak_allocated_bytes", "peak_reserved_bytes")


def generated_token_statistics(
    step_logits: Sequence[Any], generated_ids: Any
) -> tuple[list[float], list[float]]:
    """Return normalized entropy and selected-token log probability per step."""

    import torch  # type: ignore[import-not-found]

    if (
        getattr(generated_ids, "ndim", None) != 2
        or generated_ids.shape[0] != 1
        or generated_ids.shape[1] != len(step_logits)
        or not step_logits
    ):
        raise ValueError("generated IDs and step logits must be one aligned sequence")
    entropies: list[float] = []
    token_log_probabilities: list[float] = []
    for step_index, logits in enumerate(step_logits):
        if getattr(logits, "ndim", None) != 2 or logits.shape[0] != 1:
            raise ValueError("generation step logits must have shape [1, vocabulary]")
        distribution_logits = logits[0].to(torch.float32)
        if distribution_logits.shape[0] < 2:
            raise ValueError("generation vocabulary must contain at least two tokens")
        log_probabilities = torch.log_softmax(distribution_logits, dim=-1)
        probabilities = torch.exp(log_probabilities)
        entropy = -(probabilities * log_probabilities).sum()
        normalized = entropy / math.log(distribution_logits.shape[-1])
        token_id = int(generated_ids[0, step_index].item())
        if token_id < 0 or token_id >= distribution_logits.shape[-1]:
            raise ValueError("generated token ID is outside the vocabulary")
        entropies.append(float(normalized.item()))
        token_log_probabilities.append(float(log_probabilities[token_id].item()))
    if not all(
        math.isfinite(value) for value in (*entropies, *token_log_probabilities)
    ):
        raise RuntimeError("generation statistics contain non-finite values")
    return entropies, token_log_probabilities


def generated_token_confidence_statistics(step_logits: Sequence[Any]) -> dict[str, Any]:
    """Return cost-free confidence summaries from baseline generation logits."""

    import torch  # type: ignore[import-not-found]

    if not step_logits:
        raise ValueError("confidence statistics require at least one generation step")
    maximum_probabilities: list[float] = []
    top1_top2_margins: list[float] = []
    for logits in step_logits:
        if getattr(logits, "ndim", None) != 2 or logits.shape[0] != 1:
            raise ValueError("generation step logits must have shape [1, vocabulary]")
        if logits.shape[1] < 2:
            raise ValueError("generation vocabulary must contain at least two tokens")
        probabilities = torch.softmax(logits[0].to(torch.float32), dim=-1)
        top_two = torch.topk(probabilities, k=2).values
        maximum_probabilities.append(float(top_two[0].item()))
        top1_top2_margins.append(float((top_two[0] - top_two[1]).item()))
    values = (*maximum_probabilities, *top1_top2_margins)
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise RuntimeError("generation confidence statistics are invalid")
    return {
        "maximum_token_probabilities": maximum_probabilities,
        "top1_top2_token_probability_margins": top1_top2_margins,
        "mean_maximum_token_probability": sum(maximum_probabilities)
        / len(maximum_probabilities),
        "mean_top1_top2_token_probability_margin": sum(top1_top2_margins)
        / len(top1_top2_margins),
    }


def merge_runtime_measurements(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any],
) -> dict[str, Any]:
    """Preserve immutable runtime metadata and the largest observed CUDA peaks."""

    merged = dict(current)
    if previous is None:
        return merged
    previous_keys = set(previous) - set(_PEAK_MEMORY_FIELDS)
    current_keys = set(current) - set(_PEAK_MEMORY_FIELDS)
    if previous_keys != current_keys or any(
        previous[key] != current[key] for key in previous_keys
    ):
        raise ValueError("Qwen runtime measurement configuration changed on resume")
    for field in _PEAK_MEMORY_FIELDS:
        old_value = int(previous.get(field, 0))
        new_value = int(current.get(field, 0))
        if old_value < 0 or new_value < 0:
            raise ValueError("Qwen runtime peak memory must be non-negative")
        merged[field] = max(old_value, new_value)
    return merged


class Qwen25VLBackend:
    """Frozen Qwen2.5-VL inference with normalized generated-token entropy."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        revision: str = "main",
        device_map: str = "cuda:0",
        dtype: str = "bfloat16",
        attention_implementation: str = "sdpa",
        max_new_tokens: int = 16,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 768 * 28 * 28,
        local_files_only: bool = True,
        system_prompt: str = "You are a helpful assistant.",
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if min_pixels <= 0 or max_pixels < min_pixels:
            raise ValueError("pixel limits must be positive and ordered")
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import (  # type: ignore[import-not-found]
                AutoProcessor,
                Qwen2_5_VLForConditionalGeneration,
            )
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError(
                "Qwen25VLBackend requires torch, transformers, Pillow, and qwen-vl-utils"
            ) from exc

        if not hasattr(torch, dtype):
            raise ValueError(f"unsupported torch dtype: {dtype}")
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
        self.max_new_tokens = max_new_tokens
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.system_prompt = system_prompt
        self.local_files_only = local_files_only

    def measurement_config(self) -> dict[str, Any]:
        """Return the frozen numerical and accelerator configuration."""

        import torch  # type: ignore[import-not-found]
        import transformers  # type: ignore[import-not-found]

        parameter = next(self.model.parameters())
        device = parameter.device
        accelerator_name: str | None = None
        compute_capability: list[int] | None = None
        if device.type == "cuda":
            accelerator_name = str(torch.cuda.get_device_name(device))
            capability = torch.cuda.get_device_capability(device)
            compute_capability = [int(capability[0]), int(capability[1])]
        return {
            "device_map": self.device_map,
            "device_type": device.type,
            "accelerator_name": accelerator_name,
            "compute_capability": compute_capability,
            "requested_dtype": self.dtype,
            "parameter_dtype": str(parameter.dtype),
            "attention_implementation": self.attention_implementation,
            "actual_attention_implementation": str(
                getattr(self.model.config, "_attn_implementation", "unknown")
            ),
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "system_prompt": self.system_prompt,
            "torch_version": str(torch.__version__),
            "cuda_runtime_version": (
                None if torch.version.cuda is None else str(torch.version.cuda)
            ),
            "transformers_version": str(transformers.__version__),
            "local_files_only": self.local_files_only,
        }

    def runtime_measurement(self) -> dict[str, Any]:
        """Return the numerical contract plus process-local CUDA peak memory."""

        import torch  # type: ignore[import-not-found]

        result = self.measurement_config()
        device = next(self.model.parameters()).device
        if device.type == "cuda":
            result["peak_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated(device)
            )
            result["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
        else:
            result["peak_allocated_bytes"] = 0
            result["peak_reserved_bytes"] = 0
        return result

    @staticmethod
    def _crop_pixels(image: Any, observation: VisualObservation) -> Any:
        if observation.bbox is None:
            return image.copy()
        return normalized_crop_resized_to_source(image, observation.bbox)

    def _messages(
        self,
        state: AgentState,
        observations: Sequence[VisualObservation],
    ) -> list[dict[str, Any]]:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("Qwen25VLBackend requires Pillow") from exc

        content: list[dict[str, Any]] = []
        opened: dict[str, Any] = {}
        try:
            for observation in observations:
                image_path = str(Path(observation.image_path).resolve())
                if image_path not in opened:
                    with Image.open(image_path) as loaded:
                        opened[image_path] = loaded.convert("RGB")
                visual = self._crop_pixels(opened[image_path], observation)
                content.append(
                    {
                        "type": "image",
                        "image": visual,
                        "min_pixels": self.min_pixels,
                        "max_pixels": self.max_pixels,
                    }
                )
            content.append({"type": "text", "text": state.backend_prompt})
            return [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content},
            ]
        finally:
            for image in opened.values():
                image.close()

    def infer(
        self,
        *,
        state: AgentState,
        observations: Sequence[VisualObservation],
        generation_seed: int | None,
    ) -> ModelOutput:
        import torch  # type: ignore[import-not-found]
        from qwen_vl_utils import process_vision_info  # type: ignore[import-not-found]

        if not observations or observations[0].kind != "ORIGINAL":
            raise ValueError("Qwen backend requires ORIGINAL as the first observation")
        if generation_seed is not None:
            torch.manual_seed(generation_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(generation_seed)

        messages = self._messages(state, observations)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        target_device = next(self.model.parameters()).device
        inputs = inputs.to(target_device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                temperature=None,
                top_p=None,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                return_dict_in_generate=True,
                output_logits=True,
            )
        prompt_length = inputs.input_ids.shape[1]
        generated_ids = generated.sequences[:, prompt_length:]
        answer = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        step_logits = getattr(generated, "logits", None)
        if step_logits is None:
            raise RuntimeError(
                "transformers did not return raw generation logits; "
                "install the pinned Qwen optional dependencies"
            )
        token_entropies, token_log_probabilities = generated_token_statistics(
            step_logits, generated_ids
        )
        confidence_statistics = generated_token_confidence_statistics(step_logits)
        entropy = sum(token_entropies) / len(token_entropies)
        return ModelOutput(
            answer=answer,
            entropy=entropy,
            metadata={
                "model": self.model_name_or_path,
                "model_revision": self.revision,
                "device_map": self.device_map,
                "dtype": self.dtype,
                "attention_implementation": self.attention_implementation,
                "max_new_tokens": self.max_new_tokens,
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
                "system_prompt": self.system_prompt,
                "num_observations": len(observations),
                "generated_tokens": len(token_entropies),
                "normalized_token_entropies": token_entropies,
                "generated_token_log_probabilities": token_log_probabilities,
                "mean_generated_token_log_probability": (
                    sum(token_log_probabilities) / len(token_log_probabilities)
                ),
                **confidence_statistics,
                "input_text_sha256": hashlib.sha256(
                    state.backend_prompt.encode()
                ).hexdigest(),
                "distinct_model_prompt": state.model_prompt is not None,
            },
        )
