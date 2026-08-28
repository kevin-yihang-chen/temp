from __future__ import annotations

import math
import hashlib
from pathlib import Path
from typing import Any, Sequence

from .rollout import AgentState, ModelOutput, VisualObservation


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

    @staticmethod
    def _crop_pixels(image: Any, observation: VisualObservation) -> Any:
        if observation.bbox is None:
            return image.copy()
        width, height = image.size
        bbox = observation.bbox
        pixel_box = (
            round(bbox.x1 * width),
            round(bbox.y1 * height),
            round(bbox.x2 * width),
            round(bbox.y2 * height),
        )
        crop = image.crop(pixel_box)
        resampling = getattr(getattr(type(image), "Resampling", None), "LANCZOS", None)
        if resampling is None:
            from PIL import Image

            resampling = Image.Resampling.LANCZOS
        return crop.resize((width, height), resampling)

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
        token_entropies: list[float] = []
        for logits in step_logits:
            distribution_logits = logits[0].to(torch.float32)
            log_probabilities = torch.log_softmax(distribution_logits, dim=-1)
            probabilities = torch.exp(log_probabilities)
            entropy = -(probabilities * log_probabilities).sum()
            normalized = entropy / math.log(distribution_logits.shape[-1])
            token_entropies.append(float(normalized.item()))
        if not token_entropies:
            raise RuntimeError("Qwen generation returned no token distributions")
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
                "input_text_sha256": hashlib.sha256(
                    state.backend_prompt.encode()
                ).hexdigest(),
                "distinct_model_prompt": state.model_prompt is not None,
            },
        )
