"""Original-image-only, differentiable Qwen2.5-VL utility selector.

The answering backend is separate and remains fixed. No candidate crop is run
here. Adapts the locally pinned Transformers Qwen2.5-VL model/vision output API.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .qwen_semantic import build_multimodal_prompt_messages, reshape_merged_visual_tokens
from .utility_dataset import UtilityInputs
from .utility_head import SpatialUtilityHead


class QwenSpatialUtility(nn.Module):
    def __init__(self, backbone: Any, processor: Any, *, temperature: float = .25,
                 head_dim: int = 128, min_pixels: int = 256*28*28,
                 max_pixels: int = 768*28*28, train_backbone: bool = True) -> None:
        super().__init__()
        self.backbone = backbone
        self.processor = processor
        self.min_pixels, self.max_pixels = min_pixels, max_pixels
        if min_pixels <= 0 or max_pixels < min_pixels:
            raise ValueError("invalid pixel budget")
        core = backbone.model
        if not hasattr(core, "visual") or not hasattr(core, "language_model"):
            raise RuntimeError("unsupported Qwen layout; verify pinned runtime before training")
        self.backbone.requires_grad_(False)
        if train_backbone:
            core.visual.merger.requires_grad_(True)
            core.language_model.layers[-1].requires_grad_(True)
            core.language_model.norm.requires_grad_(True)
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        size = int(backbone.config.text_config.hidden_size)
        device = next(backbone.parameters()).device
        self.head = SpatialUtilityHead(size, head_dim=head_dim, temperature=temperature).to(device)
        # Keep optimizer/master parameters in FP32; frozen backbone remains BF16.
        # Otherwise small SFT updates can round away in BF16 weights.
        for parameter in self.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        self.vision_calls = 0
        self.last_measurement: dict[str, int] = {}

    def forward(self, inputs: UtilityInputs, *, region_ablation: bool = False) -> dict[str, Any]:
        if not isinstance(inputs, UtilityInputs):
            raise TypeError("selector accepts only typed, outcome-free UtilityInputs")
        from PIL import Image

        with Image.open(inputs.state.image_path) as opened:
            original = opened.convert("RGB")
            messages = build_multimodal_prompt_messages(
                image=original, model_prompt=inputs.state.backend_prompt,
                system_prompt="You are a helpful assistant.",
                min_pixels=self.min_pixels, max_pixels=self.max_pixels,
            )
            tensors = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            )
        device = next(self.backbone.parameters()).device
        tensors = tensors.to(device)
        if tensors.image_grid_thw.shape[0] != 1:
            raise ValueError("selector must encode exactly one original image")
        visual_outputs = []

        def capture(_module: Any, _args: Any, output: Any) -> None:
            # Keep graph: never detach or use the frozen semantic extractor here.
            merged = getattr(output, "pooler_output", output)
            if not isinstance(merged, torch.Tensor) or merged.ndim != 2:
                raise RuntimeError("unexpected Qwen merged-vision output contract")
            visual_outputs.append(merged)

        hook = self.backbone.model.visual.register_forward_hook(capture)
        try:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = self.backbone.model(**tensors, use_cache=False, return_dict=True)
        finally:
            hook.remove()
        if len(visual_outputs) != 1:
            raise RuntimeError("expected one and only one original-image vision forward")
        self.vision_calls += 1
        grid = reshape_merged_visual_tokens(
            visual_outputs[0], tensors.image_grid_thw[0],
            spatial_merge_size=int(self.backbone.config.vision_config.spatial_merge_size),
        )
        if region_ablation:
            grid = grid.mean(dim=(0, 1), keepdim=True).expand_as(grid)
        positions = tensors.attention_mask[0].nonzero(as_tuple=False).flatten()
        question = output.last_hidden_state[:, positions[-1], :]
        boxes = torch.tensor([
            a.bbox.to_list() for a in inputs.action_space.actions[1:]
        ], device=device, dtype=torch.float32).unsqueeze(0)
        result = self.head(question, grid.unsqueeze(0), boxes)
        self.last_measurement = {
            "original_image_tokens": int(visual_outputs[0].shape[0]),
            "prompt_tokens": int(tensors.attention_mask.sum().item()),
            "vision_encoder_calls": 1, "candidate_crop_executions": 0,
        }
        return result

    def trainable_state_dict(self) -> dict[str, Any]:
        names = {name for name, p in self.named_parameters() if p.requires_grad}
        return {name: t.detach().cpu() for name, t in self.state_dict().items() if name in names}

    def gradient_report(self) -> dict[str, float]:
        groups = {"head": "head.", "visual_merger": "backbone.model.visual.merger.",
                  "language_last": f"backbone.model.language_model.layers.{len(self.backbone.model.language_model.layers)-1}."}
        return {
            group: sum(float(p.grad.detach().float().square().sum().item()) for name, p in self.named_parameters()
                       if name.startswith(prefix) and p.grad is not None) ** .5
            for group, prefix in groups.items()
        }
