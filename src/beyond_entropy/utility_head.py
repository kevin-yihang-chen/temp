"""Trainable ROI utility head and cost-independent supervised objectives."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .semantic import SemanticGainHead, roi_pool_spatial_tokens


class SpatialUtilityHead(nn.Module):
    def __init__(self, hidden_size: int, *, head_dim: int = 128, temperature: float = .25) -> None:
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.temperature = temperature
        self.answer_embedding = nn.Parameter(torch.zeros(hidden_size))
        self.scorer = SemanticGainHead(
            question_dim=hidden_size, visual_dim=hidden_size,
            state_signal_dim=1, hidden_dim=head_dim, dropout=0.,
        )

    def forward(self, question: Any, visual_grid: Any, zoom_bboxes: Any) -> dict[str, Any]:
        # Inputs are original-image features only. ROI pooling remains differentiable.
        regions = roi_pool_spatial_tokens(visual_grid, zoom_bboxes)
        global_visual = visual_grid.mean(dim=(1, 2))
        answer = global_visual + self.answer_embedding
        regions = torch.cat((answer[:, None, :], regions), dim=1)
        b, k, _ = zoom_bboxes.shape
        full = zoom_bboxes.new_tensor([0., 0., 1., 1.]).reshape(1, 1, 4).expand(b, 1, 4)
        boxes = torch.cat((full, zoom_bboxes), dim=1)
        raw = self.scorer(
            question_embedding=question.float(), global_visual_embedding=global_visual.float(),
            region_embeddings=regions.float(), bboxes=boxes.float(),
            # Constant signal, not entropy or a generated answer.
            state_signals=question.new_zeros((b, 1), dtype=torch.float32),
        )
        gains = raw - raw[:, :1]
        return {"action_logits": gains / self.temperature, "predicted_gain": gains}


def utility_sft_loss(
    action_logits: Any, *, method: str, gains: Any = None,
    support_labels: Any = None, temperature: float = .25,
) -> Any:
    if action_logits.ndim != 2 or not torch.isfinite(action_logits).all():
        raise ValueError("finite [batch, actions] logits required")
    if method == "format":
        if gains is not None:
            raise ValueError("Format/Support SFT must not receive utility labels")
        if support_labels is None:
            raise ValueError("gain-free support labels required")
        return F.cross_entropy(action_logits.float(), support_labels)
    if method not in ("best_action", "utility", "pairwise"):
        raise ValueError("unknown supervised objective")
    if gains is None or gains.shape != action_logits.shape or not torch.isfinite(gains).all():
        raise ValueError("finite aligned gains required")
    if not torch.allclose(gains[:, 0], torch.zeros_like(gains[:, 0])):
        raise ValueError("ANSWER gain must be zero")
    if (gains.abs() > 1 + 1e-6).any():
        raise ValueError("raw reward differences must lie in [-1,1]")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if method == "best_action":
        return F.cross_entropy(action_logits.float(), gains.argmax(dim=-1))
    if method == "utility":
        target = F.softmax(gains.detach().float() / temperature, dim=-1)
        return F.kl_div(F.log_softmax(action_logits.float(), dim=-1), target, reduction="batchmean")
    # Optional ablation only; no extra objective search in the MVP.
    predicted_gain = action_logits * temperature
    differences = predicted_gain[:, :, None] - predicted_gain[:, None, :]
    preferred = gains[:, :, None] > gains[:, None, :]
    return F.softplus(-differences[preferred]).mean() if preferred.any() else action_logits.sum() * 0
