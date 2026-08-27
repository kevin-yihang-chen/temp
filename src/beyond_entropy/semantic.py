from __future__ import annotations

import math
from typing import Any

try:
    import torch  # type: ignore[import-not-found]
    from torch import nn  # type: ignore[import-not-found]
except ModuleNotFoundError:  # Core package remains dependency-free.
    torch = None
    nn = None


def require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "semantic gain head requires PyTorch; install with `pip install -e '.[semantic]'`"
        )


def roi_pool_spatial_tokens(visual_tokens: Any, bboxes: Any) -> Any:
    """Average-pool normalized xyxy ROIs from a spatial token grid.

    Args:
        visual_tokens: Tensor shaped ``[batch, height, width, visual_dim]``.
        bboxes: Tensor shaped ``[batch, candidates, 4]`` in normalized xyxy.

    The operation extracts candidate representations from one cached full-image
    encoding. It does not execute candidate crops through the VLM.
    """

    require_torch()
    if visual_tokens.ndim != 4 or bboxes.ndim != 3 or bboxes.shape[-1] != 4:
        raise ValueError(
            "expected visual_tokens [B,H,W,D] and bboxes [B,K,4]"
        )
    batch, height, width, _ = visual_tokens.shape
    if bboxes.shape[0] != batch:
        raise ValueError("visual token and bbox batch dimensions must match")
    pooled_batches = []
    for batch_index in range(batch):
        pooled_candidates = []
        for candidate_index in range(bboxes.shape[1]):
            x1, y1, x2, y2 = [
                float(item)
                for item in bboxes[batch_index, candidate_index].detach().cpu()
            ]
            if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
                raise ValueError("bboxes must be normalized positive-area xyxy boxes")
            left = min(width - 1, int(x1 * width))
            top = min(height - 1, int(y1 * height))
            right = max(left + 1, min(width, math.ceil(x2 * width)))
            bottom = max(top + 1, min(height, math.ceil(y2 * height)))
            pooled_candidates.append(
                visual_tokens[batch_index, top:bottom, left:right].mean(dim=(0, 1))
            )
        pooled_batches.append(torch.stack(pooled_candidates))
    return torch.stack(pooled_batches)


if nn is not None:

    class SemanticGainHead(nn.Module):
        """Predict pre-action success gain from question/image/region semantics.

        The head intentionally predicts ``Delta success`` rather than VOI. The
        policy subtracts ``lambda * visual_cost`` at decision time.
        """

        def __init__(
            self,
            *,
            question_dim: int,
            visual_dim: int,
            state_signal_dim: int = 1,
            hidden_dim: int = 256,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            if min(question_dim, visual_dim, state_signal_dim, hidden_dim) <= 0:
                raise ValueError("all semantic head dimensions must be positive")
            self.question_projection = nn.Linear(question_dim, hidden_dim)
            self.region_projection = nn.Linear(visual_dim, hidden_dim)
            self.global_projection = nn.Linear(visual_dim, hidden_dim)
            fused_dim = hidden_dim * 4 + 4 + state_signal_dim
            self.value = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def forward(
            self,
            *,
            question_embedding: Any,
            global_visual_embedding: Any,
            region_embeddings: Any,
            bboxes: Any,
            state_signals: Any,
        ) -> Any:
            if region_embeddings.ndim != 3:
                raise ValueError("region_embeddings must have shape [B,K,D]")
            batch, candidates, _ = region_embeddings.shape
            if question_embedding.shape[0] != batch or global_visual_embedding.shape[0] != batch:
                raise ValueError("semantic input batch dimensions must match")
            if bboxes.shape != (batch, candidates, 4):
                raise ValueError("bboxes must have shape [B,K,4]")
            if state_signals.ndim != 2 or state_signals.shape[0] != batch:
                raise ValueError("state_signals must have shape [B,S]")
            question = self.question_projection(question_embedding)
            region = self.region_projection(region_embeddings)
            global_visual = self.global_projection(global_visual_embedding)
            question = question[:, None, :].expand(-1, candidates, -1)
            global_visual = global_visual[:, None, :].expand(-1, candidates, -1)
            signals = state_signals[:, None, :].expand(-1, candidates, -1)
            fused = torch.cat(
                (
                    question,
                    region,
                    question * region,
                    global_visual,
                    bboxes,
                    signals,
                ),
                dim=-1,
            )
            return self.value(fused).squeeze(-1)


    class CounterfactualSuccessHead(nn.Module):
        """Estimate success before and after each candidate from pre-action inputs."""

        def __init__(
            self,
            *,
            question_dim: int,
            visual_dim: int,
            state_signal_dim: int = 1,
            hidden_dim: int = 256,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            if min(question_dim, visual_dim, state_signal_dim, hidden_dim) <= 0:
                raise ValueError("all semantic head dimensions must be positive")
            self.question_projection = nn.Linear(question_dim, hidden_dim)
            self.region_projection = nn.Linear(visual_dim, hidden_dim)
            self.global_projection = nn.Linear(visual_dim, hidden_dim)
            self.baseline = nn.Sequential(
                nn.Linear(hidden_dim * 3 + state_signal_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            self.action = nn.Sequential(
                nn.Linear(hidden_dim * 4 + 4 + state_signal_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def forward(
            self,
            *,
            question_embedding: Any,
            global_visual_embedding: Any,
            region_embeddings: Any,
            bboxes: Any,
            state_signals: Any,
        ) -> tuple[Any, Any]:
            if region_embeddings.ndim != 3:
                raise ValueError("region_embeddings must have shape [B,K,D]")
            batch, candidates, _ = region_embeddings.shape
            if question_embedding.shape[0] != batch or global_visual_embedding.shape[0] != batch:
                raise ValueError("semantic input batch dimensions must match")
            if bboxes.shape != (batch, candidates, 4):
                raise ValueError("bboxes must have shape [B,K,4]")
            if state_signals.ndim != 2 or state_signals.shape[0] != batch:
                raise ValueError("state_signals must have shape [B,S]")
            question = self.question_projection(question_embedding)
            global_visual = self.global_projection(global_visual_embedding)
            baseline_logits = self.baseline(
                torch.cat(
                    (question, global_visual, question * global_visual, state_signals),
                    dim=-1,
                )
            ).squeeze(-1)
            region = self.region_projection(region_embeddings)
            expanded_question = question[:, None, :].expand(-1, candidates, -1)
            expanded_global = global_visual[:, None, :].expand(-1, candidates, -1)
            expanded_signals = state_signals[:, None, :].expand(-1, candidates, -1)
            action_logits = self.action(
                torch.cat(
                    (
                        expanded_question,
                        region,
                        expanded_question * region,
                        expanded_global,
                        bboxes,
                        expanded_signals,
                    ),
                    dim=-1,
                )
            ).squeeze(-1)
            return baseline_logits, action_logits


    def success_gain_mse(predicted_gain: Any, success_after: Any, success_before: Any) -> Any:
        target_gain = success_after - success_before[:, None]
        return torch.nn.functional.mse_loss(predicted_gain, target_gain)

else:

    class SemanticGainHead:  # type: ignore[no-redef]
        def __init__(self, **_: Any) -> None:
            require_torch()


    class CounterfactualSuccessHead:  # type: ignore[no-redef]
        def __init__(self, **_: Any) -> None:
            require_torch()


    def success_gain_mse(predicted_gain: Any, success_after: Any, success_before: Any) -> Any:
        require_torch()
