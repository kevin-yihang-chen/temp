"""Leakage-safe lightweight post-training for binary visual acquisition.

The deployable input contains the original image, observations already acquired,
the question prompt, and geometry of the fixed proposed action.  Counterfactual
outcomes live only on :class:`SequentialTrainingExample` and are never accepted
by the model forward method.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .image_ops import normalized_crop_resized_to_source
from .qwen_semantic import build_multimodal_observation_prompt_messages
from .schema import BBox
from .sequential_schema import AcquiredObservationSpec, SequentialRolloutRecord


@dataclass(frozen=True)
class SequentialPolicyInput:
    """Strict pre-action view of one STOP/CONTINUE decision."""

    state_id: str
    image_id: str
    source_id: str
    image_path: str
    question: str
    model_prompt: str
    acquired_observations: tuple[AcquiredObservationSpec, ...]
    proposed_action_id: str
    proposed_bbox: BBox
    proposed_visual_cost: float

    def __post_init__(self) -> None:
        if not all((self.state_id, self.image_id, self.source_id, self.image_path,
                    self.question, self.model_prompt, self.proposed_action_id)):
            raise ValueError("policy input identities and prompts must be non-empty")
        if len(self.acquired_observations) != 1:
            raise ValueError("v1 post-training state requires exactly one acquired observation")
        if not math.isfinite(self.proposed_visual_cost) or self.proposed_visual_cost < 0:
            raise ValueError("proposed visual cost must be finite and non-negative")

    @classmethod
    def from_untrusted_mapping(cls, value: Mapping[str, Any]) -> "SequentialPolicyInput":
        allowed = {
            "state_id", "image_id", "source_id", "image_path", "question",
            "model_prompt", "acquired_observations", "proposed_action_id",
            "proposed_bbox", "proposed_visual_cost",
        }
        if set(value) != allowed:
            raise ValueError(
                "policy input violates strict pre-action allowlist; "
                f"extra={sorted(set(value)-allowed)}, missing={sorted(allowed-set(value))}"
            )
        bbox = BBox.from_value(value["proposed_bbox"])
        if bbox is None:
            raise ValueError("CONTINUE requires a proposed bbox")
        return cls(
            state_id=str(value["state_id"]), image_id=str(value["image_id"]),
            source_id=str(value["source_id"]), image_path=str(value["image_path"]),
            question=str(value["question"]), model_prompt=str(value["model_prompt"]),
            acquired_observations=tuple(
                AcquiredObservationSpec.from_dict(item)
                for item in value["acquired_observations"]
            ),
            proposed_action_id=str(value["proposed_action_id"]), proposed_bbox=bbox,
            proposed_visual_cost=float(value["proposed_visual_cost"]),
        )

    def geometry(self) -> tuple[float, ...]:
        acquired = self.acquired_observations[0].bbox
        return (
            *acquired.to_list(), acquired.width, acquired.height, acquired.area,
            *self.proposed_bbox.to_list(), self.proposed_bbox.width,
            self.proposed_bbox.height, self.proposed_bbox.area,
            self.proposed_visual_cost,
        )


@dataclass(frozen=True)
class SequentialTrainingExample:
    inputs: SequentialPolicyInput
    stop_reward: float
    continue_reward: float
    replicate_id: str

    def __post_init__(self) -> None:
        if not self.replicate_id:
            raise ValueError("replicate_id must be non-empty")
        for name in ("stop_reward", "continue_reward"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")

    @property
    def gain(self) -> float:
        return self.continue_reward - self.stop_reward

    @property
    def decision_id(self) -> tuple[str, str]:
        return self.inputs.state_id, self.replicate_id


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("JSONL rows must be objects")
                rows.append(value)
    if not rows:
        raise ValueError(f"empty JSONL file: {path}")
    return rows


def load_sequential_training_examples(
    rollout_path: str | Path,
    manifest_path: str | Path,
) -> list[SequentialTrainingExample]:
    """Join rollout labels to prompts while constructing a strict input view."""

    manifest_rows = _read_jsonl(manifest_path)
    manifest = {str(row["state_id"]): row for row in manifest_rows}
    if len(manifest) != len(manifest_rows):
        raise ValueError("duplicate manifest state")
    records = [SequentialRolloutRecord.from_dict(row) for row in _read_jsonl(rollout_path)]
    if len({record.decision_id for record in records}) != len(records):
        raise ValueError("duplicate sequential decision")
    examples = []
    for record in records:
        row = manifest.get(record.state_id)
        if row is None:
            raise ValueError(f"rollout state absent from manifest: {record.state_id}")
        expected_image = str(Path(manifest_path).resolve().parent / str(row["image_path"]))
        if Path(record.original_image).resolve() != Path(expected_image).resolve():
            raise ValueError("manifest and rollout image paths differ")
        if (str(row["image_id"]) != record.image_id
                or str(row["source_id"]) != record.source_id
                or str(row["question"]) != record.question):
            raise ValueError("manifest and rollout identities differ")
        raw_input = {
            "state_id": record.state_id, "image_id": record.image_id,
            "source_id": record.source_id, "image_path": record.original_image,
            "question": record.question,
            "model_prompt": str(row.get("model_prompt", record.question)),
            "acquired_observations": [item.to_dict() for item in record.acquired_observations],
            "proposed_action_id": record.proposed_action_id,
            "proposed_bbox": record.proposed_bbox.to_list(),
            "proposed_visual_cost": record.proposed_visual_cost,
        }
        examples.append(SequentialTrainingExample(
            inputs=SequentialPolicyInput.from_untrusted_mapping(raw_input),
            stop_reward=record.stop_correct, continue_reward=record.continue_correct,
            replicate_id=record.replicate_id,
        ))
    return sorted(examples, key=lambda item: item.decision_id)


def state_hash_subset(
    examples: Sequence[SequentialTrainingExample], *, maximum_states: int,
    seed: int, namespace: str,
) -> list[SequentialTrainingExample]:
    """Outcome-independent deterministic state subset for Phase A."""

    if maximum_states <= 0 or not namespace:
        raise ValueError("positive state limit and namespace required")
    ordered = sorted(
        examples,
        key=lambda item: (
            hashlib.sha256(
                f"{namespace}:{seed}:{item.inputs.state_id}:{item.replicate_id}".encode()
            ).hexdigest(),
            item.decision_id,
        ),
    )
    return ordered[:maximum_states]


def deterministic_joint_schedule(
    by_benchmark: Mapping[str, Sequence[SequentialTrainingExample]], *,
    draws: int, seed: int, namespace: str,
) -> list[tuple[str, SequentialTrainingExample]]:
    """Matched outcome-independent domain-balanced cyclic training schedule."""

    if draws <= 0 or not by_benchmark or any(not values for values in by_benchmark.values()):
        raise ValueError("non-empty domains and positive draws required")
    domains = sorted(by_benchmark)
    ordered = {
        domain: sorted(
            values,
            key=lambda item: hashlib.sha256(
                f"{namespace}:{seed}:{domain}:{item.inputs.state_id}:{item.replicate_id}".encode()
            ).hexdigest(),
        )
        for domain, values in by_benchmark.items()
    }
    cursors = {domain: 0 for domain in domains}
    result = []
    for index in range(draws):
        domain = domains[index % len(domains)]
        values = ordered[domain]
        result.append((domain, values[cursors[domain] % len(values)]))
        cursors[domain] += 1
    return result


def sequential_post_training_loss(
    logits: torch.Tensor,
    example: SequentialTrainingExample,
    *,
    method: str,
) -> torch.Tensor:
    """Matched Outcome-only or explicit counterfactual preference objective.

    Outcome-only uses only absolute final branch rewards.  Counterfactual uses
    the signed paired gain to supervise which branch is preferred.  A neutral
    pair contributes no preference gradient to the counterfactual objective.
    """

    if logits.shape != (1, 2):
        raise ValueError("binary policy logits must have shape [1, 2]")
    log_prob = F.log_softmax(logits.float(), dim=-1)[0]
    if method == "outcome_only":
        weights = logits.new_tensor([example.stop_reward, example.continue_reward]).float()
        total = weights.sum()
        return logits.sum() * 0 if total.item() == 0 else -(weights * log_prob).sum() / total
    if method == "counterfactual_utility":
        gain = example.gain
        if gain == 0:
            return logits.sum() * 0
        target = logits.new_tensor([1 if gain > 0 else 0], dtype=torch.long)
        return F.cross_entropy(logits.float(), target)
    raise ValueError(f"unsupported post-training method: {method}")


class QwenSequentialPolicy(nn.Module):
    """Qwen partial-prefix encoder plus a small binary acquisition head."""

    def __init__(self, backbone: Any, processor: Any, *, head_dim: int = 128,
                 min_pixels: int = 256*28*28, max_pixels: int = 768*28*28,
                 train_backbone: bool = True) -> None:
        super().__init__()
        if head_dim <= 0 or min_pixels <= 0 or max_pixels < min_pixels:
            raise ValueError("invalid policy dimensions or pixel budget")
        self.backbone, self.processor = backbone, processor
        self.min_pixels, self.max_pixels = min_pixels, max_pixels
        core = backbone.model
        if not hasattr(core, "visual") or not hasattr(core, "language_model"):
            raise RuntimeError("unsupported Qwen layout")
        self.backbone.requires_grad_(False)
        if train_backbone:
            core.visual.merger.requires_grad_(True)
            core.language_model.layers[-1].requires_grad_(True)
            core.language_model.norm.requires_grad_(True)
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        hidden = int(backbone.config.text_config.hidden_size)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden + 15),
            nn.Linear(hidden + 15, head_dim),
            nn.GELU(),
            nn.Linear(head_dim, 2),
        ).to(next(backbone.parameters()).device)
        for parameter in self.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        self.last_measurement: dict[str, int] = {}

    def forward(self, inputs: SequentialPolicyInput) -> dict[str, torch.Tensor]:
        if not isinstance(inputs, SequentialPolicyInput):
            raise TypeError("policy accepts only typed, outcome-free SequentialPolicyInput")
        from PIL import Image

        with Image.open(inputs.image_path) as opened:
            original = opened.convert("RGB")
            acquired = [
                normalized_crop_resized_to_source(original, item.bbox)
                for item in inputs.acquired_observations
            ]
            images = (original, *acquired)
            messages = build_multimodal_observation_prompt_messages(
                images=images, model_prompt=inputs.model_prompt,
                system_prompt="You are a helpful assistant.",
                min_pixels=self.min_pixels, max_pixels=self.max_pixels,
            )
            tensors = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            )
        device = next(self.backbone.parameters()).device
        tensors = tensors.to(device)
        if int(tensors.image_grid_thw.shape[0]) != len(images):
            raise ValueError("processor image count differs from acquired-prefix state")
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            output = self.backbone.model(**tensors, use_cache=False, return_dict=True)
        attended = tensors.attention_mask[0].nonzero(as_tuple=False).flatten()
        fused = output.last_hidden_state[:, attended[-1], :].float()
        geometry = torch.tensor(
            [inputs.geometry()], device=device, dtype=torch.float32
        )
        logits = self.head(torch.cat((fused, geometry), dim=-1))
        self.last_measurement = {
            "observed_images": len(images),
            "already_acquired_crops": len(acquired),
            "proposed_crop_executions": 0,
            "prompt_tokens": int(tensors.attention_mask.sum().item()),
        }
        return {"action_logits": logits, "continue_score": logits[:, 1] - logits[:, 0]}

    def trainable_state_dict(self) -> dict[str, Any]:
        names = {name for name, value in self.named_parameters() if value.requires_grad}
        return {name: value.detach().cpu() for name, value in self.state_dict().items()
                if name in names}

    def gradient_report(self) -> dict[str, float]:
        prefixes = {
            "head": "head.", "visual_merger": "backbone.model.visual.merger.",
            "language_last": (
                f"backbone.model.language_model.layers."
                f"{len(self.backbone.model.language_model.layers)-1}."
            ),
        }
        return {
            group: sum(
                float(param.grad.detach().float().square().sum())
                for name, param in self.named_parameters()
                if name.startswith(prefix) and param.grad is not None
            ) ** .5
            for group, prefix in prefixes.items()
        }
