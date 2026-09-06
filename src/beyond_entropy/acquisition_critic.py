"""Leakage-safe inputs and small frozen-feature acquisition critics."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - package remains dependency-free
    torch = None
    nn = None


SEQUENTIAL_FEATURE_FORMAT_VERSION = 1
CRITIC_FEATURE_LEVELS = (
    "uncertainty",
    "shallow",
    "semantic",
    "state_semantic",
    "relational",
)
_INPUT_FIELDS = frozenset(
    {
        "stop_entropy",
        "stop_max_probability",
        "stop_top1_top2_margin",
        "shallow_question_features",
        "question_embedding",
        "global_visual_embedding",
        "acquired_region_embedding",
        "proposed_region_embedding",
        "current_pooled_language_state",
        "current_pooled_visual_state",
        "current_fused_multimodal_state",
        "acquired_bbox",
        "proposed_bbox",
        "step_index",
        "acquired_visual_cost",
        "proposed_visual_cost",
    }
)
_FORBIDDEN_FRAGMENTS = (
    "answer",
    "continue_entropy",
    "continue_probability",
    "correct",
    "delta",
    "gain",
    "label",
    "outcome",
    "reward",
    "target",
)


def _finite(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _vector(value: Any, *, name: str) -> tuple[float, ...]:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().reshape(-1).tolist()
    elif hasattr(value, "tolist") and callable(value.tolist):
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a numeric sequence")
    result = tuple(_finite(item, name=name) for item in value)
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


@dataclass(frozen=True)
class AcquisitionInputs:
    """The only object a stopping critic is allowed to consume."""

    state_id: str
    image_id: str
    source_id: str
    stop_entropy: float
    stop_max_probability: float
    stop_top1_top2_margin: float
    shallow_question_features: tuple[float, ...]
    question_embedding: tuple[float, ...]
    global_visual_embedding: tuple[float, ...]
    acquired_region_embedding: tuple[float, ...]
    proposed_region_embedding: tuple[float, ...]
    current_pooled_language_state: tuple[float, ...]
    current_pooled_visual_state: tuple[float, ...]
    current_fused_multimodal_state: tuple[float, ...]
    acquired_bbox: tuple[float, float, float, float]
    proposed_bbox: tuple[float, float, float, float]
    step_index: float
    acquired_visual_cost: float
    proposed_visual_cost: float

    def __post_init__(self) -> None:
        if not self.state_id or not self.image_id or not self.source_id:
            raise ValueError("critic identities must be non-empty")
        scalar_names = (
            "stop_entropy",
            "stop_max_probability",
            "stop_top1_top2_margin",
            "step_index",
            "acquired_visual_cost",
            "proposed_visual_cost",
        )
        for name in scalar_names:
            _finite(getattr(self, name), name=name)
        if self.stop_entropy < 0.0 or self.step_index < 1.0:
            raise ValueError("entropy must be non-negative and step must be positive")
        if not 0 <= self.stop_max_probability <= 1:
            raise ValueError("stop_max_probability must be in [0, 1]")
        if not 0 <= self.stop_top1_top2_margin <= 1:
            raise ValueError("stop_top1_top2_margin must be in [0, 1]")
        if self.acquired_visual_cost < 0 or self.proposed_visual_cost < 0:
            raise ValueError("visual costs must be non-negative")
        vectors = (
            self.question_embedding,
            self.global_visual_embedding,
            self.acquired_region_embedding,
            self.proposed_region_embedding,
            self.current_pooled_language_state,
            self.current_pooled_visual_state,
            self.current_fused_multimodal_state,
        )
        dimensions = {len(item) for item in vectors}
        if len(dimensions) != 1:
            raise ValueError("all frozen VLM vectors must have the same dimension")
        for name in ("acquired_bbox", "proposed_bbox"):
            x1, y1, x2, y2 = getattr(self, name)
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError(f"{name} must be a normalized xyxy box")

    @classmethod
    def from_untrusted_mapping(cls, value: Mapping[str, Any]) -> "AcquisitionInputs":
        raw = value.get("pre_action")
        if not isinstance(raw, Mapping):
            raise ValueError("feature row requires a pre_action mapping")
        if set(raw) != _INPUT_FIELDS:
            raise ValueError(
                "pre_action keys differ from strict allowlist: "
                f"missing={sorted(_INPUT_FIELDS-set(raw))}, "
                f"extra={sorted(set(raw)-_INPUT_FIELDS)}"
            )
        for key in raw:
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in _FORBIDDEN_FRAGMENTS):
                raise ValueError(f"post-action/target feature is forbidden: {key}")
        vector_names = (
            "shallow_question_features",
            "question_embedding",
            "global_visual_embedding",
            "acquired_region_embedding",
            "proposed_region_embedding",
            "current_pooled_language_state",
            "current_pooled_visual_state",
            "current_fused_multimodal_state",
        )
        vectors = {name: _vector(raw[name], name=name) for name in vector_names}
        acquired_bbox = _vector(raw["acquired_bbox"], name="acquired_bbox")
        proposed_bbox = _vector(raw["proposed_bbox"], name="proposed_bbox")
        if len(acquired_bbox) != 4 or len(proposed_bbox) != 4:
            raise ValueError("critic bboxes must each contain four values")
        return cls(
            state_id=str(value["state_id"]),
            image_id=str(value["image_id"]),
            source_id=str(value["source_id"]),
            stop_entropy=_finite(raw["stop_entropy"], name="stop_entropy"),
            stop_max_probability=_finite(
                raw["stop_max_probability"], name="stop_max_probability"
            ),
            stop_top1_top2_margin=_finite(
                raw["stop_top1_top2_margin"], name="stop_top1_top2_margin"
            ),
            acquired_bbox=tuple(acquired_bbox),  # type: ignore[arg-type]
            proposed_bbox=tuple(proposed_bbox),  # type: ignore[arg-type]
            step_index=_finite(raw["step_index"], name="step_index"),
            acquired_visual_cost=_finite(
                raw["acquired_visual_cost"], name="acquired_visual_cost"
            ),
            proposed_visual_cost=_finite(
                raw["proposed_visual_cost"], name="proposed_visual_cost"
            ),
            **vectors,
        )

    def scalar_vector(self) -> tuple[float, ...]:
        ax1, ay1, ax2, ay2 = self.acquired_bbox
        px1, py1, px2, py2 = self.proposed_bbox
        return (
            self.stop_entropy,
            self.stop_max_probability,
            self.stop_top1_top2_margin,
            self.step_index,
            self.acquired_visual_cost,
            self.proposed_visual_cost,
            ax1,
            ay1,
            ax2,
            ay2,
            (ax1 + ax2) / 2,
            (ay1 + ay2) / 2,
            (ax2 - ax1) * (ay2 - ay1),
            px1,
            py1,
            px2,
            py2,
            (px1 + px2) / 2,
            (py1 + py2) / 2,
            (px2 - px1) * (py2 - py1),
        )

    def feature_vector(self, level: str = "state_semantic") -> tuple[float, ...]:
        if level not in CRITIC_FEATURE_LEVELS:
            raise ValueError(f"unsupported critic feature level: {level}")
        uncertainty = (
            self.stop_entropy,
            self.stop_max_probability,
            self.stop_top1_top2_margin,
        )
        if level == "uncertainty":
            return uncertainty
        shallow = self.scalar_vector() + self.shallow_question_features
        if level == "shallow":
            return shallow
        q = self.question_embedding
        proposed = self.proposed_region_embedding
        acquired = self.acquired_region_embedding
        semantic = (
            q
            + self.global_visual_embedding
            + acquired
            + proposed
            + tuple(a * b for a, b in zip(q, proposed))
            + tuple(a * b for a, b in zip(acquired, proposed))
        )
        if level == "semantic":
            return shallow + semantic
        state_semantic = (
            shallow
            + semantic
            + self.current_pooled_language_state
            + self.current_pooled_visual_state
            + self.current_fused_multimodal_state
        )
        if level == "state_semantic":
            return state_semantic

        # The single preregistered representation correction.  It removes the
        # underdetermined raw coordinate concatenation (18k+ dimensions in the
        # pilot) while retaining label-free, action-conditional relationships.
        named = {
            "question": self.question_embedding,
            "global": self.global_visual_embedding,
            "acquired": self.acquired_region_embedding,
            "proposed": self.proposed_region_embedding,
            "language": self.current_pooled_language_state,
            "visual": self.current_pooled_visual_state,
            "fused": self.current_fused_multimodal_state,
        }

        def summary(vector: tuple[float, ...]) -> tuple[float, ...]:
            count = len(vector)
            center = sum(vector) / count
            mean_square = sum(value * value for value in vector) / count
            variance = sum((value - center) ** 2 for value in vector) / count
            return (
                center,
                math.sqrt(mean_square),
                math.sqrt(variance),
                sum(abs(value) for value in vector) / count,
            )

        def relation(
            left: tuple[float, ...], right: tuple[float, ...]
        ) -> tuple[float, ...]:
            count = len(left)
            dot = sum(a * b for a, b in zip(left, right))
            left_norm = math.sqrt(sum(value * value for value in left))
            right_norm = math.sqrt(sum(value * value for value in right))
            cosine = dot / max(left_norm * right_norm, 1e-12)
            return (
                cosine,
                dot / count,
                math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)) / count),
            )

        pairs = (
            ("question", "global"),
            ("question", "acquired"),
            ("question", "proposed"),
            ("global", "acquired"),
            ("global", "proposed"),
            ("acquired", "proposed"),
            ("language", "visual"),
            ("language", "fused"),
            ("visual", "fused"),
            ("question", "fused"),
            ("proposed", "fused"),
        )
        summaries = tuple(value for vector in named.values() for value in summary(vector))
        relations = tuple(
            value
            for left, right in pairs
            for value in relation(named[left], named[right])
        )
        return shallow + summaries + relations


@dataclass(frozen=True)
class AcquisitionExample:
    inputs: AcquisitionInputs
    replicate_id: str
    stop_correct: float
    continue_correct: float

    @property
    def remaining_risk(self) -> float:
        return 1.0 - self.stop_correct

    @property
    def gain(self) -> float:
        return self.continue_correct - self.stop_correct


if nn is not None:

    class AcquisitionCritic(nn.Module):
        """Linear or two-hidden-layer scalar critic over frozen features."""

        def __init__(
            self,
            input_dim: int,
            *,
            architecture: str = "linear",
            hidden_dim: int = 128,
        ) -> None:
            super().__init__()
            if input_dim <= 0 or hidden_dim <= 0:
                raise ValueError("critic dimensions must be positive")
            if architecture == "linear":
                self.network = nn.Linear(input_dim, 1)
            elif architecture == "mlp":
                self.network = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim // 2, 1),
                )
            else:
                raise ValueError("architecture must be linear or mlp")

        def forward(self, features: Any) -> Any:
            if features.ndim != 2:
                raise ValueError("critic features must have shape [batch, dim]")
            return self.network(features.float()).squeeze(-1)

else:

    class AcquisitionCritic:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("AcquisitionCritic requires PyTorch")


def examples_from_feature_dataset(
    value: Mapping[str, Any], *, allow_test: bool = False
) -> list[AcquisitionExample]:
    if value.get("format_version") != SEQUENTIAL_FEATURE_FORMAT_VERSION:
        raise ValueError("unsupported sequential feature format")
    metadata = value.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("feature metadata must be a mapping")
    test_accessed = metadata.get("test_accessed")
    if test_accessed is not False and not (allow_test and test_accessed is True):
        raise ValueError("feature metadata/test authorization mismatch")
    rows = value.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("feature dataset requires non-empty rows")
    examples: list[AcquisitionExample] = []
    identities: set[tuple[str, str]] = set()
    for row in rows:
        inputs = AcquisitionInputs.from_untrusted_mapping(row)
        raw = row.get("labels")
        if not isinstance(raw, Mapping) or set(raw) != {
            "replicate_id",
            "stop_correct",
            "continue_correct",
        }:
            raise ValueError("labels namespace differs from the frozen contract")
        example = AcquisitionExample(
            inputs=inputs,
            replicate_id=str(raw["replicate_id"]),
            stop_correct=_finite(raw["stop_correct"], name="stop_correct"),
            continue_correct=_finite(raw["continue_correct"], name="continue_correct"),
        )
        if not 0 <= example.stop_correct <= 1 or not 0 <= example.continue_correct <= 1:
            raise ValueError("correctness labels must lie in [0, 1]")
        identity = (inputs.state_id, example.replicate_id)
        if identity in identities:
            raise ValueError("duplicate sequential feature decision")
        identities.add(identity)
        examples.append(example)
    for level in CRITIC_FEATURE_LEVELS:
        if len({len(item.inputs.feature_vector(level)) for item in examples}) != 1:
            raise ValueError(f"feature dimension changes within {level}")
    return examples


def audit_development_disjointness(
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
) -> dict[str, int | bool]:
    """Require both source-level and decoded-image disjointness."""

    if not train_rows or not validation_rows:
        raise ValueError("split audit requires non-empty train and validation rows")
    train_sources = {str(row["source_id"]) for row in train_rows}
    validation_sources = {str(row["source_id"]) for row in validation_rows}
    train_rgb = {str(row["image_rgb_sha256"]) for row in train_rows}
    validation_rgb = {str(row["image_rgb_sha256"]) for row in validation_rows}
    source_overlap = train_sources & validation_sources
    rgb_overlap = train_rgb & validation_rgb
    if source_overlap or rgb_overlap:
        raise ValueError(
            "source/RGB leakage between train and validation: "
            f"sources={len(source_overlap)}, rgb={len(rgb_overlap)}"
        )
    return {
        "passed": True,
        "train_sources": len(train_sources),
        "validation_sources": len(validation_sources),
        "source_overlap": 0,
        "rgb_overlap": 0,
    }


def _question_features(question: str, stop_backend: Mapping[str, Any]) -> tuple[float, ...]:
    """Small deployable text/confidence summary; never consumes answer content."""

    tokens = question.split()
    characters = len(question)
    token_entropies = _vector(
        stop_backend["normalized_token_entropies"],
        name="normalized_token_entropies",
    )
    token_log_probs = _vector(
        stop_backend["generated_token_log_probabilities"],
        name="generated_token_log_probabilities",
    )
    if len(token_entropies) != len(token_log_probs):
        raise ValueError("current-state token statistics must align")
    return (
        math.log1p(characters),
        math.log1p(len(tokens)),
        sum(char.isdigit() for char in question) / max(1, characters),
        sum(char.isalpha() for char in question) / max(1, characters),
        math.log1p(len(token_entropies)),
        max(token_entropies),
        min(token_entropies),
        sum(token_entropies) / len(token_entropies),
        sum(token_log_probs) / len(token_log_probs),
    )


def build_sequential_feature_row(
    record: Any,
    *,
    semantic: Mapping[str, Any],
    current_multimodal: Mapping[str, Any],
    image_rgb_sha256: str,
) -> dict[str, Any]:
    """Construct a strict pre-action row from a paired rollout record.

    ``semantic`` must have been computed on the original image with acquired
    boxes followed by the proposed box. ``current_multimodal`` must encode only
    original plus already-acquired observations, never the proposed crop.
    """

    from .sequential_schema import SequentialRolloutRecord

    if not isinstance(record, SequentialRolloutRecord):
        raise TypeError("record must be a SequentialRolloutRecord")
    if len(record.acquired_observations) != 1:
        raise ValueError("the frozen MVP supports exactly one acquired observation")
    if len(image_rgb_sha256) != 64:
        raise ValueError("decoded RGB SHA-256 required")
    stop_count = int(record.stop_backend.get("num_observations", -1))
    continue_count = int(record.continue_backend.get("num_observations", -1))
    if stop_count != 2 or continue_count != 3:
        raise ValueError("backend evidence does not represent shared 2/3-image branches")
    region_embeddings = semantic["region_embeddings"]
    bboxes = semantic["bboxes"]
    region_count = (
        int(region_embeddings.shape[0])
        if getattr(region_embeddings, "shape", None) is not None
        else len(region_embeddings)
    )
    if region_count != 2:
        raise ValueError("semantic extraction must contain acquired and proposed ROIs")
    expected_boxes = (
        record.acquired_observations[0].bbox.to_list(),
        record.proposed_bbox.to_list(),
    )
    actual_boxes = (
        _vector(bboxes[0], name="acquired semantic bbox"),
        _vector(bboxes[1], name="proposed semantic bbox"),
    )
    if any(
        any(abs(a - b) > 1e-6 for a, b in zip(actual, expected))
        for actual, expected in zip(actual_boxes, expected_boxes)
    ):
        raise ValueError("semantic bbox order differs from sequential action order")
    pre_action = {
        "stop_entropy": record.stop_entropy,
        "stop_max_probability": record.stop_max_probability,
        "stop_top1_top2_margin": record.stop_top1_top2_margin,
        "shallow_question_features": _question_features(
            record.question, record.stop_backend
        ),
        "question_embedding": semantic["question_embedding"],
        "global_visual_embedding": semantic["global_visual_embedding"],
        "acquired_region_embedding": region_embeddings[0],
        "proposed_region_embedding": region_embeddings[1],
        "current_pooled_language_state": current_multimodal[
            "pooled_language_state"
        ],
        "current_pooled_visual_state": current_multimodal["pooled_visual_state"],
        "current_fused_multimodal_state": current_multimodal[
            "fused_multimodal_state"
        ],
        "acquired_bbox": expected_boxes[0],
        "proposed_bbox": expected_boxes[1],
        "step_index": float(record.step_index),
        "acquired_visual_cost": record.acquired_visual_cost,
        "proposed_visual_cost": record.proposed_visual_cost,
    }
    row = {
        "state_id": record.state_id,
        "image_id": record.image_id,
        "source_id": record.source_id,
        "image_rgb_sha256": image_rgb_sha256,
        "pre_action": pre_action,
        "labels": {
            "replicate_id": record.replicate_id,
            "stop_correct": record.stop_correct,
            "continue_correct": record.continue_correct,
        },
        "diagnostics": {
            "proposed_action_id": record.proposed_action_id,
            "stop_answer": record.stop_answer,
            "continue_answer": record.continue_answer,
            "continue_entropy": record.continue_entropy,
        },
    }
    # The typed view copies only pre_action and proves the input contract now.
    AcquisitionInputs.from_untrusted_mapping(row)
    return row


def extract_sequential_feature_dataset(
    *,
    records: Sequence[Any],
    manifest_states: Mapping[str, Any],
    output_path: str | Path,
    dataset_role: str,
    benchmark: str,
    model_name_or_path: str,
    revision: str,
    device_map: str = "cuda:0",
    dtype: str = "bfloat16",
    attention_implementation: str = "sdpa",
    min_pixels: int = 256 * 28 * 28,
    max_pixels: int = 768 * 28 * 28,
    checkpoint_interval: int = 16,
    test_accessed: bool = False,
) -> dict[str, Any]:
    """Extract original ROI features and the current-prefix frozen Qwen state."""

    if dataset_role not in {"train", "validation", "test"}:
        raise ValueError("unsupported sequential feature role")
    if (dataset_role == "test") != bool(test_accessed):
        raise ValueError("test role requires a ledger-first access attestation")
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    import torch as torch_runtime

    from .predictability_features import decoded_rgb_sha256
    from .qwen_semantic import Qwen25VLSemanticExtractor, _atomic_torch_save

    destination = Path(output_path).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite feature dataset: {destination}")
    extractor = Qwen25VLSemanticExtractor(
        model_name_or_path,
        revision=revision,
        device_map=device_map,
        dtype=dtype,
        attention_implementation=attention_implementation,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        local_files_only=True,
    )
    rows: list[dict[str, Any]] = []
    cached_state_id: str | None = None
    cached: tuple[dict[str, Any], dict[str, Any], str] | None = None
    metadata = {
        "schema": "sequential_acquisition_feature_metadata_v1",
        "dataset_role": dataset_role,
        "benchmark": benchmark,
        "model": model_name_or_path,
        "model_revision": revision,
        "test_accessed": bool(test_accessed),
        "feature_contract": "strict_pre_action_allowlist_v1",
        "proposed_crop_executed_for_features": False,
        "outcomes_included_only_as_labels": True,
    }
    for position, record in enumerate(records, start=1):
        state = manifest_states.get(record.state_id)
        if state is None:
            raise ValueError(f"record absent from manifest: {record.state_id}")
        if record.state_id != cached_state_id:
            acquired = record.acquired_observations[0]
            semantic = extractor.encode(
                image_path=record.original_image,
                question=record.question,
                bboxes=(acquired.bbox, record.proposed_bbox),
            )
            current = extractor.encode_multimodal_states(
                image_path=record.original_image,
                model_prompt=state.backend_prompt,
                system_prompt=str(record.stop_backend["system_prompt"]),
                crop_bbox=acquired.bbox,
            )
            cached_state_id = record.state_id
            cached = (semantic, current, decoded_rgb_sha256(record.original_image))
        if cached is None:
            raise AssertionError("feature cache was not initialized")
        semantic, current, rgb_hash = cached
        rows.append(
            build_sequential_feature_row(
                record,
                semantic=semantic,
                current_multimodal=current,
                image_rgb_sha256=rgb_hash,
            )
        )
        if position % checkpoint_interval == 0:
            _atomic_torch_save(
                {
                    "format_version": SEQUENTIAL_FEATURE_FORMAT_VERSION,
                    "metadata": metadata,
                    "rows": rows,
                    "incomplete": True,
                },
                destination,
            )
    result = {
        "format_version": SEQUENTIAL_FEATURE_FORMAT_VERSION,
        "metadata": metadata,
        "rows": rows,
        "incomplete": False,
    }
    examples_from_feature_dataset(result, allow_test=test_accessed)
    _atomic_torch_save(result, destination)
    # Hash is returned for experiment ledgers without retaining model tensors.
    result["output_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    if torch_runtime.cuda.is_available():
        result["peak_gpu_bytes"] = int(torch_runtime.cuda.max_memory_allocated())
    return result
