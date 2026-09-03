from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping, Sequence

from .predictability_audit import BinaryToolOutcome, PreActionInputs
from .dataset import group_by_decision, read_jsonl
from .predictability_audit import collapse_fixed_entropy_tool
from .predictability_modeling import AuditExample
from .qwen_semantic import Qwen25VLSemanticExtractor


PREDICTABILITY_FEATURE_FORMAT_VERSION = 1
SHALLOW_FEATURE_NAMES = (
    "log1p_question_characters",
    "log1p_question_tokens",
    "question_digit_fraction",
    "question_alphabetic_fraction",
    "first_token_what",
    "first_token_which",
    "first_token_how",
    "first_token_when",
    "first_token_where",
    "first_token_who",
    "first_token_why",
    "first_token_is_or_are",
    "first_token_other",
    "log1p_baseline_answer_characters",
    "log1p_generated_tokens",
    "maximum_normalized_token_entropy",
    "minimum_normalized_token_entropy",
    "std_normalized_token_entropy",
    "mean_generated_token_log_probability",
)


def decoded_rgb_sha256(image_path: str | Path) -> str:
    from PIL import Image

    with Image.open(image_path) as loaded:
        image = loaded.convert("RGB")
        width, height = image.size
        payload = width.to_bytes(8, "big") + height.to_bytes(8, "big") + image.tobytes()
    return hashlib.sha256(payload).hexdigest()


def shallow_question_state_features(
    *, question: str, baseline_answer: str, baseline_backend: Mapping[str, Any]
) -> tuple[float, ...]:
    if not question:
        raise ValueError("question must be non-empty")
    token_entropies = [
        float(item) for item in baseline_backend["normalized_token_entropies"]
    ]
    token_log_probabilities = [
        float(item) for item in baseline_backend["generated_token_log_probabilities"]
    ]
    if (
        not token_entropies
        or len(token_entropies) != len(token_log_probabilities)
        or not all(
            math.isfinite(item) for item in (*token_entropies, *token_log_probabilities)
        )
    ):
        raise ValueError("baseline token statistics must be finite and aligned")
    characters = len(question)
    digits = sum(character.isdigit() for character in question)
    alphabetic = sum(character.isalpha() for character in question)
    tokens = question.split()
    first = tokens[0].casefold().strip("?.,:;!\"'")
    buckets = ("what", "which", "how", "when", "where", "who", "why")
    indicators = [float(first == bucket) for bucket in buckets]
    indicators.append(float(first in {"is", "are"}))
    indicators.append(float(not any(indicators)))
    result = (
        math.log1p(characters),
        math.log1p(len(tokens)),
        digits / max(1, characters),
        alphabetic / max(1, characters),
        *indicators,
        math.log1p(len(baseline_answer)),
        math.log1p(len(token_entropies)),
        max(token_entropies),
        min(token_entropies),
        pstdev(token_entropies),
        mean(token_log_probabilities),
    )
    if len(result) != len(SHALLOW_FEATURE_NAMES) or not all(
        math.isfinite(item) for item in result
    ):
        raise AssertionError("shallow feature contract produced invalid output")
    return tuple(result)


def _vector(value: Any, *, name: str) -> tuple[float, ...]:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().reshape(-1).tolist()
    elif hasattr(value, "tolist") and callable(value.tolist):
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    result = tuple(float(item) for item in value)
    if not result or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must be non-empty and finite")
    return result


def build_predictability_feature_row(
    *,
    outcome: BinaryToolOutcome,
    image_rgb_sha256: str,
    question: str,
    baseline_answer: str,
    baseline_entropy: float,
    baseline_backend: Mapping[str, Any],
    semantic: Mapping[str, Any],
    multimodal: Mapping[str, Any],
) -> dict[str, Any]:
    if int(baseline_backend.get("num_observations", -1)) != 1:
        raise ValueError(
            "L0/L3 features must come from the original-image baseline only"
        )
    maximum = float(baseline_backend["mean_maximum_token_probability"])
    margin = float(baseline_backend["mean_top1_top2_token_probability_margin"])
    pre_action = {
        "entropy_before": float(baseline_entropy),
        "max_probability": maximum,
        "top1_top2_margin": margin,
        "shallow_question_features": shallow_question_state_features(
            question=question,
            baseline_answer=baseline_answer,
            baseline_backend=baseline_backend,
        ),
        "question_embedding": _vector(
            semantic["question_embedding"], name="question_embedding"
        ),
        "global_visual_embedding": _vector(
            semantic["global_visual_embedding"], name="global_visual_embedding"
        ),
        "pooled_language_state": _vector(
            multimodal["pooled_language_state"], name="pooled_language_state"
        ),
        "pooled_visual_state": _vector(
            multimodal["pooled_visual_state"], name="pooled_visual_state"
        ),
        "fused_multimodal_state": _vector(
            multimodal["fused_multimodal_state"], name="fused_multimodal_state"
        ),
    }
    raw = {
        "state_id": outcome.state_id,
        "image_id": outcome.image_id,
        "source_id": outcome.source_id,
        "replicate_id": outcome.replicate_id,
        "image_rgb_sha256": image_rgb_sha256,
        "pre_action": pre_action,
        "outcome": asdict(outcome),
        "feature_diagnostics": {
            "multimodal_prompt_tokens": int(multimodal["multimodal_prompt_tokens"]),
            "multimodal_image_tokens": int(multimodal["multimodal_image_tokens"]),
            "multimodal_language_tokens": int(multimodal["multimodal_language_tokens"]),
        },
    }
    # Round-trip through the strict view before allowing serialization.
    PreActionInputs.from_untrusted_mapping(raw)
    return raw


def audit_example_from_feature_row(value: Mapping[str, Any]) -> AuditExample:
    inputs = PreActionInputs.from_untrusted_mapping(value)
    raw_outcome = value.get("outcome")
    if not isinstance(raw_outcome, Mapping):
        raise ValueError("feature row is missing outcome mapping")
    outcome = BinaryToolOutcome(
        state_id=str(raw_outcome["state_id"]),
        replicate_id=str(raw_outcome["replicate_id"]),
        image_id=str(raw_outcome["image_id"]),
        source_id=str(raw_outcome["source_id"]),
        selected_action_id=str(raw_outcome["selected_action_id"]),
        y0=float(raw_outcome["y0"]),
        y_tool=float(raw_outcome["y_tool"]),
        tool_cost=float(raw_outcome["tool_cost"]),
        tool_calls=int(raw_outcome["tool_calls"]),
    )
    return AuditExample(
        inputs=inputs,
        outcome=outcome,
        image_rgb_sha256=str(value["image_rgb_sha256"]),
    )


def validate_predictability_feature_dataset(
    value: Mapping[str, Any]
) -> list[AuditExample]:
    if value.get("format_version") != PREDICTABILITY_FEATURE_FORMAT_VERSION:
        raise ValueError("unsupported predictability feature format")
    if not isinstance(value.get("metadata"), Mapping):
        raise ValueError("feature dataset metadata must be a mapping")
    raw_rows = value.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("feature dataset rows must be a non-empty list")
    examples = [audit_example_from_feature_row(row) for row in raw_rows]
    identities = [
        (item.outcome.state_id, item.outcome.replicate_id) for item in examples
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("feature dataset decision IDs must be unique")
    dimensions = {
        level: {len(item.inputs.feature_vector(level)) for item in examples}
        for level in ("l0_uncertainty", "l1_shallow", "l2_semantic", "l3_frozen_qwen")
    }
    if any(len(values) != 1 for values in dimensions.values()):
        raise ValueError(f"feature dimensions changed across rows: {dimensions}")
    return examples


def load_predictability_feature_dataset(
    path: str | Path,
) -> tuple[dict[str, Any], list[AuditExample]]:
    import torch  # type: ignore[import-not-found]

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("predictability feature dataset must be a mapping")
    return payload, validate_predictability_feature_dataset(payload)


def _read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                state_id = str(row["state_id"])
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"invalid manifest row {path}:{line_number}") from exc
            if state_id in result:
                raise ValueError(f"duplicate manifest state_id: {state_id}")
            result[state_id] = row
    if not result:
        raise ValueError("manifest is empty")
    return result


def _atomic_torch_save(payload: object, destination: Path) -> None:
    import torch  # type: ignore[import-not-found]

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _compact_feature_row_for_torch(value: dict[str, Any]) -> dict[str, Any]:
    import torch  # type: ignore[import-not-found]

    pre_action = dict(value["pre_action"])
    for name, item in pre_action.items():
        if isinstance(item, tuple):
            pre_action[name] = torch.tensor(item, dtype=torch.float32)
    return {**value, "pre_action": pre_action}


def extract_predictability_feature_dataset(
    *,
    rollouts_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    dataset_role: str,
    model_name_or_path: str,
    revision: str,
    device_map: str = "cuda:0",
    dtype: str = "bfloat16",
    attention_implementation: str = "sdpa",
    min_pixels: int = 256 * 28 * 28,
    max_pixels: int = 768 * 28 * 28,
    local_files_only: bool = True,
    checkpoint_interval: int = 32,
    resume: bool = False,
    require_prompt_hash: bool = True,
) -> dict[str, Any]:
    """Extract the frozen L0--L3 contract from original-image state only."""

    if dataset_role not in {"train", "validation", "test", "retrospective_smoke"}:
        raise ValueError("unsupported dataset role")
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    rollout_file = Path(rollouts_path).resolve()
    manifest_file = Path(manifest_path).resolve()
    destination = Path(output_path).resolve()
    records = read_jsonl(rollout_file)
    grouped = group_by_decision(records)
    outcomes = {
        (item.state_id, item.replicate_id): item
        for item in collapse_fixed_entropy_tool(records)
    }
    manifest = _read_manifest(manifest_file)
    state_ids = {state_id for state_id, _ in grouped}
    if set(manifest) != state_ids:
        raise ValueError("manifest and rollout state coverage differ")
    metadata = {
        "schema": "predictability_feature_metadata_v1",
        "dataset_role": dataset_role,
        "rollouts": str(rollout_file),
        "rollouts_sha256": hashlib.sha256(rollout_file.read_bytes()).hexdigest(),
        "manifest": str(manifest_file),
        "manifest_sha256": hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
        "model": model_name_or_path,
        "model_revision": revision,
        "device_map": device_map,
        "dtype": dtype,
        "attention_implementation": attention_implementation,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "local_files_only": local_files_only,
        "checkpoint_interval": checkpoint_interval,
        "require_prompt_hash": require_prompt_hash,
        "code_revision": os.environ.get("BE_CODE_REVISION"),
        "shallow_feature_names": list(SHALLOW_FEATURE_NAMES),
        "l3_pooling": {
            "language": "mean final-layer non-image attended prompt tokens",
            "visual": "mean final-layer image tokens",
            "fused": "final attended prompt token before generation",
        },
        "outcomes_included_as_separate_non_input_namespace": True,
    }
    rows: list[dict[str, Any]] = []
    if destination.exists():
        if not resume:
            raise FileExistsError(f"output already exists: {destination}")
        loaded, _ = load_predictability_feature_dataset(destination)
        loaded_metadata = loaded["metadata"]
        for field in (
            "dataset_role",
            "rollouts_sha256",
            "manifest_sha256",
            "model",
            "model_revision",
            "dtype",
            "attention_implementation",
            "min_pixels",
            "max_pixels",
            "checkpoint_interval",
            "require_prompt_hash",
        ):
            if loaded_metadata.get(field) != metadata[field]:
                raise ValueError(f"resume metadata mismatch for {field}")
        metadata = dict(loaded_metadata)
        rows = list(loaded["rows"])
    completed = {(str(row["state_id"]), str(row["replicate_id"])) for row in rows}
    if completed - set(grouped):
        raise ValueError("resume checkpoint contains unexpected decisions")
    pending = [(key, grouped[key]) for key in sorted(grouped) if key not in completed]
    if not pending:
        result = {
            "format_version": PREDICTABILITY_FEATURE_FORMAT_VERSION,
            "metadata": metadata,
            "rows": rows,
        }
        validate_predictability_feature_dataset(result)
        return result

    extractor = Qwen25VLSemanticExtractor(
        model_name_or_path,
        revision=revision,
        device_map=device_map,
        dtype=dtype,
        attention_implementation=attention_implementation,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        local_files_only=local_files_only,
        question_feature_mode="input_mean",
    )
    cached_state_id: str | None = None
    cached_state: tuple[dict[str, Any], dict[str, Any], str] | None = None
    for position, (key, siblings) in enumerate(pending, start=1):
        baseline = next(item for item in siblings if item.action_type == "ANSWER")
        manifest_row = manifest[baseline.state_id]
        for field, expected in (
            ("image_id", baseline.image_id),
            ("source_id", baseline.source_id),
            ("question", baseline.question),
        ):
            if str(manifest_row[field]) != expected:
                raise ValueError(f"manifest {field} differs for {baseline.state_id}")
        backend = baseline.metadata.get("baseline_backend")
        if not isinstance(backend, Mapping):
            raise ValueError("baseline_backend metadata is missing")
        model_prompt = str(manifest_row.get("model_prompt", baseline.question))
        input_hash = backend.get("input_text_sha256")
        if (
            require_prompt_hash
            and input_hash != hashlib.sha256(model_prompt.encode()).hexdigest()
        ):
            raise ValueError("manifest model_prompt does not match baseline input hash")
        if baseline.state_id != cached_state_id:
            zooms = sorted(
                (item for item in siblings if item.action_type == "ZOOM"),
                key=lambda item: item.action_id,
            )
            semantic = extractor.encode(
                image_path=baseline.original_image,
                question=baseline.question,
                bboxes=[item.candidate_bbox for item in zooms if item.candidate_bbox],
            )
            multimodal = extractor.encode_multimodal_states(
                image_path=baseline.original_image,
                model_prompt=model_prompt,
                system_prompt=str(backend["system_prompt"]),
            )
            cached_state_id = baseline.state_id
            cached_state = (
                semantic,
                multimodal,
                decoded_rgb_sha256(baseline.original_image),
            )
        if cached_state is None:
            raise AssertionError("state feature cache was not initialized")
        semantic, multimodal, rgb_hash = cached_state
        rows.append(
            _compact_feature_row_for_torch(
                build_predictability_feature_row(
                    outcome=outcomes[key],
                    image_rgb_sha256=rgb_hash,
                    question=baseline.question,
                    baseline_answer=baseline.answer_before,
                    baseline_entropy=baseline.entropy_before,
                    baseline_backend=backend,
                    semantic=semantic,
                    multimodal=multimodal,
                )
            )
        )
        checkpoint_due = position % checkpoint_interval == 0 or position == len(pending)
        if checkpoint_due:
            _atomic_torch_save(
                {
                    "format_version": PREDICTABILITY_FEATURE_FORMAT_VERSION,
                    "metadata": metadata,
                    "rows": rows,
                },
                destination,
            )
        print(
            json.dumps(
                {
                    "decision": key,
                    "completed_this_run": position,
                    "remaining_this_run": len(pending) - position,
                    "checkpoint_written": checkpoint_due,
                }
            ),
            flush=True,
        )
    result = {
        "format_version": PREDICTABILITY_FEATURE_FORMAT_VERSION,
        "metadata": metadata,
        "rows": rows,
    }
    validate_predictability_feature_dataset(result)
    _atomic_torch_save(result, destination)
    return result
