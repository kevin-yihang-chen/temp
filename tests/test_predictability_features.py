from __future__ import annotations

from pathlib import Path

import pytest

from beyond_entropy.predictability_audit import BinaryToolOutcome
from beyond_entropy.predictability_features import (
    SHALLOW_FEATURE_NAMES,
    audit_example_from_feature_row,
    build_predictability_feature_row,
    decoded_rgb_sha256,
    shallow_question_state_features,
    validate_predictability_feature_dataset,
)


def _backend() -> dict[str, object]:
    return {
        "num_observations": 1,
        "normalized_token_entropies": [0.2, 0.4],
        "generated_token_log_probabilities": [-0.1, -0.3],
        "mean_maximum_token_probability": 0.8,
        "mean_top1_top2_token_probability_margin": 0.6,
    }


def _row() -> dict:
    outcome = BinaryToolOutcome(
        state_id="state",
        replicate_id="replicate-000",
        image_id="image",
        source_id="source",
        selected_action_id="zoom-0",
        y0=0.0,
        y_tool=1.0,
        tool_cost=4.0,
        tool_calls=4,
    )
    return build_predictability_feature_row(
        outcome=outcome,
        image_rgb_sha256="a" * 64,
        question="What is 2026?",
        baseline_answer="2025",
        baseline_entropy=0.3,
        baseline_backend=_backend(),
        semantic={
            "question_embedding": [1.0, 2.0],
            "global_visual_embedding": [3.0, 4.0],
        },
        multimodal={
            "pooled_language_state": [5.0, 6.0],
            "pooled_visual_state": [7.0, 8.0],
            "fused_multimodal_state": [9.0, 10.0],
            "multimodal_prompt_tokens": 10,
            "multimodal_image_tokens": 4,
            "multimodal_language_tokens": 6,
        },
    )


def test_shallow_features_have_fixed_names_and_finite_values() -> None:
    values = shallow_question_state_features(
        question="What is 2026?", baseline_answer="2025", baseline_backend=_backend()
    )
    assert len(values) == len(SHALLOW_FEATURE_NAMES) == 19
    assert values[4] == 1.0


def test_feature_row_round_trip_exposes_no_outcome_to_input_view() -> None:
    row = _row()
    example = audit_example_from_feature_row(row)
    assert example.outcome.y_tool == 1.0
    assert example.inputs.feature_vector("l0_uncertainty") == (0.3, 0.8, 0.6)
    assert "outcome" not in example.inputs.to_feature_dict()
    examples = validate_predictability_feature_dataset(
        {"format_version": 1, "metadata": {}, "rows": [row]}
    )
    assert len(examples) == 1


def test_feature_row_rejects_nonbaseline_observation() -> None:
    backend = _backend()
    backend["num_observations"] = 2
    with pytest.raises(ValueError, match="original-image baseline"):
        build_predictability_feature_row(
            outcome=BinaryToolOutcome("s", "r", "i", "g", "z", 0.0, 1.0, 4.0, 4),
            image_rgb_sha256="b" * 64,
            question="What?",
            baseline_answer="x",
            baseline_entropy=0.2,
            baseline_backend=backend,
            semantic={"question_embedding": [1.0], "global_visual_embedding": [1.0]},
            multimodal={
                "pooled_language_state": [1.0],
                "pooled_visual_state": [1.0],
                "fused_multimodal_state": [1.0],
                "multimodal_prompt_tokens": 3,
                "multimodal_image_tokens": 1,
                "multimodal_language_tokens": 2,
            },
        )


def test_decoded_rgb_hash_depends_on_canonical_pixels_not_png_encoding(
    tmp_path: Path,
) -> None:
    image_module = pytest.importorskip("PIL.Image")
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    image = image_module.new("RGB", (2, 2), color=(1, 2, 3))
    image.save(first, compress_level=0)
    image.save(second, compress_level=9)
    assert first.read_bytes() != second.read_bytes()
    assert decoded_rgb_sha256(first) == decoded_rgb_sha256(second)
