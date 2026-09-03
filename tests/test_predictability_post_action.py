from __future__ import annotations

import json
from pathlib import Path

import pytest

from beyond_entropy.predictability_features import build_post_action_probe_features
from beyond_entropy.predictability_matrix_smoke import build_synthetic_datasets
from beyond_entropy.predictability_post_action import (
    POST_ACTION_PROBE_HIDDEN_LAYERS,
    PostActionProbeInputs,
    evaluate_frozen_post_action_probe,
    fit_frozen_post_action_probe,
)
from beyond_entropy.schema import ActionRecord
from test_predictability_baselines import _siblings


def _with_backend_metadata(record: ActionRecord) -> ActionRecord:
    baseline = {
        "num_observations": 1,
        "mean_maximum_token_probability": 0.7,
        "mean_top1_top2_token_probability_margin": 0.4,
    }
    metadata: dict[str, object] = {"baseline_backend": baseline}
    if record.action_type == "ZOOM":
        metadata["action_backend"] = {
            "num_observations": 2,
            "mean_maximum_token_probability": 0.8,
            "mean_top1_top2_token_probability_margin": 0.5,
        }
    return ActionRecord.from_dict({**record.to_dict(), "metadata": metadata})


def test_post_action_feature_builder_uses_frozen_search_trace_without_labels() -> None:
    siblings = [
        _with_backend_metadata(item)
        for item in _siblings(
            state_id="state",
            source_id="source",
            entropy_before=0.7,
            y0=0.0,
            crop_outcomes=(0.0, 1.0, 0.0, 0.0),
            entropy_order=(2, 0, 3, 1),
        )
    ]
    features = build_post_action_probe_features(
        siblings,
        multimodal={
            "pooled_language_state": [1.0, 2.0],
            "pooled_visual_state": [3.0, 4.0],
            "fused_multimodal_state": [5.0, 6.0],
            "multimodal_prompt_tokens": 20,
            "multimodal_image_tokens": 12,
            "multimodal_language_tokens": 8,
        },
    )
    assert features["selected_action_id"] == "ug-grid-01"
    assert features["selected_action_one_hot"] == (0.0, 1.0, 0.0, 0.0)
    assert len(features["candidate_post_action_confidence"]) == 12
    assert not any(
        fragment in name
        for name in features
        for fragment in ("correct", "ground_truth", "answer_after", "y_tool")
    )


def test_post_action_typed_view_rejects_unregistered_fields() -> None:
    dataset = build_synthetic_datasets()["chartqa"]
    item = dataset.post_action_train[0].inputs
    row = {
        "state_id": item.state_id,
        "image_id": item.image_id,
        "source_id": item.source_id,
        "post_action_probe": {
            "selected_action_id": item.selected_action_id,
            "candidate_action_ids": [
                "ug-grid-00",
                "ug-grid-01",
                "ug-grid-02",
                "ug-grid-03",
            ],
            "baseline_confidence": item.baseline_confidence,
            "candidate_post_action_confidence": item.candidate_post_action_confidence,
            "selected_bbox": item.selected_bbox,
            "selected_action_one_hot": item.selected_action_one_hot,
            "pooled_language_state": item.pooled_language_state,
            "pooled_visual_state": item.pooled_visual_state,
            "fused_multimodal_state": item.fused_multimodal_state,
            "correct_after": 1.0,
        },
    }
    with pytest.raises(ValueError, match="unknown post_action_probe fields"):
        PostActionProbeInputs.from_untrusted_mapping(row)


def test_fixed_post_action_probe_fits_without_variant_search() -> None:
    dataset = build_synthetic_datasets()["chartqa"]
    probe = fit_frozen_post_action_probe(
        dataset.post_action_train,
        dataset.post_action_validation,
        seed=17,
        lambda_cost=0.05,
    )
    predictions, metrics = evaluate_frozen_post_action_probe(
        probe,
        dataset.post_action_test,
        lambda_cost=0.05,
    )
    assert probe.estimator[-1].hidden_layer_sizes == POST_ACTION_PROBE_HIDDEN_LAYERS
    assert probe.input_dimension == 29
    assert len(predictions) == len(dataset.post_action_test)
    assert metrics["decisions"] == len(dataset.post_action_test)


def test_machine_protocol_freezes_the_single_post_action_probe() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (root / "configs/predictability_audit_v1.json").read_text(encoding="utf-8")
    )["post_action_probe"]
    assert protocol["count"] == 1
    assert protocol["target"] == "direct_gain"
    assert protocol["model"] == "fixed_two_layer_mlp"
    assert protocol["hidden_layers"] == [128, 32]
    assert protocol["feature_format_version"] == 2
    assert protocol["role"] == "diagnostic_only_never_deployable"
    assert "correct_after" in protocol["excludes"]
