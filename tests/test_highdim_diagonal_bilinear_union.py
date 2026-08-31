from __future__ import annotations

import numpy as np

from beyond_entropy.highdim_diagonal_bilinear_union import (
    HIGHDIM_ACTION_FEATURE_COUNT,
    HIGHDIM_EMBEDDING_DIM,
    HIGHDIM_STATE_FEATURE_COUNT,
    _fit_highdim_head,
    _highdim_features,
    _rename_candidate,
)
from beyond_entropy.schema import ActionRecord, BBox


def _records() -> tuple[ActionRecord, ActionRecord]:
    common = {
        "state_id": "state",
        "image_id": "image",
        "source_id": "source",
        "question": "where is the number?",
        "original_image": "image.png",
        "replicate_id": "replicate-000",
        "generation_seed": 0,
        "entropy_before": 0.5,
        "answer_before": "unknown",
        "correct_before": 0.0,
    }
    baseline = ActionRecord(
        **common,
        action_id="answer-now",
        action_type="ANSWER",
        candidate_bbox=None,
        entropy_after=0.5,
        answer_after="unknown",
        correct_after=0.0,
        tool_cost=0.0,
    )
    action = ActionRecord(
        **common,
        action_id="ug-grid-01",
        action_type="ZOOM",
        candidate_bbox=BBox(0.0, 0.0, 0.5, 0.5),
        entropy_after=0.4,
        answer_after="unknown",
        correct_after=0.0,
        tool_cost=1.0,
    )
    return baseline, action


def test_highdim_features_have_registered_dimensions_and_are_scale_invariant():
    baseline, action = _records()
    rng = np.random.default_rng(5)
    semantic = {
        "action_ids": ["ug-grid-00", "ug-grid-01", "ug-grid-02", "ug-grid-03"],
        "question_embedding": rng.normal(size=HIGHDIM_EMBEDDING_DIM),
        "global_visual_embedding": rng.normal(size=HIGHDIM_EMBEDDING_DIM),
        "region_embeddings": rng.normal(size=(4, HIGHDIM_EMBEDDING_DIM)),
    }
    compact = rng.normal(size=46)
    first = _highdim_features(baseline, action, semantic, compact)
    scaled = dict(semantic)
    scaled["question_embedding"] = semantic["question_embedding"] * 3.0
    scaled["global_visual_embedding"] = semantic["global_visual_embedding"] * 4.0
    scaled["region_embeddings"] = semantic["region_embeddings"] * 5.0
    second = _highdim_features(baseline, action, scaled, compact)
    assert first[0].shape == (HIGHDIM_STATE_FEATURE_COUNT,)
    assert first[1].shape == (HIGHDIM_ACTION_FEATURE_COUNT,)
    assert np.allclose(first[0], second[0], rtol=0.0, atol=1e-15)
    assert np.allclose(first[1], second[1], rtol=0.0, atol=1e-15)


def test_highdim_head_is_deterministic_and_strongly_regularized():
    rng = np.random.default_rng(6)
    features = rng.normal(size=(40, 20))
    labels = [int(index % 5 == 0) for index in range(40)]
    weights = np.ones(40, dtype=np.float64)
    first, first_audit = _fit_highdim_head(features, labels, weights, seed=7)
    second, second_audit = _fit_highdim_head(features, labels, weights, seed=7)
    assert first_audit == second_audit
    assert first_audit["C"] == 0.01
    assert np.array_equal(first["model"].coef_, second["model"].coef_)


def test_highdim_candidate_renaming_is_recursive():
    assert _rename_candidate({"decoupled_score": 1.0}) == {
        "highdim_diagonal_bilinear_score": 1.0
    }
