from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from beyond_entropy.infographicvqa_attention_signed_stop import (  # noqa: E402
    ATTENTION_SIGNED_STOP_FEATURE_COUNT,
    ATTENTION_SIGNED_STOP_SCHEMA,
    _source_utility_weights,
    evaluate_attention_signed_stop,
    smoke_attention_signed_stop,
)
from beyond_entropy.infographicvqa_decar import DECAR_ACTION_IDS  # noqa: E402
from beyond_entropy.schema import ActionRecord, BBox  # noqa: E402


def _decision(index: int) -> list[ActionRecord]:
    source_index = index // 2
    positive = index % 2 == 0
    entropy = 0.1 + 0.01 * index
    key = {
        "state_id": f"state-{index:03d}",
        "image_id": f"image-{index:03d}",
        "source_id": f"source-{source_index:03d}",
        "question": "What is shown?",
        "original_image": f"image-{index:03d}.png",
        "replicate_id": "replicate-000",
        "generation_seed": index,
        "entropy_before": entropy,
        "answer_before": "before",
        "correct_before": 0.0,
        "metadata": {},
    }
    baseline = ActionRecord(
        **key,
        action_id="answer-now",
        action_type="ANSWER",
        candidate_bbox=None,
        entropy_after=entropy,
        answer_after="before",
        correct_after=0.0,
        tool_cost=0.0,
        pre_action_features={},
    )
    zooms = []
    for action_index, action_id in enumerate(DECAR_ACTION_IDS):
        selected = action_index == index % 4
        zooms.append(
            ActionRecord(
                **key,
                action_id=action_id,
                action_type="ZOOM",
                candidate_bbox=BBox(
                    0.25 * action_index,
                    0.0,
                    0.25 * (action_index + 1),
                    1.0,
                ),
                entropy_after=0.05,
                answer_after="after",
                correct_after=1.0 if selected and positive else 0.0,
                tool_cost=1.0,
                pre_action_features={"ug_grid_size": 4.0},
            )
        )
    return [baseline, *zooms]


def _payload(decisions: int) -> dict:
    rows = []
    for index in range(decisions):
        selected = index % 4
        attention = torch.full((4,), 0.1)
        attention[selected] = 0.7
        positive_signal = float(index % 2 == 0)
        rows.append(
            {
                "state_id": f"state-{index:03d}",
                "replicate_id": "replicate-000",
                "source_id": f"source-{index // 2:03d}",
                "image_id": f"image-{index:03d}",
                "question": "What is shown?",
                "original_image": f"image-{index:03d}.png",
                "action_ids": list(DECAR_ACTION_IDS),
                "question_embedding": torch.tensor([1.0, positive_signal, 0.5]),
                "global_visual_embedding": torch.tensor([0.5, 1.0, 0.25]),
                "region_embeddings": torch.tensor(
                    [
                        [1.0, 0.0, 0.1],
                        [0.0, 1.0, 0.2],
                        [0.5, 0.5, 0.3],
                        [0.2, 0.1, 1.0],
                    ]
                ),
                "bboxes": torch.tensor(
                    [[0.25 * i, 0.0, 0.25 * (i + 1), 1.0] for i in range(4)]
                ),
                "state_signals": torch.tensor([0.1 + 0.01 * index]),
                "tool_costs": torch.ones(4),
                "question_region_attention": attention,
                "question_image_attention_mass": 1.0 + 0.01 * index,
            }
        )
    return {
        "format_version": 1,
        "metadata": {
            "outcomes_included": False,
            "question_region_attention": {
                "source_features_sha256": "features",
                "source_rollouts_sha256": "rollouts",
                "model_revision": "model",
                "attention_implementation": "eager",
                "top_layers": 4,
                "head_pooling": "mean",
                "question_token_pooling": "mean",
                "candidate_pooling": "ROI mean then normalize across candidates",
                "candidate_actions_executed": False,
                "replace_question_embedding": False,
                "code_revision": "code",
                "completed_decisions": decisions,
                "total_decisions": decisions,
            },
        },
        "decisions": rows,
    }


def _shared(decisions: int, sources: int) -> dict:
    return {
        "expected_attention_code_revision": "code",
        "expected_model_revision": "model",
        "expected_source_features_sha256": "features",
        "expected_rollouts_sha256": "rollouts",
        "expected_decisions": decisions,
        "expected_sources": sources,
        "expected_positive_net_states": decisions // 2,
    }


def test_source_utility_weights_give_each_source_equal_mass() -> None:
    weights = _source_utility_weights([0.95, -0.05, -0.05], ["a", "a", "b"])
    assert weights[:2].sum() == pytest.approx(weights[2])
    assert weights.sum() == pytest.approx(3.0)


def test_real_feature_smoke_is_performance_free_and_source_disjoint() -> None:
    decisions = 20
    records = [record for index in range(decisions) for record in _decision(index)]
    smoke = smoke_attention_signed_stop(
        records, _payload(decisions), **_shared(decisions, 10)
    )
    assert smoke["passed"] is True
    assert smoke["fit_performed"] is False
    assert smoke["policy_metrics_computed"] is False
    assert smoke["feature_count"] == ATTENTION_SIGNED_STOP_FEATURE_COUNT
    assert all(fold["source_overlap"] == 0 for fold in smoke["folds"])


def test_signed_stop_produces_complete_outcome_free_oof_scores() -> None:
    decisions = 20
    records = [record for index in range(decisions) for record in _decision(index)]
    bootstrap = np.tile(np.arange(10, dtype=np.int32), (100, 1))
    report, model, scores = evaluate_attention_signed_stop(
        records,
        _payload(decisions),
        bootstrap_indices=bootstrap,
        expected_bootstrap_resamples=100,
        **_shared(decisions, 10),
    )
    assert report["schema"] == ATTENTION_SIGNED_STOP_SCHEMA
    assert report["validation_or_test_inputs_used"] is False
    assert len(scores) == decisions
    assert len({(row["state_id"], row["replicate_id"]) for row in scores}) == decisions
    forbidden = {"target", "utility", "gain", "correct_before", "correct_after"}
    assert all(not forbidden.intersection(row) for row in scores)
    assert len(model["folds"]) == 5
    assert all(fold["source_overlap"] == 0 for fold in report["fold_audits"])
