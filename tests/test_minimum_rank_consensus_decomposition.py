from __future__ import annotations

import pytest

from beyond_entropy.minimum_rank_consensus_decomposition import (
    _bucket_metrics,
    _intersection_paired,
)
from beyond_entropy.schema import ActionRecord, BBox


def _record(
    state_id: str,
    source_id: str,
    action_id: str,
    gain: float,
) -> ActionRecord:
    correct_before = 1.0 if gain < 0.0 else 0.0
    return ActionRecord(
        state_id=state_id,
        replicate_id="r",
        source_id=source_id,
        image_id=state_id,
        original_image="image.png",
        question="question",
        action_id=action_id,
        action_type="ZOOM",
        candidate_bbox=BBox(0.0, 0.0, 0.5, 0.5),
        answer_before="before",
        answer_after="after",
        correct_before=correct_before,
        correct_after=correct_before + gain,
        entropy_before=0.1,
        entropy_after=0.1,
        tool_cost=1.0,
        generation_seed=0,
    )


def test_bucket_metrics_preserve_source_balanced_contribution() -> None:
    keys = (("s0", "r"), ("s1", "r"), ("s2", "r"))
    actions = {
        keys[0]: _record("s0", "a", "x", 1.0),
        keys[1]: _record("s1", "a", "x", -1.0),
        keys[2]: _record("s2", "b", "x", 0.0),
    }
    metrics = _bucket_metrics(
        {keys[0]}, actions=actions, baselines=actions
    )
    assert metrics["calls"] == 1
    assert metrics["source_balanced_contribution"]["call"] == pytest.approx(0.25)
    assert metrics["question_balanced_contribution"]["call"] == pytest.approx(1 / 3)
    assert metrics["per_called_decision"]["utility"] == pytest.approx(0.95)


def test_intersection_paired_describes_only_realized_actions() -> None:
    key = ("s", "r")
    incumbent = {key: _record("s", "a", "i", -1.0)}
    consensus = {key: _record("s", "a", "c", 1.0)}
    result = _intersection_paired(
        {key}, incumbent_actions=incumbent, consensus_actions=consensus
    )
    assert result["action_disagreements"] == 1
    assert result["consensus_action_better_equal_worse"] == {
        "better": 1,
        "equal": 0,
        "worse": 0,
    }
    assert result["mean_consensus_minus_incumbent_per_intersection_call"][
        "utility"
    ] == pytest.approx(2.0)
