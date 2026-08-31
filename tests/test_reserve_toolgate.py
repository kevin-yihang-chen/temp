from __future__ import annotations

from beyond_entropy.reserve_toolgate import (
    evaluate_reserve_policies,
    match_source_balanced_threshold,
)
from beyond_entropy.schema import ActionRecord, BBox


def _decision(state: str, source: str, *, helpful_action: str) -> list[ActionRecord]:
    common = {
        "state_id": state,
        "image_id": f"image-{source}",
        "source_id": source,
        "question": "q",
        "original_image": f"{source}.png",
        "replicate_id": "replicate-000",
        "generation_seed": 0,
        "entropy_before": 0.5,
        "answer_before": "wrong",
        "correct_before": 0.0,
    }
    records = [
        ActionRecord(
            **common,
            action_id="answer-now",
            action_type="ANSWER",
            candidate_bbox=None,
            entropy_after=0.5,
            answer_after="wrong",
            correct_after=0.0,
            tool_cost=0.0,
        )
    ]
    for index in range(4):
        action_id = f"ug-grid-0{index}"
        records.append(
            ActionRecord(
                **common,
                action_id=action_id,
                action_type="ZOOM",
                candidate_bbox=BBox(0.0, 0.0, 0.5, 0.5),
                entropy_after=0.4,
                answer_after="right" if action_id == helpful_action else "wrong",
                correct_after=float(action_id == helpful_action),
                tool_cost=1.0,
            )
        )
    return records


def test_match_source_balanced_threshold_preserves_ties_and_source_weights():
    scores = {
        ("a", "r"): 0.9,
        ("b", "r"): 0.8,
        ("c", "r"): 0.8,
    }
    sources = {
        ("a", "r"): "source-1",
        ("b", "r"): "source-2",
        ("c", "r"): "source-2",
    }
    result = match_source_balanced_threshold(scores, sources, target_rate=0.5)
    assert result["threshold"] == 0.9
    assert result["source_call_rate"] == 0.5
    assert result["calls"] == 1
    assert result["selection_uses_outcomes"] is False


def test_evaluate_reserve_policies_uses_frozen_shared_action_and_paired_sources():
    records = [
        *_decision("state-1", "source-1", helpful_action="ug-grid-00"),
        *_decision("state-2", "source-2", helpful_action="ug-grid-00"),
    ]
    rows = [
        {
            "state_id": state,
            "replicate_id": "replicate-000",
            "source_id": source,
            "action_id": "ug-grid-00",
            "policy_a_value": 0.1,
            "policy_a_called": True,
            "policy_b_probability": 0.01,
            "policy_b_frozen_called": False,
            "policy_b_matched_called": True,
        }
        for state, source in (("state-1", "source-1"), ("state-2", "source-2"))
    ]
    report = evaluate_reserve_policies(
        records,
        rows,
        bootstrap_resamples=200,
        bootstrap_seed=7,
    )
    assert report["supports_policy_a_over_policy_b"] is True
    assert report["primary_estimand"]["point_estimate"] == 0.95
    assert report["source_balanced"]["policy_a"]["call"] == 1.0
    assert report["source_balanced"]["policy_b_frozen"]["call"] == 0.0
    assert report["gate_disagreement"]["question_balanced_rate"] == 1.0
    assert report["formal_outcomes_used_for_thresholds"] is False


def test_evaluate_reserve_policies_rejects_outcome_leakage_in_score_rows():
    records = _decision("state-1", "source-1", helpful_action="ug-grid-00")
    rows = [
        {
            "state_id": "state-1",
            "replicate_id": "replicate-000",
            "source_id": "source-1",
            "action_id": "ug-grid-00",
            "policy_a_called": True,
            "policy_b_frozen_called": False,
            "policy_b_matched_called": False,
            "correct_after": 1.0,
        }
    ]
    try:
        evaluate_reserve_policies(records, rows, bootstrap_resamples=10)
    except ValueError as exc:
        assert "outcome fields" in str(exc)
    else:
        raise AssertionError("outcome-bearing score rows must fail closed")
