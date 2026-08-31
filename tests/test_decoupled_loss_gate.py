from __future__ import annotations

from beyond_entropy.decoupled_loss_gate import (
    _evaluate,
    _prediction_index,
    match_call_count_threshold,
)
from beyond_entropy.schema import ActionRecord, BBox


def _decision(state: str, source: str) -> list[ActionRecord]:
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
                answer_after="right" if index == 0 else "wrong",
                correct_after=float(index == 0),
                tool_cost=1.0,
            )
        )
    return records


def test_match_call_count_threshold_preserves_ties_and_prefers_fewer_calls():
    scores = {
        ("a", "r"): 0.9,
        ("b", "r"): 0.8,
        ("c", "r"): 0.8,
    }
    result = match_call_count_threshold(scores, target_calls=2)
    assert result["threshold"] == 0.9
    assert result["calls"] == 1
    assert result["absolute_call_error"] == 1
    assert result["selection_uses_outcomes"] is False


def test_prediction_index_rejects_oracle_or_outcome_fields():
    key = ("state", "replicate-000")
    clean = [
        {
            "state_id": key[0],
            "replicate_id": key[1],
            "loss_only_action_id": "ug-grid-00",
            "loss_only_score": 0.1,
        }
    ]
    actions, scores = _prediction_index(clean, {key})
    assert actions[key] == "ug-grid-00"
    assert scores[key] == 0.1
    clean[0]["oracle_action_id"] = "ug-grid-00"
    try:
        _prediction_index(clean, {key})
    except ValueError as exc:
        assert "forbidden" in str(exc)
    else:
        raise AssertionError("oracle-bearing proposer predictions must fail")


def test_decoupled_evaluation_is_paired_and_emits_outcome_free_score_rows():
    records = [
        *_decision("state-1", "source-1"),
        *_decision("state-2", "source-2"),
        *_decision("state-3", "source-3"),
        *_decision("state-4", "source-4"),
    ]
    baselines = {}
    zooms = {}
    for record in records:
        key = (record.state_id, record.replicate_id)
        if record.action_type == "ANSWER":
            baselines[key] = record
        else:
            zooms.setdefault(key, []).append(record)
    incumbent = {key: "ug-grid-01" for key in baselines}
    decoupled = {key: "ug-grid-00" for key in baselines}
    scores = {key: 1.0 for key in baselines}
    result = _evaluate(
        baselines=baselines,
        zooms=zooms,
        actions_by_method={"incumbent": incumbent, "decoupled": decoupled},
        scores_by_method={"incumbent": scores, "decoupled": scores},
        threshold_by_method={"incumbent": 0.5, "decoupled": 0.5},
        bootstrap_resamples=100,
        bootstrap_seed=3,
    )
    assert result["source_balanced"]["incumbent"]["utility"] == -0.05
    assert result["source_balanced"]["decoupled"]["utility"] == 0.95
    assert result["primary_estimand"]["point_estimate"] == 1.0
    assert result["action_disagreement_rate"] == 1.0
    forbidden = {"gain", "harm", "correct_after", "oracle_action_id"}
    assert all(not forbidden.intersection(row) for row in result["score_rows"])
