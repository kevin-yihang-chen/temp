from __future__ import annotations

import pytest

from beyond_entropy.minimum_rank_consensus_gate import (
    _COST_FIELDS,
    _empirical_upper_percentiles,
    _score_index,
)


def test_empirical_percentiles_preserve_complete_ties() -> None:
    scores = {
        ("s0", "r"): 1.0,
        ("s1", "r"): 1.0,
        ("s2", "r"): 2.0,
        ("s3", "r"): 4.0,
    }
    ranks, lookup = _empirical_upper_percentiles(scores)
    assert ranks[("s0", "r")] == ranks[("s1", "r")] == 0.5
    assert ranks[("s2", "r")] == 0.75
    assert ranks[("s3", "r")] == 1.0
    assert lookup == {
        "raw_scores": [1.0, 2.0, 4.0],
        "percentiles": [0.5, 0.75, 1.0],
    }


def test_minimum_rank_consensus_is_conservative_in_both_scores() -> None:
    left, _ = _empirical_upper_percentiles(
        {("a", "r"): 0.0, ("b", "r"): 1.0, ("c", "r"): 2.0}
    )
    right, _ = _empirical_upper_percentiles(
        {("a", "r"): 2.0, ("b", "r"): 1.0, ("c", "r"): 0.0}
    )
    consensus = {key: min(left[key], right[key]) for key in left}
    assert consensus[("a", "r")] == pytest.approx(1.0 / 3.0)
    assert consensus[("b", "r")] == pytest.approx(2.0 / 3.0)
    assert consensus[("c", "r")] == pytest.approx(1.0 / 3.0)


def _cost_row() -> dict[str, object]:
    return {
        "state_id": "s",
        "replicate_id": "r",
        "source_id": "source",
        "outer_fold": 0,
        "cost_sensitive_direct_action_value_action_id": "a",
        "cost_sensitive_direct_action_value_score": 0.25,
        "cost_sensitive_direct_action_value_called": False,
        "incumbent_action_id": "b",
        "incumbent_score": 0.125,
        "incumbent_called": False,
    }


def test_score_index_accepts_only_exact_outcome_free_schema() -> None:
    actions, scores, calls, sources = _score_index(
        [_cost_row()],
        expected_fields=_COST_FIELDS,
        action_field="cost_sensitive_direct_action_value_action_id",
        score_field="cost_sensitive_direct_action_value_score",
        called_field="cost_sensitive_direct_action_value_called",
        name="cost-sensitive",
    )
    key = ("s", "r")
    assert actions[key] == "a"
    assert scores[key] == 0.25
    assert calls[key] is False
    assert sources[key] == "source"


def test_score_index_rejects_outcome_field() -> None:
    row = _cost_row()
    row["gain"] = 1.0
    with pytest.raises(ValueError, match="schema changed"):
        _score_index(
            [row],
            expected_fields=_COST_FIELDS,
            action_field="cost_sensitive_direct_action_value_action_id",
            score_field="cost_sensitive_direct_action_value_score",
            called_field="cost_sensitive_direct_action_value_called",
            name="cost-sensitive",
        )
