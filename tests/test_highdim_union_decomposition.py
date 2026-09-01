from __future__ import annotations

import pytest

from beyond_entropy.highdim_union_decomposition import (
    _rename_candidate,
    _score_index,
    _union_oracle_actions,
)
from beyond_entropy.schema import ActionRecord, BBox


def _record(action_id: str, gain: float) -> ActionRecord:
    return ActionRecord(
        state_id="state",
        replicate_id="replicate",
        source_id="source",
        image_id="image",
        original_image="image.png",
        question="question",
        action_id=action_id,
        action_type="ZOOM",
        candidate_bbox=BBox(0.0, 0.0, 0.5, 0.5),
        answer_before="before",
        answer_after="after",
        correct_before=0.0,
        correct_after=gain,
        entropy_before=0.1,
        entropy_after=0.1,
        tool_cost=1.0,
        generation_seed=0,
    )


def _row() -> dict[str, object]:
    return {
        "state_id": "state",
        "replicate_id": "replicate",
        "incumbent_action_id": "a",
        "highdim_diagonal_bilinear_action_id": "b",
        "incumbent_proposal_action_id": "a",
        "loss_proposal_action_id": "b",
        "incumbent_called": True,
        "highdim_diagonal_bilinear_called": False,
        "incumbent_score": 0.2,
        "highdim_diagonal_bilinear_score": -0.1,
    }


def test_score_index_accepts_outcome_free_union_rows() -> None:
    key = ("state", "replicate")
    indexed = _score_index([_row()], {key})
    assert indexed["union"][key] == ("a", "b")
    assert indexed["incumbent_call"][key] is True
    assert indexed["highdim_call"][key] is False


def test_score_index_rejects_outcome_leakage() -> None:
    row = _row()
    row["correct_after"] = 1.0
    with pytest.raises(ValueError, match="leak"):
        _score_index([row], {("state", "replicate")})


def test_union_oracle_uses_utility_then_action_id_tie_break() -> None:
    key = ("state", "replicate")
    selected = _union_oracle_actions(
        {key: ("a", "b")},
        {key: (_record("a", 0.4), _record("b", 0.8))},
    )
    assert selected[key] == "b"


def test_candidate_renaming_is_recursive() -> None:
    assert _rename_candidate(
        {"decoupled": {"decoupled_minus_incumbent": 1.0}}, "hybrid"
    ) == {"hybrid": {"hybrid_minus_incumbent": 1.0}}
