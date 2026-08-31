from __future__ import annotations

import math

import numpy as np

from beyond_entropy.dual_proposer_union_factorized_gate import (
    _candidate_balanced_weights,
    _fit_candidate_head,
    _rename_candidate,
    _union_actions,
)
from beyond_entropy.schema import ActionRecord


def _baseline(state: str, source: str) -> ActionRecord:
    return ActionRecord(
        state_id=state,
        image_id=f"image-{source}",
        source_id=source,
        question="q",
        original_image=f"{source}.png",
        replicate_id="replicate-000",
        generation_seed=0,
        action_id="answer-now",
        action_type="ANSWER",
        candidate_bbox=None,
        entropy_before=0.5,
        entropy_after=0.5,
        answer_before="wrong",
        answer_after="wrong",
        correct_before=0.0,
        correct_after=0.0,
        tool_cost=0.0,
    )


def test_union_deduplicates_equal_proposals_and_sorts_unequal_actions():
    keys = [(f"state-{index}", "replicate-000") for index in range(3)]
    incumbent = {keys[0]: "b", keys[1]: "a", keys[2]: "c"}
    loss = {keys[0]: "a", keys[1]: "a", keys[2]: "d"}
    union, counts = _union_actions(keys, incumbent_actions=incumbent, loss_actions=loss)
    assert union[keys[0]] == ("a", "b")
    assert union[keys[1]] == ("a",)
    assert union[keys[2]] == ("c", "d")
    assert counts == {"equal_proposal_pairs": 1, "unequal_proposal_pairs": 2, "unique_union_rows": 5}


def test_candidate_weights_equalize_decisions_with_one_or_two_candidates():
    key_a = ("state-a", "replicate-000")
    key_b = ("state-b", "replicate-000")
    baselines = {key_a: _baseline(key_a[0], "source"), key_b: _baseline(key_b[0], "source")}
    domains = {key_a: "d", key_b: "d"}
    pairs = [(key_a, "a"), (key_b, "a"), (key_b, "b")]
    weights = _candidate_balanced_weights(pairs, domain_by_key=domains, baselines=baselines)
    assert math.isclose(float(weights.sum()), 3.0)
    assert math.isclose(float(weights[0]), float(weights[1] + weights[2]))
    assert math.isclose(float(weights[1]), float(weights[2]))


def test_candidate_head_is_deterministic_and_not_class_balanced():
    rng = np.random.default_rng(8)
    features = rng.normal(size=(30, 46))
    labels = [int(index % 6 == 0) for index in range(30)]
    weights = np.ones(30, dtype=np.float64)
    first, first_audit = _fit_candidate_head(features, labels, weights, seed=10)
    second, second_audit = _fit_candidate_head(features, labels, weights, seed=10)
    assert first_audit == second_audit
    assert first_audit["class_balancing"] is False
    assert np.array_equal(first["model"].coef_, second["model"].coef_)


def test_dual_union_candidate_renaming_is_recursive():
    value = {"decoupled_score": 1.0, "nested": [{"decoupled_called": True}]}
    assert _rename_candidate(value) == {
        "dual_proposer_union_score": 1.0,
        "nested": [{"dual_proposer_union_called": True}],
    }
