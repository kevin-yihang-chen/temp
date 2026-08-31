from __future__ import annotations

import math

import numpy as np

from beyond_entropy.proposal_conditioned_gate import (
    PROPOSAL_CONDITIONED_FEATURE_COUNT,
    _assert_source_exclusion,
    _audited_incumbent_index,
    _class_balanced_source_weights,
    _fit_conditioned_heads,
    _rename_decoupled_candidate,
    _score_conditioned_heads,
    validate_bound_loss_proposer_report,
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


def test_class_balancing_preserves_relative_base_mass_and_exact_half_classes():
    domains = ["d"] * 6
    sources = ["a", "a", "a", "b", "c", "c"]
    labels = [0, 0, 1, 1, 0, 1]
    weights, audit = _class_balanced_source_weights(domains, sources, labels)
    assert math.isclose(sum(weights[index] for index in (0, 1, 4)), 0.5)
    assert math.isclose(sum(weights[index] for index in (2, 3, 5)), 0.5)
    assert audit["negative_weight_mass"] == 0.5
    assert audit["positive_weight_mass"] == 0.5
    assert audit["total_weight_mass"] == 1.0
    assert math.isclose(weights[0], weights[1])


def test_conditioned_heads_are_deterministic_and_scores_are_finite():
    rng = np.random.default_rng(11)
    features = rng.normal(size=(40, PROPOSAL_CONDITIONED_FEATURE_COUNT))
    rescue = [int(index % 5 == 0) for index in range(40)]
    harm = [int(index % 7 == 0) for index in range(40)]
    domains = ["docvqa"] * 40
    sources = [f"source-{index // 2}" for index in range(40)]
    first, first_audit = _fit_conditioned_heads(
        features, rescue, harm, domains, sources, seed=17
    )
    second, second_audit = _fit_conditioned_heads(
        features, rescue, harm, domains, sources, seed=17
    )
    first_rescue, first_harm, first_scores = _score_conditioned_heads(
        first, features
    )
    second_rescue, second_harm, second_scores = _score_conditioned_heads(
        second, features
    )
    assert first_audit == second_audit
    assert np.array_equal(first_rescue, second_rescue)
    assert np.array_equal(first_harm, second_harm)
    assert np.array_equal(first_scores, second_scores)
    assert np.isfinite(first_scores).all()


def test_source_exclusion_and_audited_score_leakage_guards():
    keys = [(f"state-{index}", "replicate-000") for index in range(3)]
    baselines = {
        keys[0]: _baseline(keys[0][0], "source-a"),
        keys[1]: _baseline(keys[1][0], "source-b"),
        keys[2]: _baseline(keys[2][0], "source-c"),
    }
    domains = {key: "docvqa" for key in keys}
    audit = _assert_source_exclusion(
        keys[:2], keys[2:], baselines=baselines, domain_by_key=domains
    )
    assert audit["source_exclusion_passed"] is True
    clean_rows = [
        {
            "state_id": key[0],
            "replicate_id": key[1],
            "source_id": baselines[key].source_id,
            "incumbent_action_id": "ug-grid-00",
            "incumbent_score": 0.2,
            "incumbent_called": index == 0,
            "decoupled_action_id": "ug-grid-01",
        }
        for index, key in enumerate(keys)
    ]
    loss_actions = {key: "ug-grid-01" for key in keys}
    actions, scores, calls = _audited_incumbent_index(
        clean_rows, baselines=baselines, loss_actions=loss_actions
    )
    assert set(actions) == set(scores) == set(calls) == set(keys)
    clean_rows[0]["gain"] = 1.0
    try:
        _audited_incumbent_index(
            clean_rows, baselines=baselines, loss_actions=loss_actions
        )
    except ValueError as exc:
        assert "forbidden" in str(exc)
    else:
        raise AssertionError("outcome-bearing audited scores must fail")


def test_bound_report_and_candidate_field_renaming_are_exact():
    report = {
        "decision": "joint_auxiliary_proposer_not_advanced",
        "n_sources": 3500,
        "n_decisions": 13580,
        "n_folds": 5,
        "feature_count": 46,
        "feature_mode": "hybrid-context-semantic",
        "docvqa_calibration_formal_reserve_inputs_used": False,
        "fold_source_counts": {"docvqa": [700, 700, 700, 700, 700]},
    }
    audit = validate_bound_loss_proposer_report(report)
    assert audit["loss_proposer_report_validated"] is True
    renamed = _rename_decoupled_candidate(
        {"decoupled_score": 1.0, "nested": [{"decoupled_called": True}]}
    )
    assert renamed == {
        "proposal_conditioned_score": 1.0,
        "nested": [{"proposal_conditioned_called": True}],
    }
