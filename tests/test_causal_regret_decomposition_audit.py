from __future__ import annotations

import pytest

from beyond_entropy.causal_regret_decomposition_audit import (
    AUDIT_SCHEMA,
    audit_causal_regret_candidate,
    bernoulli_best_of_k_expectation,
    decompose_prefix_and_evidence_effects,
    decompose_stop_and_selection_regret,
)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "direct_utility": 0.8,
            "action_utilities": (0.6, 1.0),
            "calls_tool": True,
            "selected_action_index": 0,
        },
        {
            "direct_utility": 0.9,
            "action_utilities": (0.6, 0.7),
            "calls_tool": True,
            "selected_action_index": 0,
        },
        {
            "direct_utility": 0.4,
            "action_utilities": (0.6, 0.8),
            "calls_tool": False,
        },
    ],
)
def test_stop_and_selection_regret_are_exact_nonnegative_terms(
    kwargs: dict[str, object],
) -> None:
    result = decompose_stop_and_selection_regret(**kwargs)  # type: ignore[arg-type]
    assert result.total_regret == pytest.approx(
        result.stop_regret + result.selection_regret
    )
    assert result.stop_regret >= 0.0
    assert result.selection_regret >= 0.0


def test_prefix_and_evidence_are_signed_effects() -> None:
    result = decompose_prefix_and_evidence_effects(
        direct_utility=0.8,
        fixed_prefix_counterfactual_utility=0.9,
        fixed_prefix_real_utility=0.6,
    )
    assert result.action_prefix_effect == pytest.approx(0.1)
    assert result.visual_evidence_effect == pytest.approx(-0.3)
    assert result.total_tool_effect == pytest.approx(-0.2)
    assert result.total_tool_effect == pytest.approx(
        result.action_prefix_effect + result.visual_evidence_effect
    )


def test_best_of_k_is_not_a_replication_invariant_oracle() -> None:
    values = [bernoulli_best_of_k_expectation(0.6, k) for k in (1, 2, 4, 8)]
    assert values == pytest.approx([0.6, 0.84, 0.9744, 0.99934464])
    assert values == sorted(values)
    assert len(set(values)) == 4


def test_audit_rejects_unidentified_and_colliding_candidate() -> None:
    audit = audit_causal_regret_candidate()
    assert audit.decision == (
        "n2_additive_causal_regret_candidate_not_identified_and_not_novel"
    )
    assert audit.to_dict()["schema"] == AUDIT_SCHEMA
    assert audit.checks["stop_selection_regret_is_exactly_additive"]
    assert audit.checks["prefix_evidence_effects_are_exactly_additive"]
    assert not audit.checks["ideal_continuation_identified_from_observed_triple"]
    assert not audit.checks["candidate_distinct_from_registered_primary_literature"]


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (
            {
                "direct_utility": 0.0,
                "action_utilities": (),
                "calls_tool": False,
            },
            "non-empty",
        ),
        (
            {
                "direct_utility": 0.0,
                "action_utilities": (1.0,),
                "calls_tool": True,
            },
            "selected action",
        ),
        (
            {
                "direct_utility": 0.0,
                "action_utilities": (1.0,),
                "calls_tool": False,
                "selected_action_index": 0,
            },
            "must be absent",
        ),
    ],
)
def test_stop_selection_rejects_invalid_contracts(
    kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        decompose_stop_and_selection_regret(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "probability,k,match",
    [(-0.1, 1, r"\[0, 1\]"), (1.1, 1, r"\[0, 1\]"), (0.5, 0, "positive")],
)
def test_best_of_k_rejects_invalid_inputs(
    probability: float, k: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        bernoulli_best_of_k_expectation(probability, k)
