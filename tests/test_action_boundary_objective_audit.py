from __future__ import annotations

import math
from typing import Any

import pytest

from beyond_entropy.action_boundary_objective_audit import (
    AUDIT_SCHEMA,
    audit_zero_support_objectives,
    boltzmann_utility_target,
    expected_interventional_utility,
    expected_utility_logit_gradient,
    listwise_projection_logit_gradient,
    normalized_macro_distribution,
)


def test_normalized_macro_distribution_is_stable_and_shift_invariant() -> None:
    baseline = normalized_macro_distribution((0.0, -20.0, -20.0))
    shifted = normalized_macro_distribution((1000.0, 980.0, 980.0))
    assert baseline == pytest.approx(shifted)
    assert sum(baseline) == pytest.approx(1.0)


def test_expected_utility_gradient_is_probability_weighted() -> None:
    logits = (0.0, -20.0, -20.0)
    utilities = (0.0, 0.95, -1.05)
    probabilities = normalized_macro_distribution(logits)
    expectation = expected_interventional_utility(logits, utilities)
    gradient = expected_utility_logit_gradient(logits, utilities)
    assert gradient == pytest.approx(
        tuple(
            probability * (utility - expectation)
            for probability, utility in zip(probabilities, utilities)
        )
    )
    assert abs(gradient[1]) < 1e-8


def test_listwise_projection_is_cross_entropy_gradient() -> None:
    logits = (0.0, -20.0, -20.0)
    utilities = (0.0, 0.95, -1.05)
    policy = normalized_macro_distribution(logits)
    target = boltzmann_utility_target(utilities, temperature=0.25)
    gradient = listwise_projection_logit_gradient(logits, utilities, temperature=0.25)
    assert gradient == pytest.approx(
        tuple(probability - label for probability, label in zip(policy, target))
    )
    assert target[1] > 0.95
    assert abs(gradient[1]) > 0.9


def test_exact_underflow_separates_policy_gradient_from_supervised_projection() -> None:
    logits = (0.0, -1000.0, -1000.0)
    utilities = (0.0, 0.95, -1.05)
    assert normalized_macro_distribution(logits)[1] == 0.0
    assert expected_utility_logit_gradient(logits, utilities)[1] == 0.0
    assert (
        abs(listwise_projection_logit_gradient(logits, utilities, temperature=0.25)[1])
        > 0.9
    )


def test_audit_passes_all_registered_checks_and_serializes() -> None:
    audit = audit_zero_support_objectives()
    assert audit.decision == (
        "action_boundary_candidate_reduces_to_existing_objective_families"
    )
    assert all(audit.checks.values())
    payload = audit.to_dict()
    assert payload["schema"] == AUDIT_SCHEMA
    assert payload["decision"] == audit.decision
    assert math.isclose(sum(audit.near_zero_probabilities), 1.0)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"actions": ("a", "a")}, "unique"),
        ({"actions": ("a", "b"), "utilities": (1.0,)}, "at least two"),
        ({"near_zero_logits": (0.0, float("nan"), 1.0)}, "finite"),
        ({"target_temperature": 0.0}, "temperature"),
    ],
)
def test_audit_rejects_invalid_contracts(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        audit_zero_support_objectives(**kwargs)
