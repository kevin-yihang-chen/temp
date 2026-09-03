from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable, Sequence


AUDIT_SCHEMA = "action_boundary_objective_zero_support_audit_v1"


def _finite_vector(name: str, values: Sequence[float]) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    if len(normalized) < 2:
        raise ValueError(f"{name} must contain at least two values")
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError(f"{name} must contain only finite values")
    return normalized


def normalized_macro_distribution(logits: Sequence[float]) -> tuple[float, ...]:
    """Return a stable categorical distribution over finite typed macro-actions."""

    normalized = _finite_vector("logits", logits)
    maximum = max(normalized)
    exponentials = tuple(math.exp(value - maximum) for value in normalized)
    denominator = sum(exponentials)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("logits do not define a finite categorical distribution")
    return tuple(value / denominator for value in exponentials)


def expected_interventional_utility(
    logits: Sequence[float], utilities: Sequence[float]
) -> float:
    probabilities, normalized_utilities = _matched_vectors(logits, utilities)
    return sum(
        probability * utility
        for probability, utility in zip(probabilities, normalized_utilities)
    )


def expected_utility_logit_gradient(
    logits: Sequence[float], utilities: Sequence[float]
) -> tuple[float, ...]:
    """Exact gradient of E_{a~pi}[U(a)] with respect to macro-action logits."""

    probabilities, normalized_utilities = _matched_vectors(logits, utilities)
    expectation = sum(
        probability * utility
        for probability, utility in zip(probabilities, normalized_utilities)
    )
    return tuple(
        probability * (utility - expectation)
        for probability, utility in zip(probabilities, normalized_utilities)
    )


def boltzmann_utility_target(
    utilities: Sequence[float], *, temperature: float
) -> tuple[float, ...]:
    normalized = _finite_vector("utilities", utilities)
    normalized_temperature = float(temperature)
    if not math.isfinite(normalized_temperature) or normalized_temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    return normalized_macro_distribution(
        tuple(value / normalized_temperature for value in normalized)
    )


def listwise_projection_loss(
    logits: Sequence[float], utilities: Sequence[float], *, temperature: float
) -> float:
    probabilities, normalized_utilities = _matched_vectors(logits, utilities)
    target = boltzmann_utility_target(normalized_utilities, temperature=temperature)
    return -sum(
        target_probability * math.log(max(policy_probability, 1e-300))
        for target_probability, policy_probability in zip(target, probabilities)
    )


def listwise_projection_logit_gradient(
    logits: Sequence[float], utilities: Sequence[float], *, temperature: float
) -> tuple[float, ...]:
    """Gradient of cross-entropy from a utility-induced listwise target."""

    probabilities, normalized_utilities = _matched_vectors(logits, utilities)
    target = boltzmann_utility_target(normalized_utilities, temperature=temperature)
    return tuple(
        probability - target_probability
        for probability, target_probability in zip(probabilities, target)
    )


def _matched_vectors(
    logits: Sequence[float], utilities: Sequence[float]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    probabilities = normalized_macro_distribution(logits)
    normalized_utilities = _finite_vector("utilities", utilities)
    if len(probabilities) != len(normalized_utilities):
        raise ValueError("logits and utilities must have equal length")
    return probabilities, normalized_utilities


def _finite_difference_gradient(
    objective: Callable[[tuple[float, ...]], float],
    logits: Sequence[float],
    *,
    epsilon: float = 1e-6,
) -> tuple[float, ...]:
    normalized = _finite_vector("logits", logits)
    estimates: list[float] = []
    for index in range(len(normalized)):
        positive = list(normalized)
        negative = list(normalized)
        positive[index] += epsilon
        negative[index] -= epsilon
        estimates.append(
            (objective(tuple(positive)) - objective(tuple(negative))) / (2.0 * epsilon)
        )
    return tuple(estimates)


@dataclass(frozen=True)
class ActionBoundaryObjectiveAudit:
    actions: tuple[str, ...]
    utilities: tuple[float, ...]
    near_zero_logits: tuple[float, ...]
    underflow_logits: tuple[float, ...]
    target_temperature: float
    near_zero_probabilities: tuple[float, ...]
    underflow_probabilities: tuple[float, ...]
    expected_utility_gradient: tuple[float, ...]
    listwise_projection_target: tuple[float, ...]
    listwise_projection_gradient: tuple[float, ...]
    underflow_expected_utility_gradient: tuple[float, ...]
    underflow_listwise_projection_gradient: tuple[float, ...]
    finite_difference_expected_gradient_max_error: float
    finite_difference_projection_gradient_max_error: float
    checks: dict[str, bool]
    decision: str

    def to_dict(self) -> dict[str, object]:
        return {"schema": AUDIT_SCHEMA, **asdict(self)}


def audit_zero_support_objectives(
    *,
    actions: Sequence[str] = ("answer_now", "beneficial_tool", "harmful_tool"),
    utilities: Sequence[float] = (0.0, 0.95, -1.05),
    near_zero_logits: Sequence[float] = (0.0, -20.0, -20.0),
    underflow_logits: Sequence[float] = (0.0, -1000.0, -1000.0),
    target_temperature: float = 0.25,
) -> ActionBoundaryObjectiveAudit:
    normalized_actions = tuple(str(action) for action in actions)
    if len(normalized_actions) < 2 or any(not action for action in normalized_actions):
        raise ValueError("actions must contain at least two non-empty names")
    if len(normalized_actions) != len(set(normalized_actions)):
        raise ValueError("action names must be unique")
    normalized_utilities = _finite_vector("utilities", utilities)
    normalized_near_zero = _finite_vector("near_zero_logits", near_zero_logits)
    normalized_underflow = _finite_vector("underflow_logits", underflow_logits)
    if not (
        len(normalized_actions)
        == len(normalized_utilities)
        == len(normalized_near_zero)
        == len(normalized_underflow)
    ):
        raise ValueError("actions, utilities, and both logit vectors must align")
    beneficial_index = normalized_actions.index("beneficial_tool")

    near_zero_probabilities = normalized_macro_distribution(normalized_near_zero)
    underflow_probabilities = normalized_macro_distribution(normalized_underflow)
    expected_gradient = expected_utility_logit_gradient(
        normalized_near_zero, normalized_utilities
    )
    target = boltzmann_utility_target(
        normalized_utilities, temperature=target_temperature
    )
    projection_gradient = listwise_projection_logit_gradient(
        normalized_near_zero,
        normalized_utilities,
        temperature=target_temperature,
    )
    underflow_expected_gradient = expected_utility_logit_gradient(
        normalized_underflow, normalized_utilities
    )
    underflow_projection_gradient = listwise_projection_logit_gradient(
        normalized_underflow,
        normalized_utilities,
        temperature=target_temperature,
    )

    expected_numeric = _finite_difference_gradient(
        lambda candidate: expected_interventional_utility(
            candidate, normalized_utilities
        ),
        normalized_near_zero,
    )
    projection_numeric = _finite_difference_gradient(
        lambda candidate: listwise_projection_loss(
            candidate,
            normalized_utilities,
            temperature=target_temperature,
        ),
        normalized_near_zero,
    )
    expected_error = max(
        abs(analytic - numeric)
        for analytic, numeric in zip(expected_gradient, expected_numeric)
    )
    projection_error = max(
        abs(analytic - numeric)
        for analytic, numeric in zip(projection_gradient, projection_numeric)
    )

    checks = {
        "probabilities_sum_to_one": math.isclose(
            sum(near_zero_probabilities), 1.0, abs_tol=1e-15
        ),
        "expected_gradient_matches_finite_difference": expected_error < 1e-9,
        # The loss is O(20) here; central differences at 1e-6 accumulate
        # roughly 1e-9 absolute floating-point error.
        "projection_gradient_matches_finite_difference": projection_error < 1e-8,
        "beneficial_policy_probability_is_near_zero": (
            near_zero_probabilities[beneficial_index] < 1e-8
        ),
        "expected_utility_gradient_is_support_suppressed": (
            abs(expected_gradient[beneficial_index]) < 1e-8
        ),
        "utility_target_prefers_beneficial_action": target[beneficial_index] > 0.95,
        "listwise_projection_has_nonzero_beneficial_gradient": (
            abs(projection_gradient[beneficial_index]) > 0.9
        ),
        "underflow_yields_exact_zero_policy_support": (
            underflow_probabilities[beneficial_index] == 0.0
        ),
        "expected_utility_gradient_is_zero_at_exact_zero_support": (
            underflow_expected_gradient[beneficial_index] == 0.0
        ),
        "listwise_projection_bypasses_support_with_supervised_target": (
            abs(underflow_projection_gradient[beneficial_index]) > 0.9
        ),
    }
    decision = (
        "action_boundary_candidate_reduces_to_existing_objective_families"
        if all(checks.values())
        else "action_boundary_objective_audit_failed"
    )
    return ActionBoundaryObjectiveAudit(
        actions=normalized_actions,
        utilities=normalized_utilities,
        near_zero_logits=normalized_near_zero,
        underflow_logits=normalized_underflow,
        target_temperature=float(target_temperature),
        near_zero_probabilities=near_zero_probabilities,
        underflow_probabilities=underflow_probabilities,
        expected_utility_gradient=expected_gradient,
        listwise_projection_target=target,
        listwise_projection_gradient=projection_gradient,
        underflow_expected_utility_gradient=underflow_expected_gradient,
        underflow_listwise_projection_gradient=underflow_projection_gradient,
        finite_difference_expected_gradient_max_error=expected_error,
        finite_difference_projection_gradient_max_error=projection_error,
        checks=checks,
        decision=decision,
    )
