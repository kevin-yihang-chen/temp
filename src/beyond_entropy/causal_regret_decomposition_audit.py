from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence


AUDIT_SCHEMA = "n2_causal_regret_decomposition_audit_v1"


def _finite(name: str, value: float) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _finite_actions(values: Sequence[float]) -> tuple[float, ...]:
    normalized = tuple(_finite("action utility", value) for value in values)
    if not normalized:
        raise ValueError("action utilities must be non-empty")
    return normalized


@dataclass(frozen=True)
class StopSelectionRegret:
    direct_utility: float
    action_utilities: tuple[float, ...]
    calls_tool: bool
    selected_action_index: int | None
    oracle_utility: float
    policy_utility: float
    stop_regret: float
    selection_regret: float
    total_regret: float
    additive_residual: float


def decompose_stop_and_selection_regret(
    *,
    direct_utility: float,
    action_utilities: Sequence[float],
    calls_tool: bool,
    selected_action_index: int | None = None,
) -> StopSelectionRegret:
    direct = _finite("direct utility", direct_utility)
    actions = _finite_actions(action_utilities)
    if calls_tool:
        if selected_action_index is None or not 0 <= selected_action_index < len(
            actions
        ):
            raise ValueError("a valid selected action index is required when calling")
    elif selected_action_index is not None:
        raise ValueError("selected action index must be absent when stopping")

    best_action = max(actions)
    oracle = max(direct, best_action)
    hybrid = best_action if calls_tool else direct
    if calls_tool:
        assert selected_action_index is not None
        policy = actions[selected_action_index]
    else:
        policy = direct
    stop_regret = oracle - hybrid
    selection_regret = best_action - policy if calls_tool else 0.0
    total_regret = oracle - policy
    residual = total_regret - stop_regret - selection_regret
    return StopSelectionRegret(
        direct_utility=direct,
        action_utilities=actions,
        calls_tool=calls_tool,
        selected_action_index=selected_action_index,
        oracle_utility=oracle,
        policy_utility=policy,
        stop_regret=stop_regret,
        selection_regret=selection_regret,
        total_regret=total_regret,
        additive_residual=residual,
    )


@dataclass(frozen=True)
class PrefixEvidenceEffects:
    direct_utility: float
    fixed_prefix_counterfactual_utility: float
    fixed_prefix_real_utility: float
    action_prefix_effect: float
    visual_evidence_effect: float
    total_tool_effect: float
    additive_residual: float


def decompose_prefix_and_evidence_effects(
    *,
    direct_utility: float,
    fixed_prefix_counterfactual_utility: float,
    fixed_prefix_real_utility: float,
) -> PrefixEvidenceEffects:
    direct = _finite("direct utility", direct_utility)
    counterfactual = _finite(
        "fixed-prefix counterfactual utility", fixed_prefix_counterfactual_utility
    )
    real = _finite("fixed-prefix real utility", fixed_prefix_real_utility)
    prefix_effect = counterfactual - direct
    evidence_effect = real - counterfactual
    total_effect = real - direct
    residual = total_effect - prefix_effect - evidence_effect
    return PrefixEvidenceEffects(
        direct_utility=direct,
        fixed_prefix_counterfactual_utility=counterfactual,
        fixed_prefix_real_utility=real,
        action_prefix_effect=prefix_effect,
        visual_evidence_effect=evidence_effect,
        total_tool_effect=total_effect,
        additive_residual=residual,
    )


def bernoulli_best_of_k_expectation(success_probability: float, k: int) -> float:
    probability = _finite("success probability", success_probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("success probability must lie in [0, 1]")
    if k <= 0:
        raise ValueError("k must be positive")
    return 1.0 - (1.0 - probability) ** k


@dataclass(frozen=True)
class CausalRegretDecompositionAudit:
    stop_selection_examples: tuple[StopSelectionRegret, ...]
    prefix_evidence_example: PrefixEvidenceEffects
    observationally_equivalent_ideal_continuations: tuple[float, ...]
    evidence_use_regrets_under_equivalent_worlds: tuple[float, ...]
    best_of_k_oracle_expectations: dict[int, float]
    literature_collision: dict[str, bool]
    checks: dict[str, bool]
    decision: str

    def to_dict(self) -> dict[str, object]:
        return {"schema": AUDIT_SCHEMA, **asdict(self)}


def audit_causal_regret_candidate() -> CausalRegretDecompositionAudit:
    examples = (
        decompose_stop_and_selection_regret(
            direct_utility=0.8,
            action_utilities=(0.6, 1.0),
            calls_tool=True,
            selected_action_index=0,
        ),
        decompose_stop_and_selection_regret(
            direct_utility=0.9,
            action_utilities=(0.6, 0.7),
            calls_tool=True,
            selected_action_index=0,
        ),
        decompose_stop_and_selection_regret(
            direct_utility=0.4,
            action_utilities=(0.6, 0.8),
            calls_tool=False,
        ),
    )
    effects = decompose_prefix_and_evidence_effects(
        direct_utility=0.8,
        fixed_prefix_counterfactual_utility=0.9,
        fixed_prefix_real_utility=0.6,
    )

    ideal_continuations = (0.6, 1.0)
    evidence_use_regrets = tuple(
        ideal - effects.fixed_prefix_real_utility for ideal in ideal_continuations
    )
    best_of_k = {k: bernoulli_best_of_k_expectation(0.6, k) for k in (1, 2, 4, 8)}
    collision = {
        "the_illusion_fixed_prefix_observation_effect": True,
        "the_illusion_action_induced_shortcut": True,
        "gapsight_stop_and_action_utility": True,
    }
    checks = {
        "stop_selection_regret_is_exactly_additive": all(
            abs(example.additive_residual) < 1e-12 for example in examples
        ),
        "stop_selection_terms_are_nonnegative": all(
            example.stop_regret >= 0.0 and example.selection_regret >= 0.0
            for example in examples
        ),
        "prefix_evidence_effects_are_exactly_additive": abs(effects.additive_residual)
        < 1e-12,
        "prefix_or_evidence_effect_can_be_negative": (
            effects.action_prefix_effect < 0.0 or effects.visual_evidence_effect < 0.0
        ),
        "ideal_continuation_identified_from_observed_triple": len(
            set(evidence_use_regrets)
        )
        == 1,
        "best_of_k_oracle_invariant_to_replication_count": len(
            {round(value, 12) for value in best_of_k.values()}
        )
        == 1,
        "candidate_distinct_from_registered_primary_literature": not any(
            collision.values()
        ),
    }
    decision = (
        "n2_additive_causal_regret_candidate_passed"
        if all(checks.values())
        else "n2_additive_causal_regret_candidate_not_identified_and_not_novel"
    )
    return CausalRegretDecompositionAudit(
        stop_selection_examples=examples,
        prefix_evidence_example=effects,
        observationally_equivalent_ideal_continuations=ideal_continuations,
        evidence_use_regrets_under_equivalent_worlds=evidence_use_regrets,
        best_of_k_oracle_expectations=best_of_k,
        literature_collision=collision,
        checks=checks,
        decision=decision,
    )
