from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence


AUDIT_SCHEMA = "n4_selector_information_boundary_audit_v1"
REGISTRY_SCHEMA = "n4_selector_information_boundary_registry_v1"


def _finite(name: str, value: float) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _required_text(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class UtilityWorld:
    world_id: str
    observable_state_id: str
    probability: float
    action_utilities: tuple[float, ...]

    @classmethod
    def build(
        cls,
        *,
        world_id: str,
        observable_state_id: str,
        probability: float,
        action_utilities: Sequence[float],
    ) -> "UtilityWorld":
        if not world_id:
            raise ValueError("world_id must be non-empty")
        if not observable_state_id:
            raise ValueError("observable_state_id must be non-empty")
        normalized_probability = _finite("probability", probability)
        if normalized_probability <= 0.0:
            raise ValueError("probability must be positive")
        normalized_utilities = tuple(
            _finite("action utility", value) for value in action_utilities
        )
        if not normalized_utilities:
            raise ValueError("action utilities must be non-empty")
        return cls(
            world_id=world_id,
            observable_state_id=observable_state_id,
            probability=normalized_probability,
            action_utilities=normalized_utilities,
        )


@dataclass(frozen=True)
class InformationBoundaryDecomposition:
    action_count: int
    full_information_value: float
    observable_bayes_value: float
    policy_value: float
    aliasing_regret: float
    policy_estimation_regret: float
    total_regret: float
    additive_residual: float
    bayes_actions_by_observable_state: dict[str, tuple[int, ...]]
    policy_actions_by_observable_state: dict[str, int]


@dataclass(frozen=True)
class SelectorEvaluationArm:
    method_id: str
    information_set_id: str
    selector_visible_fields: tuple[str, ...]
    action_bank_id: str
    utility_definition_id: str
    task_utility: float
    acquisition_cost: float
    proposer_cost: float

    @classmethod
    def build(
        cls,
        *,
        method_id: str,
        information_set_id: str,
        selector_visible_fields: Sequence[str],
        action_bank_id: str,
        utility_definition_id: str,
        task_utility: float,
        acquisition_cost: float,
        proposer_cost: float,
    ) -> "SelectorEvaluationArm":
        fields = tuple(sorted(str(field).strip() for field in selector_visible_fields))
        if not fields or any(not field for field in fields):
            raise ValueError("selector-visible fields must be non-empty strings")
        if len(set(fields)) != len(fields):
            raise ValueError("selector-visible fields must be unique")
        normalized_acquisition_cost = _finite("acquisition cost", acquisition_cost)
        normalized_proposer_cost = _finite("proposer cost", proposer_cost)
        if normalized_acquisition_cost < 0.0 or normalized_proposer_cost < 0.0:
            raise ValueError("costs must be nonnegative")
        return cls(
            method_id=_required_text({"method_id": method_id}, "method_id"),
            information_set_id=_required_text(
                {"information_set_id": information_set_id}, "information_set_id"
            ),
            selector_visible_fields=fields,
            action_bank_id=_required_text(
                {"action_bank_id": action_bank_id}, "action_bank_id"
            ),
            utility_definition_id=_required_text(
                {"utility_definition_id": utility_definition_id},
                "utility_definition_id",
            ),
            task_utility=_finite("task utility", task_utility),
            acquisition_cost=normalized_acquisition_cost,
            proposer_cost=normalized_proposer_cost,
        )

    @property
    def cost_adjusted_utility(self) -> float:
        return self.task_utility - self.acquisition_cost - self.proposer_cost


def rank_matched_selector_arms(
    arms: Sequence[SelectorEvaluationArm],
) -> tuple[str, ...]:
    normalized = tuple(arms)
    if len(normalized) < 2:
        raise ValueError("a matched comparison requires at least two arms")
    if len({arm.method_id for arm in normalized}) != len(normalized):
        raise ValueError("matched comparison method IDs must be unique")
    boundaries = {
        (
            arm.information_set_id,
            arm.selector_visible_fields,
            arm.action_bank_id,
            arm.utility_definition_id,
        )
        for arm in normalized
    }
    if len(boundaries) != 1:
        raise ValueError("selector comparison has a mismatched information boundary")
    return tuple(
        arm.method_id
        for arm in sorted(
            normalized,
            key=lambda arm: (-arm.cost_adjusted_utility, arm.method_id),
        )
    )


def pairwise_rank_reversals(
    first_values: Mapping[str, float], second_values: Mapping[str, float]
) -> tuple[tuple[str, str], ...]:
    if set(first_values) != set(second_values) or len(first_values) < 2:
        raise ValueError("rank-reversal inputs must contain the same methods")
    methods = tuple(sorted(first_values))
    normalized_first = {
        method: _finite("first utility", first_values[method]) for method in methods
    }
    normalized_second = {
        method: _finite("second utility", second_values[method]) for method in methods
    }
    reversals: list[tuple[str, str]] = []
    for left_index, left in enumerate(methods):
        for right in methods[left_index + 1 :]:
            first_difference = normalized_first[left] - normalized_first[right]
            second_difference = normalized_second[left] - normalized_second[right]
            if first_difference * second_difference < 0.0:
                reversals.append((left, right))
    return tuple(reversals)


def _validate_worlds(worlds: Sequence[UtilityWorld]) -> tuple[UtilityWorld, ...]:
    normalized = tuple(worlds)
    if not normalized:
        raise ValueError("worlds must be non-empty")
    if len({world.world_id for world in normalized}) != len(normalized):
        raise ValueError("world IDs must be unique")
    action_counts = {len(world.action_utilities) for world in normalized}
    if len(action_counts) != 1:
        raise ValueError("all worlds must share one action space")
    probability_sum = math.fsum(world.probability for world in normalized)
    if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("world probabilities must sum to one")
    return normalized


def decompose_selector_regret(
    worlds: Sequence[UtilityWorld],
    *,
    policy_actions_by_observable_state: Mapping[str, int],
) -> InformationBoundaryDecomposition:
    """Split regret into unavailable-information and learnable-policy terms.

    Utilities must already include all action, latency, token, and proposer costs.
    The policy is required to be measurable with respect to ``observable_state_id``:
    aliased worlds therefore receive the same action.
    """

    normalized = _validate_worlds(worlds)
    action_count = len(normalized[0].action_utilities)
    observable_ids = {world.observable_state_id for world in normalized}
    if set(policy_actions_by_observable_state) != observable_ids:
        raise ValueError("policy must define exactly one action per observable state")
    for action in policy_actions_by_observable_state.values():
        if type(action) is not int or not 0 <= action < action_count:
            raise ValueError("policy action is outside the shared action space")

    full_information_value = math.fsum(
        world.probability * max(world.action_utilities) for world in normalized
    )
    bayes_actions: dict[str, tuple[int, ...]] = {}
    observable_bayes_value = 0.0
    for observable_id in sorted(observable_ids):
        group = tuple(
            world for world in normalized if world.observable_state_id == observable_id
        )
        weighted_action_values = tuple(
            math.fsum(
                world.probability * world.action_utilities[action] for world in group
            )
            for action in range(action_count)
        )
        best_value = max(weighted_action_values)
        best_actions = tuple(
            action
            for action, value in enumerate(weighted_action_values)
            if math.isclose(value, best_value, rel_tol=0.0, abs_tol=1e-12)
        )
        bayes_actions[observable_id] = best_actions
        observable_bayes_value += best_value

    policy_value = math.fsum(
        world.probability
        * world.action_utilities[
            policy_actions_by_observable_state[world.observable_state_id]
        ]
        for world in normalized
    )
    aliasing_regret = full_information_value - observable_bayes_value
    policy_regret = observable_bayes_value - policy_value
    total_regret = full_information_value - policy_value
    residual = total_regret - aliasing_regret - policy_regret
    return InformationBoundaryDecomposition(
        action_count=action_count,
        full_information_value=full_information_value,
        observable_bayes_value=observable_bayes_value,
        policy_value=policy_value,
        aliasing_regret=aliasing_regret,
        policy_estimation_regret=policy_regret,
        total_regret=total_regret,
        additive_residual=residual,
        bayes_actions_by_observable_state=bayes_actions,
        policy_actions_by_observable_state=dict(
            sorted(policy_actions_by_observable_state.items())
        ),
    )


def policy_is_observable(
    worlds: Sequence[UtilityWorld], *, actions_by_world_id: Mapping[str, int]
) -> bool:
    normalized = _validate_worlds(worlds)
    if set(actions_by_world_id) != {world.world_id for world in normalized}:
        raise ValueError("world policy must define exactly one action per world")
    actions_by_observation: dict[str, int] = {}
    for world in normalized:
        action = actions_by_world_id[world.world_id]
        if type(action) is not int or not 0 <= action < len(world.action_utilities):
            raise ValueError("world policy action is outside the shared action space")
        previous = actions_by_observation.setdefault(world.observable_state_id, action)
        if previous != action:
            return False
    return True


def _block_mean_preview(
    raster: Sequence[Sequence[int]], *, block_width: int, block_height: int
) -> tuple[tuple[float, ...], ...]:
    rows = tuple(tuple(int(value) for value in row) for row in raster)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("raster must be a non-empty rectangle")
    if block_width <= 0 or block_height <= 0:
        raise ValueError("block dimensions must be positive")
    height = len(rows)
    width = len(rows[0])
    if width % block_width or height % block_height:
        raise ValueError("block dimensions must divide the raster exactly")
    return tuple(
        tuple(
            math.fsum(
                rows[y][x]
                for y in range(top, top + block_height)
                for x in range(left, left + block_width)
            )
            / (block_width * block_height)
            for left in range(0, width, block_width)
        )
        for top in range(0, height, block_height)
    )


def exact_visual_alias_fixture() -> dict[str, object]:
    """Create two distinct high-frequency worlds with one identical preview."""

    checker = ((0, 2), (2, 0))
    flat = ((1, 1), (1, 1))
    world_left = tuple(
        checker_row + flat_row for checker_row, flat_row in zip(checker, flat)
    )
    world_right = tuple(
        flat_row + checker_row for checker_row, flat_row in zip(checker, flat)
    )
    preview_left = _block_mean_preview(world_left, block_width=2, block_height=2)
    preview_right = _block_mean_preview(world_right, block_width=2, block_height=2)
    return {
        "world_left_raster": world_left,
        "world_right_raster": world_right,
        "preview_left": preview_left,
        "preview_right": preview_right,
        "high_resolution_worlds_differ": world_left != world_right,
        "low_bandwidth_previews_are_exactly_equal": preview_left == preview_right,
    }


def _literature_and_claims(
    registry: Mapping[str, Any],
) -> tuple[
    tuple[str, ...],
    tuple[dict[str, object], ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    if registry.get("schema") != REGISTRY_SCHEMA:
        raise ValueError("unexpected N4 registry schema")
    raw_claims = registry.get("candidate_core_claims")
    if not isinstance(raw_claims, list) or not raw_claims:
        raise ValueError("candidate_core_claims must be a non-empty list")
    claims = tuple(str(item).strip() for item in raw_claims)
    if any(not claim for claim in claims) or len(set(claims)) != len(claims):
        raise ValueError("candidate core claims must be unique and non-empty")
    raw_required = registry.get("required_uncovered_core_claims")
    if not isinstance(raw_required, list) or not raw_required:
        raise ValueError("required_uncovered_core_claims must be a non-empty list")
    required_uncovered = tuple(str(item).strip() for item in raw_required)
    if (
        any(not claim for claim in required_uncovered)
        or len(set(required_uncovered)) != len(required_uncovered)
        or not set(required_uncovered).issubset(claims)
    ):
        raise ValueError("required uncovered claims must be unique registered claims")
    raw_literature = registry.get("literature_audit")
    if not isinstance(raw_literature, list) or not raw_literature:
        raise ValueError("literature_audit must be a non-empty list")
    literature: list[dict[str, object]] = []
    covered: set[str] = set()
    seen_works: set[str] = set()
    for raw in raw_literature:
        if not isinstance(raw, Mapping):
            raise ValueError("every literature entry must be an object")
        work = _required_text(raw, "work")
        if work in seen_works:
            raise ValueError(f"duplicate literature work: {work}")
        seen_works.add(work)
        overlaps_raw = raw.get("overlaps_core_claims")
        if not isinstance(overlaps_raw, list):
            raise ValueError("overlaps_core_claims must be a list")
        overlaps = tuple(str(item).strip() for item in overlaps_raw)
        if any(not item for item in overlaps):
            raise ValueError("literature overlaps must be non-empty strings")
        unknown = set(overlaps) - set(claims)
        if unknown:
            raise ValueError(f"literature lists unknown core claims: {sorted(unknown)}")
        covered.update(overlaps)
        literature.append(
            {
                "work": work,
                "source": _required_text(raw, "source"),
                "adjacent_scope": _required_text(raw, "adjacent_scope"),
                "overlaps_core_claims": overlaps,
            }
        )
    return claims, tuple(literature), tuple(sorted(covered)), required_uncovered


def audit_selector_information_boundary(
    registry: Mapping[str, Any],
) -> dict[str, object]:
    claims, literature, covered_claims, required_uncovered = _literature_and_claims(
        registry
    )
    uncovered_claims = tuple(sorted(set(claims) - set(covered_claims)))

    conflicting_worlds = (
        UtilityWorld.build(
            world_id="detail-left",
            observable_state_id="same-preview",
            probability=0.5,
            action_utilities=(0.0, 1.0, 0.0),
        ),
        UtilityWorld.build(
            world_id="detail-right",
            observable_state_id="same-preview",
            probability=0.5,
            action_utilities=(0.0, 0.0, 1.0),
        ),
    )
    conflicting = decompose_selector_regret(
        conflicting_worlds,
        policy_actions_by_observable_state={"same-preview": 1},
    )
    refined_worlds = tuple(
        UtilityWorld.build(
            world_id=world.world_id,
            observable_state_id=world.world_id,
            probability=world.probability,
            action_utilities=world.action_utilities,
        )
        for world in conflicting_worlds
    )
    refined = decompose_selector_regret(
        refined_worlds,
        policy_actions_by_observable_state={"detail-left": 1, "detail-right": 2},
    )
    shared_optimum_worlds = (
        UtilityWorld.build(
            world_id="shared-a",
            observable_state_id="shared-preview",
            probability=0.5,
            action_utilities=(0.0, 1.0, 0.2),
        ),
        UtilityWorld.build(
            world_id="shared-b",
            observable_state_id="shared-preview",
            probability=0.5,
            action_utilities=(0.0, 0.8, 0.7),
        ),
    )
    shared_optimum = decompose_selector_regret(
        shared_optimum_worlds,
        policy_actions_by_observable_state={"shared-preview": 1},
    )
    visual_alias = exact_visual_alias_fixture()
    leaky_actions = {"detail-left": 1, "detail-right": 2}
    leaky_policy_is_observable = policy_is_observable(
        conflicting_worlds, actions_by_world_id=leaky_actions
    )

    preview_arms = (
        SelectorEvaluationArm.build(
            method_id="adaptive-selector",
            information_set_id="preview-only",
            selector_visible_fields=("question", "low_resolution_preview"),
            action_bank_id="answer-now-plus-two-crops",
            utility_definition_id="task-minus-acquisition-minus-proposer",
            task_utility=0.5,
            acquisition_cost=0.0,
            proposer_cost=0.0,
        ),
        SelectorEvaluationArm.build(
            method_id="conservative-selector",
            information_set_id="preview-only",
            selector_visible_fields=("question", "low_resolution_preview"),
            action_bank_id="answer-now-plus-two-crops",
            utility_definition_id="task-minus-acquisition-minus-proposer",
            task_utility=0.6,
            acquisition_cost=0.0,
            proposer_cost=0.0,
        ),
    )
    full_information_arms = tuple(
        SelectorEvaluationArm.build(
            method_id=method_id,
            information_set_id="full-resolution-selector-input",
            selector_visible_fields=("question", "full_resolution_image"),
            action_bank_id="answer-now-plus-two-crops",
            utility_definition_id="task-minus-acquisition-minus-proposer",
            task_utility=task_utility,
            acquisition_cost=0.0,
            proposer_cost=0.0,
        )
        for method_id, task_utility in (
            ("adaptive-selector", 1.0),
            ("conservative-selector", 0.6),
        )
    )
    preview_values = {arm.method_id: arm.cost_adjusted_utility for arm in preview_arms}
    full_information_values = {
        arm.method_id: arm.cost_adjusted_utility for arm in full_information_arms
    }
    reversals = pairwise_rank_reversals(preview_values, full_information_values)

    cost_fixture = (
        SelectorEvaluationArm.build(
            method_id="costly-proposer",
            information_set_id="preview-only",
            selector_visible_fields=("question", "low_resolution_preview"),
            action_bank_id="answer-now-plus-two-crops",
            utility_definition_id="task-minus-acquisition-minus-proposer",
            task_utility=0.70,
            acquisition_cost=0.05,
            proposer_cost=0.06,
        ),
        SelectorEvaluationArm.build(
            method_id="cheap-baseline",
            information_set_id="preview-only",
            selector_visible_fields=("question", "low_resolution_preview"),
            action_bank_id="answer-now-plus-two-crops",
            utility_definition_id="task-minus-acquisition-minus-proposer",
            task_utility=0.65,
            acquisition_cost=0.0,
            proposer_cost=0.0,
        ),
    )
    cost_adjusted_ranking = rank_matched_selector_arms(cost_fixture)
    mismatched_visibility_is_rejected = False
    try:
        rank_matched_selector_arms((preview_arms[0], full_information_arms[1]))
    except ValueError:
        mismatched_visibility_is_rejected = True

    checks = {
        "regret_decomposition_is_exact": abs(conflicting.additive_residual) < 1e-12,
        "aliasing_and_policy_regret_are_nonnegative": (
            conflicting.aliasing_regret >= 0.0
            and conflicting.policy_estimation_regret >= 0.0
        ),
        "conflicting_alias_has_positive_irreducible_regret": math.isclose(
            conflicting.aliasing_regret, 0.5, rel_tol=0.0, abs_tol=1e-12
        ),
        "refining_observation_removes_aliasing_regret": math.isclose(
            refined.aliasing_regret, 0.0, rel_tol=0.0, abs_tol=1e-12
        ),
        "shared_optimum_alias_has_zero_aliasing_regret": math.isclose(
            shared_optimum.aliasing_regret, 0.0, rel_tol=0.0, abs_tol=1e-12
        ),
        "world_conditioned_oracle_violates_preview_information_boundary": (
            not leaky_policy_is_observable
        ),
        "distinct_high_resolution_worlds_share_exact_preview": bool(
            visual_alias["high_resolution_worlds_differ"]
            and visual_alias["low_bandwidth_previews_are_exactly_equal"]
        ),
        "required_evaluation_core_survives_collision_screen": set(
            required_uncovered
        ).issubset(uncovered_claims),
        "candidate_does_not_claim_aliasing_or_voi_as_new": bool(
            registry.get("treats_classical_aliasing_and_voi_as_prior_art") is True
        ),
        "matched_preview_ranking_favors_conservative_selector": (
            rank_matched_selector_arms(preview_arms)[0] == "conservative-selector"
        ),
        "matched_full_information_ranking_favors_adaptive_selector": (
            rank_matched_selector_arms(full_information_arms)[0] == "adaptive-selector"
        ),
        "cross_information_set_rank_reversal_is_detected": reversals
        == (("adaptive-selector", "conservative-selector"),),
        "acquisition_and_proposer_costs_both_change_ranking": (
            cost_adjusted_ranking[0] == "cheap-baseline"
            and cost_fixture[0].task_utility > cost_fixture[1].task_utility
        ),
        "mismatched_visibility_comparison_is_rejected": (
            mismatched_visibility_is_rejected
        ),
    }
    formal_gate_passed = all(checks.values())
    decision = (
        "n4_information_boundary_candidate_survives_formal_gate"
        if formal_gate_passed
        else "n4_information_boundary_candidate_rejected"
    )
    return {
        "schema": AUDIT_SCHEMA,
        "audited_at": _required_text(registry, "audited_at"),
        "candidate_name": _required_text(registry, "candidate_name"),
        "candidate_core_claims": claims,
        "covered_core_claims": covered_claims,
        "uncovered_core_claims": uncovered_claims,
        "required_uncovered_core_claims": required_uncovered,
        "literature_audit": literature,
        "conflicting_alias_decomposition": asdict(conflicting),
        "refined_observation_decomposition": asdict(refined),
        "shared_optimum_decomposition": asdict(shared_optimum),
        "exact_visual_alias_fixture": visual_alias,
        "leaky_world_policy_actions": leaky_actions,
        "leaky_world_policy_is_preview_observable": leaky_policy_is_observable,
        "toy_rank_reversal": {
            "preview_only_arms": tuple(asdict(arm) for arm in preview_arms),
            "full_information_arms": tuple(
                asdict(arm) for arm in full_information_arms
            ),
            "preview_only_ranking": rank_matched_selector_arms(preview_arms),
            "full_information_ranking": rank_matched_selector_arms(
                full_information_arms
            ),
            "pairwise_reversals": reversals,
        },
        "toy_complete_cost_ledger": {
            "arms": tuple(asdict(arm) for arm in cost_fixture),
            "cost_adjusted_ranking": cost_adjusted_ranking,
        },
        "checks": checks,
        "formal_gate_passed": formal_gate_passed,
        "decision": decision,
        "interpretation": (
            "Formal and novelty-screen evidence only; advancement requires a "
            "pre-registered real-data rank-reversal test under matched selector "
            "visibility and complete cost accounting."
        ),
        "opened_existing_outcomes": 0,
        "authorized_new_gpu_jobs": 0,
        "authorized_new_checkpoints": 0,
    }
