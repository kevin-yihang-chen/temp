from __future__ import annotations

import pytest

from beyond_entropy.selector_information_boundary import (
    AUDIT_SCHEMA,
    SelectorEvaluationArm,
    UtilityWorld,
    audit_selector_information_boundary,
    decompose_selector_regret,
    exact_visual_alias_fixture,
    pairwise_rank_reversals,
    policy_is_observable,
    rank_matched_selector_arms,
)


def _registry() -> dict[str, object]:
    return {
        "schema": "n4_selector_information_boundary_registry_v1",
        "audited_at": "2026-09-03T15:19:34+08:00",
        "candidate_name": "test candidate",
        "candidate_core_claims": [
            "selector_input_information_ledger",
            "aliasing_vs_policy_regret_decomposition",
            "matched_visibility_method_comparison",
            "joint_acquisition_and_proposer_cost_ledger",
        ],
        "required_uncovered_core_claims": [
            "selector_input_information_ledger",
            "matched_visibility_method_comparison",
        ],
        "literature_audit": [
            {
                "work": "cost benchmark",
                "source": "https://example.test/cost",
                "adjacent_scope": "cost accounting",
                "overlaps_core_claims": ["joint_acquisition_and_proposer_cost_ledger"],
            }
        ],
        "treats_classical_aliasing_and_voi_as_prior_art": True,
    }


def _conflicting_worlds() -> tuple[UtilityWorld, ...]:
    return (
        UtilityWorld.build(
            world_id="left",
            observable_state_id="alias",
            probability=0.5,
            action_utilities=(0.0, 1.0, 0.0),
        ),
        UtilityWorld.build(
            world_id="right",
            observable_state_id="alias",
            probability=0.5,
            action_utilities=(0.0, 0.0, 1.0),
        ),
    )


def test_conflicting_exact_alias_has_irreducible_regret() -> None:
    result = decompose_selector_regret(
        _conflicting_worlds(), policy_actions_by_observable_state={"alias": 1}
    )
    assert result.full_information_value == pytest.approx(1.0)
    assert result.observable_bayes_value == pytest.approx(0.5)
    assert result.policy_value == pytest.approx(0.5)
    assert result.aliasing_regret == pytest.approx(0.5)
    assert result.policy_estimation_regret == pytest.approx(0.0)
    assert result.total_regret == pytest.approx(
        result.aliasing_regret + result.policy_estimation_regret
    )
    assert result.additive_residual == pytest.approx(0.0)


def test_suboptimal_observable_policy_adds_estimation_regret() -> None:
    result = decompose_selector_regret(
        _conflicting_worlds(), policy_actions_by_observable_state={"alias": 0}
    )
    assert result.aliasing_regret == pytest.approx(0.5)
    assert result.policy_estimation_regret == pytest.approx(0.5)
    assert result.total_regret == pytest.approx(1.0)


def test_world_specific_oracle_is_not_preview_observable() -> None:
    assert not policy_is_observable(
        _conflicting_worlds(), actions_by_world_id={"left": 1, "right": 2}
    )
    assert policy_is_observable(
        _conflicting_worlds(), actions_by_world_id={"left": 1, "right": 1}
    )


def test_exact_visual_alias_fixture_is_really_aliased() -> None:
    fixture = exact_visual_alias_fixture()
    assert fixture["high_resolution_worlds_differ"]
    assert fixture["low_bandwidth_previews_are_exactly_equal"]
    assert fixture["preview_left"] == ((1.0, 1.0),)
    assert fixture["preview_right"] == ((1.0, 1.0),)


def _arm(
    method_id: str,
    *,
    information_set_id: str = "preview",
    task_utility: float = 0.5,
    acquisition_cost: float = 0.0,
    proposer_cost: float = 0.0,
) -> SelectorEvaluationArm:
    return SelectorEvaluationArm.build(
        method_id=method_id,
        information_set_id=information_set_id,
        selector_visible_fields=("question", "preview"),
        action_bank_id="three-actions",
        utility_definition_id="net-utility",
        task_utility=task_utility,
        acquisition_cost=acquisition_cost,
        proposer_cost=proposer_cost,
    )


def test_matched_ranking_charges_acquisition_and_proposer_costs() -> None:
    ranking = rank_matched_selector_arms(
        (
            _arm(
                "raw-winner",
                task_utility=0.70,
                acquisition_cost=0.05,
                proposer_cost=0.06,
            ),
            _arm("net-winner", task_utility=0.65),
        )
    )
    assert ranking == ("net-winner", "raw-winner")


def test_matched_ranking_rejects_different_information_sets() -> None:
    with pytest.raises(ValueError, match="mismatched information boundary"):
        rank_matched_selector_arms(
            (
                _arm("preview-method", information_set_id="preview"),
                _arm("leaky-method", information_set_id="full-resolution"),
            )
        )


def test_pairwise_rank_reversal_is_strict_and_ties_do_not_count() -> None:
    assert pairwise_rank_reversals(
        {"adaptive": 0.5, "conservative": 0.6},
        {"adaptive": 1.0, "conservative": 0.6},
    ) == (("adaptive", "conservative"),)
    assert not pairwise_rank_reversals(
        {"adaptive": 0.6, "conservative": 0.6},
        {"adaptive": 1.0, "conservative": 0.6},
    )


def test_audit_survives_only_as_a_real_test_candidate() -> None:
    report = audit_selector_information_boundary(_registry())
    assert report["schema"] == AUDIT_SCHEMA
    assert report["decision"] == (
        "n4_information_boundary_candidate_survives_formal_gate"
    )
    assert report["formal_gate_passed"]
    assert report["opened_existing_outcomes"] == 0
    assert report["authorized_new_gpu_jobs"] == 0
    assert report["authorized_new_checkpoints"] == 0


@pytest.mark.parametrize(
    "worlds,policy,match",
    [
        ((), {}, "non-empty"),
        (
            (
                UtilityWorld.build(
                    world_id="a",
                    observable_state_id="x",
                    probability=0.4,
                    action_utilities=(0.0, 1.0),
                ),
                UtilityWorld.build(
                    world_id="b",
                    observable_state_id="x",
                    probability=0.5,
                    action_utilities=(1.0, 0.0),
                ),
            ),
            {"x": 0},
            "sum to one",
        ),
        (
            (
                UtilityWorld.build(
                    world_id="a",
                    observable_state_id="x",
                    probability=1.0,
                    action_utilities=(0.0, 1.0),
                ),
            ),
            {"x": 2},
            "outside",
        ),
    ],
)
def test_decomposition_rejects_invalid_contracts(
    worlds: tuple[UtilityWorld, ...], policy: dict[str, int], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        decompose_selector_regret(worlds, policy_actions_by_observable_state=policy)


def test_registry_rejects_unknown_overlap_claim() -> None:
    registry = _registry()
    literature = registry["literature_audit"]
    assert isinstance(literature, list)
    literature[0]["overlaps_core_claims"] = ["not-registered"]
    with pytest.raises(ValueError, match="unknown core claims"):
        audit_selector_information_boundary(registry)
