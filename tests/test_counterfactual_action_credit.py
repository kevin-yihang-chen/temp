from __future__ import annotations

import copy

import pytest

from beyond_entropy.counterfactual_action_credit import (
    CounterfactualActionPair,
    CounterfactualArmOutcome,
    TokenRoleMasks,
    TokenSpan,
    build_token_role_masks,
    compose_token_local_advantages,
    cyclically_derange_action_credits,
)


def _arm(
    branch_id: str,
    *,
    score: float,
    cost: float,
    observation: str,
    seed: int = 17,
    prefix: str = "a",
    target: str = "c",
) -> CounterfactualArmOutcome:
    return CounterfactualArmOutcome(
        branch_id=branch_id,
        prefix_sha256=prefix * 64,
        action_sha256="b" * 64,
        observation_sha256=observation * 64,
        target_sha256=target * 64,
        policy_sha256="d" * 64,
        decoding_sha256="e" * 64,
        scorer_sha256="f" * 64,
        continuation_seed=seed,
        task_score=score,
        action_cost=cost,
    )


def _pair(
    factual_score: float = 1.0,
    counterfactual_score: float = 0.0,
    *,
    factual_cost: float = 1.0,
    counterfactual_cost: float = 0.0,
) -> CounterfactualActionPair:
    return CounterfactualActionPair(
        trajectory_id="trajectory-0",
        factual=_arm(
            "factual", score=factual_score, cost=factual_cost, observation="1"
        ),
        counterfactual=_arm(
            "counterfactual",
            score=counterfactual_score,
            cost=counterfactual_cost,
            observation="2",
        ),
        lambda_cost=0.05,
    )


def test_tool_trajectory_masks_partition_valid_response_and_keep_observation_frozen() -> (
    None
):
    masks = build_token_role_masks(
        response_length=12,
        valid_response_length=10,
        action_spans=(TokenSpan(0, 3),),
        observation_spans=(TokenSpan(3, 6),),
        answer_spans=(TokenSpan(6, 10),),
    )
    assert masks.action == (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    assert masks.observation == (0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0)
    assert masks.answer == (0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0)
    assert masks.policy == (1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0)
    assert masks.valid_response_length == 10
    assert TokenRoleMasks.from_dict(masks.to_dict()) == masks


def test_no_tool_trajectory_assigns_all_valid_tokens_to_answer() -> None:
    masks = build_token_role_masks(
        response_length=6,
        valid_response_length=4,
        answer_spans=(TokenSpan(0, 4),),
    )
    assert masks.action == masks.observation == (0, 0, 0, 0, 0, 0)
    assert masks.answer == masks.policy == (1, 1, 1, 1, 0, 0)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (
            {
                "response_length": 6,
                "valid_response_length": 5,
                "action_spans": (TokenSpan(0, 2),),
                "observation_spans": (TokenSpan(2, 4),),
                "answer_spans": (TokenSpan(3, 5),),
            },
            "overlap",
        ),
        (
            {
                "response_length": 6,
                "valid_response_length": 5,
                "action_spans": (TokenSpan(0, 1),),
                "observation_spans": (TokenSpan(2, 3),),
                "answer_spans": (TokenSpan(3, 5),),
            },
            "gap",
        ),
        (
            {
                "response_length": 6,
                "valid_response_length": 5,
                "action_spans": (TokenSpan(0, 2),),
                "answer_spans": (TokenSpan(2, 5),),
            },
            "both be present",
        ),
        (
            {
                "response_length": 6,
                "valid_response_length": 5,
                "answer_spans": (TokenSpan(0, 6),),
            },
            "beyond",
        ),
    ],
)
def test_mask_builder_rejects_ambiguous_or_leaking_roles(
    kwargs: dict, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        build_token_role_masks(**kwargs)


def test_masks_reject_non_suffix_padding_and_non_integer_spans() -> None:
    with pytest.raises(ValueError, match="contiguous response suffix"):
        TokenRoleMasks(
            action=(0, 0, 0),
            answer=(1, 0, 1),
            observation=(0, 0, 0),
            padding=(0, 1, 0),
        )
    with pytest.raises(ValueError, match="integers"):
        TokenSpan(0.0, 2)  # type: ignore[arg-type]


def test_pair_credit_includes_arm_specific_cost_and_arm_swap_is_antisymmetric() -> None:
    pair = _pair()
    assert pair.raw_score_effect == pytest.approx(1.0)
    assert pair.action_credit == pytest.approx(0.95)
    swapped = pair.swapped()
    assert swapped.raw_score_effect == pytest.approx(-pair.raw_score_effect)
    assert swapped.action_credit == pytest.approx(-pair.action_credit)


@pytest.mark.parametrize(
    "factual,counterfactual,expected",
    [(1.0, 0.0, 0.95), (1.0, 1.0, -0.05), (0.0, 0.0, -0.05), (0.0, 1.0, -1.05)],
)
def test_primary_binary_pair_credit_preserves_rescue_neutral_and_harm(
    factual: float, counterfactual: float, expected: float
) -> None:
    assert _pair(factual, counterfactual).action_credit == pytest.approx(expected)


def test_pair_rejects_provenance_mismatch() -> None:
    with pytest.raises(ValueError, match="continuation_seed"):
        CounterfactualActionPair(
            trajectory_id="trajectory-0",
            factual=_arm("factual", score=1.0, cost=1.0, observation="1", seed=3),
            counterfactual=_arm(
                "counterfactual", score=0.0, cost=0.0, observation="2", seed=4
            ),
        )
    with pytest.raises(ValueError, match="prefix_sha256"):
        CounterfactualActionPair(
            trajectory_id="trajectory-0",
            factual=_arm("factual", score=1.0, cost=1.0, observation="1", prefix="a"),
            counterfactual=_arm(
                "counterfactual",
                score=0.0,
                cost=0.0,
                observation="2",
                prefix="c",
            ),
        )

    with pytest.raises(ValueError, match="target_sha256"):
        CounterfactualActionPair(
            trajectory_id="trajectory-0",
            factual=_arm("factual", score=1.0, cost=1.0, observation="1"),
            counterfactual=_arm(
                "counterfactual",
                score=0.0,
                cost=0.0,
                observation="2",
                target="0",
            ),
        )


def test_pair_serialization_roundtrip_detects_derived_value_tampering() -> None:
    pair = _pair()
    payload = pair.to_dict()
    assert CounterfactualActionPair.from_dict(payload) == pair
    tampered = copy.deepcopy(payload)
    tampered["action_credit"] = 0.0
    with pytest.raises(ValueError, match="action_credit"):
        CounterfactualActionPair.from_dict(tampered)


def test_token_local_advantage_separates_action_answer_and_environment() -> None:
    masks = build_token_role_masks(
        response_length=8,
        valid_response_length=7,
        action_spans=(TokenSpan(0, 2),),
        observation_spans=(TokenSpan(2, 4),),
        answer_spans=(TokenSpan(4, 7),),
    )
    advantages = compose_token_local_advantages(
        outcome_advantage=-0.25,
        action_credit=0.95,
        masks=masks,
        beta=1.0,
    )
    assert advantages == pytest.approx((0.95, 0.95, 0.0, 0.0, -0.25, -0.25, -0.25, 0.0))


def test_no_tool_trajectory_rejects_nonzero_action_credit() -> None:
    masks = build_token_role_masks(
        response_length=4,
        valid_response_length=3,
        answer_spans=(TokenSpan(0, 3),),
    )
    with pytest.raises(ValueError, match="without action tokens"):
        compose_token_local_advantages(
            outcome_advantage=1.0,
            action_credit=0.1,
            masks=masks,
        )


def test_deterministic_shuffled_credit_has_no_self_donor_and_preserves_multiset() -> (
    None
):
    assignments = cyclically_derange_action_credits(
        ["trajectory-c", "trajectory-a", "trajectory-b"],
        [0.95, -1.05, -0.05],
    )
    assert [item.target_trajectory_id for item in assignments] == [
        "trajectory-c",
        "trajectory-a",
        "trajectory-b",
    ]
    assert all(
        item.target_trajectory_id != item.donor_trajectory_id for item in assignments
    )
    assert sorted(item.action_credit for item in assignments) == pytest.approx(
        sorted([0.95, -1.05, -0.05])
    )
    with pytest.raises(ValueError, match="at least two"):
        cyclically_derange_action_credits(["only"], [0.95])
    with pytest.raises(ValueError, match="unique"):
        cyclically_derange_action_credits(["same", "same"], [0.95, -0.05])
