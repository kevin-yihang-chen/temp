import json

import pytest

torch = pytest.importorskip("torch")

from beyond_entropy.schema import BBox
from beyond_entropy.sequential_post_training import (
    SequentialPolicyInput,
    SequentialTrainingExample,
    deterministic_joint_schedule,
    factorized_potential_outcome_probabilities,
    factorized_potential_outcome_targets,
    load_sequential_training_examples,
    sequential_post_training_loss,
    state_hash_subset,
)
from beyond_entropy.sequential_schema import AcquiredObservationSpec, SequentialRolloutRecord


def policy_input(state_id="s1"):
    return SequentialPolicyInput.from_untrusted_mapping({
        "state_id": state_id, "image_id": "i1", "source_id": f"source-{state_id}",
        "image_path": "/tmp/image.png", "question": "question?",
        "model_prompt": "question? Answer briefly.",
        "acquired_observations": [{
            "action_id": "crop-a", "bbox": [0, 0, .5, .5], "visual_cost": 1,
        }],
        "proposed_action_id": "crop-b", "proposed_bbox": [.5, .5, 1, 1],
        "proposed_visual_cost": 1,
    })


def example(state_id="s1", stop=0, continued=1):
    return SequentialTrainingExample(policy_input(state_id), stop, continued, "replicate-000")


def test_policy_input_strictly_rejects_outcome_leakage():
    value = dict(policy_input().__dict__)
    value["stop_correct"] = 1.0
    with pytest.raises(ValueError, match="strict pre-action allowlist"):
        SequentialPolicyInput.from_untrusted_mapping(value)
    assert len(policy_input().geometry()) == 15


def test_outcome_only_and_counterfactual_losses_have_distinct_neutral_semantics():
    logits = torch.tensor([[2.0, -1.0]], requires_grad=True)
    neutral = example(stop=1, continued=1)
    outcome = sequential_post_training_loss(logits, neutral, method="outcome_only")
    counterfactual = sequential_post_training_loss(
        logits, neutral, method="counterfactual_utility"
    )
    assert outcome.item() > 0
    assert counterfactual.item() == 0
    rescue = example(stop=0, continued=1)
    assert sequential_post_training_loss(
        logits, rescue, method="counterfactual_utility"
    ).item() > 0


@pytest.mark.parametrize(
    ("stop", "continued", "risk_target", "conditional_index", "conditional_target"),
    (
        (0, 0, 1, 1, 0),
        (0, 1, 1, 1, 1),
        (1, 0, 0, 2, 1),
        (1, 1, 0, 2, 0),
    ),
)
def test_factorized_loss_uses_dense_risk_and_observable_conditional(
    stop, continued, risk_target, conditional_index, conditional_target
):
    logits = torch.zeros((1, 3), requires_grad=True)
    loss = sequential_post_training_loss(
        logits, example(stop=stop, continued=continued),
        method="factorized_potential_outcomes",
    )
    loss.backward()
    assert loss.item() == pytest.approx(0.69314718)
    gradient = logits.grad[0]
    assert gradient[0].sign().item() == (1 if risk_target == 0 else -1)
    assert gradient[conditional_index].sign().item() == (
        1 if conditional_target == 0 else -1
    )
    unused = 3 - conditional_index
    assert gradient[unused].item() == 0


def test_factorized_gain_is_rescue_mass_minus_harm_mass():
    logits = torch.logit(torch.tensor([[.8, .5, .1]]))
    result = factorized_potential_outcome_probabilities(logits)
    assert result["expected_gain"].item() == pytest.approx(.8 * .5 - .2 * .1)
    assert result["expected_gain"].item() == pytest.approx(.38)


@pytest.mark.parametrize(
    ("stop", "continued", "rescue", "harm"),
    (
        (.6, .8, .5, 0),
        (.6, .3, 0, .5),
        (.25, .25, 0, 0),
        (0, .4, .4, 0),
        (1, .4, 0, .6),
    ),
)
def test_factorized_targets_exactly_reconstruct_soft_reward_gain(
    stop, continued, rescue, harm
):
    item = example(stop=stop, continued=continued)
    targets = factorized_potential_outcome_targets(item)
    assert targets["rescue_fraction"] == pytest.approx(rescue)
    assert targets["harm_fraction"] == pytest.approx(harm)
    reconstructed = (
        targets["error_mass"] * targets["rescue_fraction"]
        - targets["correct_mass"] * targets["harm_fraction"]
    )
    assert reconstructed == pytest.approx(item.gain)


def test_factorized_soft_reward_loss_uses_both_weighted_conditionals():
    logits = torch.zeros((1, 3), requires_grad=True)
    loss = sequential_post_training_loss(
        logits, example(stop=.6, continued=.8),
        method="factorized_potential_outcomes",
    )
    loss.backward()
    assert loss.item() == pytest.approx(0.69314718)
    assert logits.grad[0, 0].item() > 0  # target error mass is .4
    assert logits.grad[0, 1].item() == pytest.approx(0)  # rescue target is .5
    assert logits.grad[0, 2].item() > 0  # zero harm with .6 observed mass


def test_factorized_loss_rejects_binary_action_logits():
    with pytest.raises(ValueError, match="factorized logits"):
        sequential_post_training_loss(
            torch.zeros((1, 2)), example(),
            method="factorized_potential_outcomes",
        )


def test_state_subset_and_joint_schedule_are_outcome_independent():
    left = [example(f"s{i}", stop=i % 2, continued=(i + 1) % 2) for i in range(8)]
    right = [example(f"s{i}", stop=0, continued=0) for i in range(8)]
    left_ids = [item.inputs.state_id for item in state_hash_subset(
        left, maximum_states=3, seed=17, namespace="test"
    )]
    right_ids = [item.inputs.state_id for item in state_hash_subset(
        right, maximum_states=3, seed=17, namespace="test"
    )]
    assert left_ids == right_ids
    schedule = deterministic_joint_schedule(
        {"chartqa": left, "docvqa": right}, draws=6, seed=17, namespace="test"
    )
    assert [domain for domain, _ in schedule] == [
        "chartqa", "docvqa", "chartqa", "docvqa", "chartqa", "docvqa"
    ]


def test_loader_joins_manifest_prompt_but_keeps_labels_outside_input(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"not-opened-by-loader")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({
        "state_id": "s1", "image_id": "i1", "source_id": "source-s1",
        "image_path": "image.png", "question": "question?",
        "model_prompt": "question? Answer briefly.",
    }) + "\n")
    record = SequentialRolloutRecord(
        state_id="s1", image_id="i1", source_id="source-s1", question="question?",
        original_image=str(image), step_index=1,
        acquired_observations=(AcquiredObservationSpec(
            "crop-a", BBox(0, 0, .5, .5), 1
        ),),
        proposed_action_id="crop-b", proposed_bbox=BBox(.5, .5, 1, 1),
        proposed_visual_cost=1, replicate_id="replicate-000", generation_seed=0,
        stop_answer="no", stop_correct=0, stop_entropy=.5,
        stop_max_probability=.6, stop_top1_top2_margin=.2,
        continue_answer="yes", continue_correct=1, continue_entropy=.2,
        continue_max_probability=.8, continue_top1_top2_margin=.6,
    )
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text(json.dumps(record.to_dict()) + "\n")
    loaded = load_sequential_training_examples(rollouts, manifest)
    assert loaded[0].gain == 1
    assert loaded[0].inputs.model_prompt == "question? Answer briefly."
    assert "correct" not in repr(loaded[0].inputs)
