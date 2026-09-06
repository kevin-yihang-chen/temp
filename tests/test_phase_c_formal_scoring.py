import pytest

pytest.importorskip("torch")

from beyond_entropy.phase_c_formal_scoring import ablated_policy_inputs
from beyond_entropy.schema import BBox
from beyond_entropy.sequential_post_training import (
    SequentialPolicyInput,
    SequentialTrainingExample,
)
from beyond_entropy.sequential_schema import AcquiredObservationSpec


def _example(index: int) -> SequentialTrainingExample:
    inputs = SequentialPolicyInput(
        state_id=f"s{index}", image_id=f"i{index}", source_id=f"source-{index}",
        image_path=f"/tmp/{index}.png", question=f"question-{index}",
        model_prompt=f"prompt-{index}",
        acquired_observations=(AcquiredObservationSpec(
            "crop-a", BBox(0, 0, .5, .5), 1,
        ),),
        proposed_action_id=f"crop-{index}", proposed_bbox=BBox(.5, .5, 1, 1),
        proposed_visual_cost=1,
    )
    return SequentialTrainingExample(inputs, 0, 1, "replicate-000")


def test_semantic_derangements_change_only_the_registered_input_component() -> None:
    examples = [_example(index) for index in range(4)]
    question = ablated_policy_inputs(
        examples, mode="question_shuffle", seed=17, namespace="test-question",
    )
    image = ablated_policy_inputs(
        examples, mode="image_shuffle", seed=17, namespace="test-image",
    )
    region = ablated_policy_inputs(
        examples, mode="region_shuffle", seed=17, namespace="test-region",
    )
    for item in examples:
        key = item.decision_id
        assert question[key].question != item.inputs.question
        assert question[key].image_path == item.inputs.image_path
        assert image[key].image_path != item.inputs.image_path
        assert image[key].question == item.inputs.question
        assert region[key].proposed_action_id != item.inputs.proposed_action_id
        assert region[key].question == item.inputs.question
