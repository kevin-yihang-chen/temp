from scripts.run_utility_sft_ablation import controlled_inputs, source_derangement
from tests.test_utility_sft import make_samples


def test_source_shuffle_is_deterministic_and_changes_source():
    samples = [
        make_samples(
            state_id=f"s{index}", source_id=source, image_id=f"i{index}",
            image_path=f"/tmp/i{index}.png", rgb_hash=f"{index+1:064x}",
        )[0]
        for index, source in enumerate(("a", "a", "b", "c"))
    ]
    donors = source_derangement(samples)
    assert donors == source_derangement(samples[::-1])
    assert all(source != donor.inputs.state.source_id for source, donor in donors.items())
    original = samples[0]
    donor = donors[original.inputs.state.source_id]
    question, question_ablation = controlled_inputs(original, donor, "question_shuffle")
    image, image_ablation = controlled_inputs(original, donor, "image_shuffle")
    region, region_ablation = controlled_inputs(original, donor, "region_ablation")
    assert question.state.question == donor.inputs.state.question
    assert question.state.image_path == original.inputs.state.image_path
    assert image.state.image_path == donor.inputs.state.image_path
    assert image.state.question == original.inputs.state.question
    assert not question_ablation and not image_ablation
    assert region == original.inputs and region_ablation
