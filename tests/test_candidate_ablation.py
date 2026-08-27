from dataclasses import replace

import pytest

from beyond_entropy.candidate_ablation import (
    build_candidate_ablation_markdown,
    compare_candidate_sets,
)
from beyond_entropy.schema import BBox
from beyond_entropy.simulate import simulate_counterfactual_dataset


def test_candidate_ablation_uses_matched_state_bootstrap():
    left = simulate_counterfactual_dataset(
        n_states=10,
        num_candidates=4,
        questions_per_image=1,
        seed=31,
    )
    right = list(left)
    for answer in [record for record in left if record.action_type == "ANSWER"]:
        exemplar = next(
            record
            for record in left
            if record.state_id == answer.state_id and record.action_type == "ZOOM"
        )
        right.append(
            replace(
                exemplar,
                action_id="extra-candidate",
                candidate_bbox=BBox(0.25, 0.25, 0.75, 0.75),
                correct_after=1.0,
            )
        )

    first = compare_candidate_sets(left, right, bootstrap_resamples=40, seed=7)
    second = compare_candidate_sets(left, right, bootstrap_resamples=40, seed=7)
    assert first == second
    assert first["left_candidates_per_decision"] == 4
    assert first["right_candidates_per_decision"] == 5
    entropy = next(
        result
        for result in first["policy_differences"]
        if result["right_policy"] == "entropy_search"
    )
    assert entropy["resampling_unit"] == "state_id"
    assert entropy["metrics"]["avg_tool_calls"]["estimate"] == 1.0
    assert "Right minus left" in build_candidate_ablation_markdown(first)


def test_candidate_ablation_rejects_unmatched_baseline():
    left = simulate_counterfactual_dataset(n_states=4, num_candidates=2, seed=9)
    right = [
        replace(record, answer_before="different")
        if record.state_id == left[0].state_id
        else record
        for record in left
    ]
    with pytest.raises(ValueError, match="answer_before"):
        compare_candidate_sets(left, right, bootstrap_resamples=5)


def test_candidate_ablation_allows_same_hashed_image_in_different_directories():
    left = simulate_counterfactual_dataset(n_states=4, num_candidates=2, seed=12)
    right = [
        replace(record, original_image=f"/another/root/{record.original_image.rsplit('/', 1)[-1]}")
        for record in left
    ]
    report = compare_candidate_sets(left, right, bootstrap_resamples=5)
    assert report["n_decisions"] == 4


def test_candidate_ablation_rejects_different_image_filename():
    left = simulate_counterfactual_dataset(n_states=4, num_candidates=2, seed=13)
    right = [
        replace(record, original_image="/another/root/different.png")
        if record.state_id == left[0].state_id
        else record
        for record in left
    ]
    with pytest.raises(ValueError, match="image filename"):
        compare_candidate_sets(left, right, bootstrap_resamples=5)
