from dataclasses import replace

import pytest

from beyond_entropy.decoding_sensitivity import capped_state_ids, generated_token_count
from beyond_entropy.simulate import simulate_counterfactual_dataset


def test_capped_states_use_the_executed_action_backend_metadata():
    records = simulate_counterfactual_dataset(
        n_states=2,
        num_candidates=2,
        questions_per_image=1,
        seed=2,
    )
    decorated = []
    for index, record in enumerate(records):
        backend_name = "baseline_backend" if record.action_type == "ANSWER" else "action_backend"
        decorated.append(
            replace(
                record,
                metadata={backend_name: {"generated_tokens": 16 if index == 2 else 3}},
            )
        )
    assert generated_token_count(decorated[0]) == 3
    assert capped_state_ids(decorated, token_cap=16) == {decorated[2].state_id}


def test_generated_token_count_rejects_missing_metadata():
    record = simulate_counterfactual_dataset(n_states=2, seed=3)[0]
    with pytest.raises(ValueError, match="missing"):
        generated_token_count(record)
