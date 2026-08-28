from beyond_entropy.action_bank import summarize_action_bank
from beyond_entropy.simulate import simulate_counterfactual_dataset


def test_action_bank_summary_exposes_random_and_oracle_headroom():
    records = simulate_counterfactual_dataset(
        n_states=120,
        num_candidates=4,
        questions_per_image=2,
        seed=41,
    )
    report = summarize_action_bank(
        records,
        bootstrap_resamples=20,
    )
    assert report["decisions"] == 120
    assert report["candidate_counts"] == [4]
    assert 0.0 <= report["helpful_state_rate"] <= 1.0
    assert report["policies"]["oracle_voi"]["mean_policy_utility"] >= 0.0
    assert report["policies"]["answer_now"]["tool_use_rate"] == 0.0
