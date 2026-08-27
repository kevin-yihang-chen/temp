from beyond_entropy.dataset import split_by_state
from beyond_entropy.metrics import entropy_diagnostic, evaluate_policy
from beyond_entropy.model import LinearValueModel
from beyond_entropy.policies import EntropySearchPolicy, LearnedVOIPolicy
from beyond_entropy.simulate import simulate_counterfactual_dataset


def test_end_to_end_value_pipeline_is_deterministic(tmp_path):
    records = simulate_counterfactual_dataset(n_states=300, num_candidates=4, seed=7)
    train, test = split_by_state(records, train_fraction=0.7, seed=7)
    model = LinearValueModel.fit(train, lambda_cost=0.05, alpha=1.0)
    model_path = tmp_path / "model.json"
    model.save(model_path)
    loaded = LinearValueModel.load(model_path)

    diagnostic = entropy_diagnostic(test)
    entropy_result = evaluate_policy(
        test, EntropySearchPolicy(), lambda_cost=0.05
    )
    learned_result = evaluate_policy(
        test, LearnedVOIPolicy(loaded), lambda_cost=0.05
    )

    assert diagnostic.spurious_confidence_gain_rate > 0.0
    assert entropy_result["avg_tool_calls"] == 4.0
    assert learned_result["avg_tool_calls"] <= 1.0
    assert learned_result["mean_oracle_regret"] < entropy_result["mean_oracle_regret"]
