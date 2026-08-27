import pytest

from beyond_entropy.cli import main
from beyond_entropy.dataset import write_jsonl
from beyond_entropy.dataset import split_by_group
from beyond_entropy.metrics import (
    bootstrap_entropy_diagnostic,
    bootstrap_policy_evaluation,
    entropy_diagnostic,
    evaluate_policy,
)
from beyond_entropy.model import LinearGainModel
from beyond_entropy.policies import (
    EntropySearchPolicy,
    EntropyRandomZoomPolicy,
    EntropyThresholdPolicy,
    LearnedVOIPolicy,
    tune_entropy_thresholds,
    tune_entropy_single_crop_thresholds,
)
from beyond_entropy.simulate import simulate_counterfactual_dataset


def test_end_to_end_gain_pipeline_is_cost_correct_and_deterministic(tmp_path):
    records = simulate_counterfactual_dataset(
        n_states=300,
        num_candidates=4,
        questions_per_image=2,
        seed=7,
    )
    train, test = split_by_group(
        records,
        group="image_id",
        train_fraction=0.7,
        seed=7,
    )
    model = LinearGainModel.fit(train, alpha=1.0)
    model_path = tmp_path / "model.json"
    model.save(model_path)
    loaded = LinearGainModel.load(model_path)

    diagnostic = entropy_diagnostic(test)
    entropy_result = evaluate_policy(test, EntropySearchPolicy(), lambda_cost=0.05)
    learned_result = evaluate_policy(
        test,
        LearnedVOIPolicy(loaded, lambda_cost=0.05),
        lambda_cost=0.05,
    )

    assert diagnostic.spurious_confidence_gain_rate > 0.0
    assert diagnostic.nonbeneficial_confidence_gain_rate >= (
        diagnostic.spurious_confidence_gain_rate
    )
    assert diagnostic.entropy_top1_mismatch_rate > 0.0
    assert entropy_result["avg_tool_calls"] == 4.0
    assert entropy_result["mean_policy_utility"] == pytest.approx(
        entropy_result["mean_success_gain"] - 0.05 * 4.0
    )
    assert learned_result["avg_tool_calls"] <= 1.0
    assert learned_result["mean_oracle_regret"] < entropy_result["mean_oracle_regret"]


def test_lambda_is_applied_only_by_policy():
    records = simulate_counterfactual_dataset(n_states=20, num_candidates=4, seed=9)
    model = LinearGainModel.fit(records)
    zoom = next(record for record in records if record.action_type == "ZOOM")
    prediction = model.predict_gain(zoom)
    low_cost_policy = LearnedVOIPolicy(model, lambda_cost=0.0)
    high_cost_policy = LearnedVOIPolicy(model, lambda_cost=10.0)
    siblings = [record for record in records if record.state_id == zoom.state_id]
    assert model.predict_gain(zoom) == prediction
    assert low_cost_policy.select(siblings).tool_calls >= high_cost_policy.select(siblings).tool_calls


def test_entropy_thresholds_are_tuned_without_test_labels():
    records = simulate_counterfactual_dataset(n_states=40, num_candidates=4, seed=10)
    train, test = split_by_group(records, group="image_id", seed=10)
    entropy_threshold, reduction_threshold = tune_entropy_thresholds(
        train,
        lambda_cost=0.05,
    )
    assert isinstance(entropy_threshold, float)
    assert isinstance(reduction_threshold, float)
    result = evaluate_policy(
        test,
        EntropyThresholdPolicy(entropy_threshold),
        lambda_cost=0.05,
    )
    assert result["avg_tool_calls"] <= 4.0


def test_entropy_single_crop_gate_never_pays_for_candidate_search():
    records = simulate_counterfactual_dataset(n_states=40, num_candidates=4, seed=12)
    train, test = split_by_group(records, group="image_id", seed=12)
    random_threshold, fixed_threshold = tune_entropy_single_crop_thresholds(
        train,
        lambda_cost=0.05,
        seed=12,
    )
    assert isinstance(fixed_threshold, float)
    result = evaluate_policy(
        test,
        EntropyRandomZoomPolicy(random_threshold, seed=12),
        lambda_cost=0.05,
    )
    assert result["avg_tool_calls"] <= 1.0
    assert result["avg_visual_cost"] == result["avg_tool_calls"]


def test_entropy_bootstrap_resamples_whole_states_deterministically():
    records = simulate_counterfactual_dataset(
        n_states=12,
        num_candidates=4,
        questions_per_image=2,
        seed=11,
    )
    first = bootstrap_entropy_diagnostic(records, n_resamples=50, seed=4)
    second = bootstrap_entropy_diagnostic(records, n_resamples=50, seed=4)
    assert first == second
    assert first["resampling_unit"] == "state_id"
    assert first["n_resamples"] == 50
    metrics = first["metrics"]
    assert isinstance(metrics, dict)
    mismatch = metrics["entropy_top1_mismatch_rate"]
    assert mismatch["ci_low"] <= mismatch["ci_high"]


def test_policy_bootstrap_fixes_decisions_and_resamples_whole_states():
    records = simulate_counterfactual_dataset(
        n_states=12,
        num_candidates=4,
        questions_per_image=2,
        seed=13,
    )
    policy = EntropySearchPolicy()
    first = bootstrap_policy_evaluation(
        records,
        policy,
        lambda_cost=0.05,
        n_resamples=50,
        seed=6,
    )
    second = bootstrap_policy_evaluation(
        records,
        policy,
        lambda_cost=0.05,
        n_resamples=50,
        seed=6,
    )
    point = evaluate_policy(records, policy, lambda_cost=0.05)
    assert first == second
    assert first["resampling_unit"] == "state_id"
    assert first["n_states"] == 12
    assert first["n_decisions"] == 12
    metrics = first["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["accuracy"]["estimate"] == point["accuracy"]
    assert metrics["mean_policy_utility"]["ci_low"] <= metrics[
        "mean_policy_utility"
    ]["ci_high"]


def test_fit_baseline_command_uses_grouped_real_split(tmp_path):
    records = simulate_counterfactual_dataset(
        n_states=20,
        num_candidates=4,
        questions_per_image=2,
        seed=19,
    )
    data_path = tmp_path / "rollouts.jsonl"
    output_dir = tmp_path / "baseline"
    write_jsonl(records, data_path)
    main(
        [
            "fit-baseline",
            "--data",
            str(data_path),
            "--output-dir",
            str(output_dir),
            "--split-group",
            "image_id",
            "--seed",
            "3",
        ]
    )
    report = (output_dir / "report.md").read_text()
    assert "Frozen-rollout baseline report" in report
    assert "not a final benchmark claim" in report
