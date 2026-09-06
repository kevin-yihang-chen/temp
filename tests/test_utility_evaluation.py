from __future__ import annotations

import pytest

from beyond_entropy.utility_evaluation import choice_metrics, paired_choice_interval, policy_choices
from test_utility_sft import make_samples


def test_eight_baselines_charge_full_search_and_single_selection_differently():
    samples = make_samples()
    gains = {"s1": [0, .5, -.5]}
    learned = {name: gains for name in ("format_sft", "best_action_sft", "utility_sft")}
    choices = policy_choices(samples, lambda_cost=.05, learned_gains=learned, frozen_voi_calls={"s1": True})
    assert len(choices) == 8
    assert choices["ug"][0].tool_calls == 2  # This unit fixture has K=2.
    assert choices["frozen_voi"] == choices["ug"]
    assert choices["utility_sft"][0].tool_calls == 1
    learned_metrics = choice_metrics(samples, choices["utility_sft"], lambda_cost=.05)["source_balanced"]
    assert learned_metrics["accuracy"] == 1
    assert learned_metrics["net_utility"] == pytest.approx(.45)
    assert learned_metrics["top1_regret"] == 0
    assert learned_metrics["useful_tool_recall"] == 1
    ug = choice_metrics(samples, choices["ug"], lambda_cost=.05)["source_balanced"]
    assert ug["accuracy"] == 0  # Lower entropy belongs to harmful crop.
    assert ug["avg_visual_cost"] == 2
    assert ug["net_utility"] == pytest.approx(-.6)
    assert ug["unnecessary_tool_call_rate"] == 1
    interval = paired_choice_interval(samples, choices["utility_sft"], choices["ug"], lambda_cost=.05, resamples=20)
    assert interval["point"] == pytest.approx(1.05)
    assert interval["resampling_unit"] == "source_id"
    stopped = policy_choices(samples, lambda_cost=1, learned_gains=learned, frozen_voi_calls={"s1": False})
    assert stopped["utility_sft"][0].index == 0
    assert stopped["oracle"][0].index == 0
    assert stopped["frozen_voi"][0].tool_calls == 0
    assert choice_metrics(samples, stopped["utility_sft"], lambda_cost=1)["source_balanced"]["useful_tool_precision"] is None


def test_missing_comparator_rejected():
    with pytest.raises(ValueError, match="all three"):
        policy_choices(make_samples(), lambda_cost=0, learned_gains={}, frozen_voi_calls={"s1": True})
