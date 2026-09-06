import pytest

from scripts.render_utility_sft_figures import ranking_summary


def test_ranking_summary_reports_action_ranking_and_regret():
    report = ranking_summary(
        [[0, 1, -1], [0, 0, 1]],
        [[0, .5, -.5], [0, 1, .5]],
    )
    assert report["top1_action_accuracy"] == .5
    assert report["mean_top1_regret"] == .5
    assert len(report["true_rank_by_predicted_rank_counts"]) == 3
    assert sum(map(sum, report["true_rank_by_predicted_rank_counts"])) == 6
    assert 0 <= report["pairwise_ranking_accuracy"] <= 1
    with pytest.raises(ValueError):
        ranking_summary([], [])
