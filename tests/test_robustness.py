import json

import pytest

from beyond_entropy.robustness import (
    aggregate_semantic_reports,
    build_robustness_markdown,
)


def _report(seed, utility):
    return {
        "run": {"seed": seed},
        "lambda_sweep": [
            {
                "lambda_cost": 0.05,
                "policy_results": [
                    {
                        "policy": "learned",
                        "accuracy": 0.8 + utility,
                        "accuracy_gain": utility,
                        "tool_use_rate": 0.1,
                        "mean_policy_utility": utility,
                        "mean_oracle_regret": 0.05 - utility,
                    }
                ],
            }
        ],
    }


def test_semantic_robustness_aggregation_tracks_split_range(tmp_path):
    paths = []
    for seed, utility in ((3, -0.01), (7, 0.02)):
        path = tmp_path / f"seed-{seed}.json"
        path.write_text(json.dumps(_report(seed, utility)))
        paths.append(path)
    report = aggregate_semantic_reports(paths, lambda_cost=0.05)
    learned = report["policies"]["learned"]
    assert learned["mean_policy_utility"]["mean"] == pytest.approx(0.005)
    assert learned["mean_policy_utility"]["min"] == -0.01
    assert learned["mean_policy_utility"]["max"] == 0.02
    assert learned["positive_utility_splits"] == 1
    assert "overlapping test sets" in build_robustness_markdown(report)


def test_semantic_robustness_rejects_duplicate_seeds(tmp_path):
    paths = []
    for index in range(2):
        path = tmp_path / f"report-{index}.json"
        path.write_text(json.dumps(_report(3, 0.0)))
        paths.append(path)
    with pytest.raises(ValueError, match="duplicate seeds"):
        aggregate_semantic_reports(paths)
