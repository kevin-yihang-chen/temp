from __future__ import annotations

import pytest

from beyond_entropy.phase_c_formal_evaluation import (
    call_set_jaccard,
    evaluate_phase_c_formal,
    seed_averaged_cluster_bootstrap_delta,
    spearman_rank_correlation,
)
from beyond_entropy.phase_c_formal_transaction import FORMAL_MODES
from beyond_entropy.phase_c_training import BENCHMARKS, METHODS, SEEDS
from beyond_entropy.schema import BBox
from beyond_entropy.sequential_schema import AcquiredObservationSpec, SequentialRolloutRecord
from scripts.evaluate_factorized_phase_c_formal import _markdown, _render_figures


def _record(benchmark: str, index: int) -> SequentialRolloutRecord:
    stop, continued = ((0, 1), (0, 1), (1, 0), (1, 1))[index]
    source = f"{benchmark}-shared-source"
    image = f"{benchmark}-shared-image" if benchmark == "hrbench" else f"image-{index}"
    return SequentialRolloutRecord(
        state_id=f"{benchmark}-s{index}", image_id=image, source_id=source,
        question=f"question {index}", original_image=f"/tmp/{benchmark}-{index}.png",
        step_index=1,
        acquired_observations=(AcquiredObservationSpec(
            "crop-a", BBox(0, 0, .5, .5), 1,
        ),),
        proposed_action_id="crop-b", proposed_bbox=BBox(.5, .5, 1, 1),
        proposed_visual_cost=1, replicate_id="replicate-000", generation_seed=0,
        stop_answer="stop", stop_correct=stop,
        stop_entropy=(4, 3, 2, 1)[index],
        stop_max_probability=(.1, .2, .3, .4)[index],
        stop_top1_top2_margin=(.1, .2, .3, .4)[index],
        continue_answer="continue", continue_correct=continued,
        continue_entropy=.1, continue_max_probability=.9,
        continue_top1_top2_margin=.8,
    )


def _plan() -> dict:
    return {
        "benchmarks": {name: {"states": 4} for name in BENCHMARKS},
        "policy": {
            "rates": [0, .1, .25, .5, .75, 1],
            "lambdas": [0, .025, .05, .1, .2],
            "primary_call_rate": .25, "primary_lambda": .05,
            "bootstrap_samples": 10_000, "bootstrap_seed": 20260913,
        },
        "baselines": {
            "uncertainty": ["entropy", "confidence", "margin"],
            "random_seed": 20260913,
        },
        "ablations": {
            "semantic_score_correlation_max": .98,
            "semantic_call_set_jaccard_max": .95,
            "required_domains_with_changed_ranking": 2,
        },
        "go_rule": {
            "positive_mean_delta_vs_outcome_domains": 2,
            "required_domain_source_bootstrap_ci_low_vs_outcome_positive": 1,
            "successful_domain_max_regression_vs_strongest_uncertainty": .005,
        },
    }


def _payloads(records_by_benchmark: dict, *, identical_ablations: bool = False) -> list[dict]:
    payloads = []
    method_scores = {
        "outcome_only": [1, 2, 4, 3],
        "counterfactual_utility": [3, 4, 2, 1],
        "factorized_potential_outcomes": [.8, .6, .4, .2],
    }
    ablated = [.8, .6, .4, .2] if identical_ablations else [.2, .4, .8, .6]
    for seed in SEEDS:
        rows = []
        for benchmark, records in records_by_benchmark.items():
            for method in METHODS:
                modes = FORMAL_MODES if method == "factorized_potential_outcomes" else ("original",)
                for mode in modes:
                    values = method_scores[method] if mode == "original" else ablated
                    for record, score in zip(records, values):
                        row = {
                            "method": method, "seed": seed, "mode": mode,
                            "benchmark": benchmark, "state_id": record.state_id,
                            "replicate_id": record.replicate_id,
                            "source_id": record.source_id,
                            "continue_score": score,
                            "action_logits": [0, 0, score]
                            if method == "factorized_potential_outcomes" else [0, score],
                            "measurement": {
                                "observed_images": 2, "already_acquired_crops": 1,
                                "proposed_crop_executions": 0, "prompt_tokens": 100,
                            },
                        }
                        if method == "factorized_potential_outcomes":
                            row["factorized_probabilities"] = {
                                "error_probability": score,
                                "rescue_probability_given_error": 1,
                                "harm_probability_given_correct": 0,
                                "expected_gain": score,
                            }
                        rows.append(row)
        payloads.append({
            "schema": "factorized_phase_c_formal_predictions_v1",
            "one_shot": True, "test_accessed": True,
            "formal_claim_eligible": True, "seed": seed, "predictions": rows,
        })
    return payloads


def _records() -> dict:
    return {name: [_record(name, index) for index in range(4)] for name in BENCHMARKS}


def test_rank_and_call_set_diagnostics_are_deterministic() -> None:
    assert spearman_rank_correlation([1, 2, 3], [3, 2, 1]) == pytest.approx(-1)
    assert spearman_rank_correlation([1, 1, 1], [1, 1, 1]) == 1
    assert call_set_jaccard([True, False, True], [False, True, True]) == pytest.approx(1 / 3)


def test_seed_averaged_bootstrap_uses_image_clusters_for_hrbench() -> None:
    records = _records()["hrbench"]
    left = {seed: [True, False, False, False] for seed in SEEDS}
    right = {seed: [False, False, True, False] for seed in SEEDS}
    result = seed_averaged_cluster_bootstrap_delta(
        records, left, right, benchmark="hrbench", lambda_cost=0,
        samples=10_000, seed=17,
    )
    assert result["observed_delta"] == .5
    assert result["ci_low"] == .5
    assert result["resampling_unit"] == "image_id"
    assert result["clusters"] == 1


def test_formal_go_requires_seed_averaged_outcome_gain_and_semantic_dependence() -> None:
    records = _records()
    report = evaluate_phase_c_formal(_plan(), records, _payloads(records))
    assert report["decision"] == "GO"
    assert report["positive_vs_outcome_domains"] == list(BENCHMARKS)
    assert report["significant_vs_outcome_domains"] == list(BENCHMARKS)
    assert all(value["passed"] for value in report["semantic_gates"].values())
    assert report["benchmarks"]["hrbench"]["source_cluster_field"] == "image_id"
    primary = report["benchmarks"]["chartqa"]["primary_metrics"]
    assert primary["factorized_potential_outcomes"]["accuracy"] == .75
    assert primary["outcome_only"]["accuracy"] == .25
    assert primary["factorized_potential_outcomes"]["top1_action_regret"] == .25


def test_identical_semantic_controls_force_no_go() -> None:
    records = _records()
    report = evaluate_phase_c_formal(
        _plan(), records, _payloads(records, identical_ablations=True)
    )
    assert report["decision"] == "NO_GO"
    assert report["go_checks"]["all_semantic_ablations_passed"] is False
    assert all(not value["passed"] for value in report["semantic_gates"].values())


def test_prediction_rows_fail_closed_on_outcome_leakage_field() -> None:
    records = _records()
    payloads = _payloads(records)
    payloads[0]["predictions"][0]["stop_correct"] = 1
    with pytest.raises(ValueError, match="outcome-free contract"):
        evaluate_phase_c_formal(_plan(), records, payloads)


def test_formal_renderer_writes_both_registered_figures_and_four_decisions(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    records = _records()
    report = evaluate_phase_c_formal(_plan(), records, _payloads(records))
    _render_figures(report, tmp_path)
    assert len(list(tmp_path.glob("*-accuracy-cost-frontier.png"))) == 3
    assert len(list(tmp_path.glob("*-utility-prediction.png"))) == 3
    markdown = _markdown(report)
    assert "Decision: **GO**" in markdown
    assert "## Four decisions" in markdown
    assert "ChartQA" not in markdown  # benchmark IDs stay canonical/lowercase
