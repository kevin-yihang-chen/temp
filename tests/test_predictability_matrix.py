from __future__ import annotations

import pytest

from beyond_entropy.predictability_matrix import run_predictability_matrix
from beyond_entropy.predictability_matrix_smoke import build_synthetic_datasets


def test_partial_matrix_runner_preserves_incomplete_status() -> None:
    report = run_predictability_matrix(
        build_synthetic_datasets(),
        lambda_cost=0.05,
        bootstrap_resamples=20,
        bootstrap_confidence=0.95,
        bootstrap_seed=17,
        call_rates=(0.0, 1.0),
        seeds=(17,),
        predictor_levels=("l0_uncertainty",),
        target_families=("direct_gain",),
        formal_claim_eligible=False,
    )
    assert len(report["cells"]) == 3
    assert report["matrix"]["completed_cells"] == 3
    assert report["matrix"]["expected_cells"] == 36
    assert report["matrix"]["complete"] is False
    assert report["formal_claim_eligible"] is False
    assert all(value["passed"] for value in report["split_audits"].values())
    assert set(report["strong_baselines"]) == {"chartqa", "docvqa", "hrbench"}
    assert all(
        value["selection_role"] == "validation_only"
        for value in report["strong_baselines"].values()
    )
    assert all(
        cell["strongest_baseline"]
        == report["strong_baselines"][cell["benchmark"]]["strongest_baseline"]
        for cell in report["cells"]
    )
    assert set(report["post_action_probe"]) == {"chartqa", "docvqa", "hrbench"}
    assert all(
        value["role"] == "diagnostic_only_never_deployable"
        and len(value["seeds"]) == 1
        and value["seeds"][0]["target"] == "direct_gain"
        for value in report["post_action_probe"].values()
    )


def test_formal_matrix_rejects_partial_grid() -> None:
    with pytest.raises(ValueError, match="complete frozen"):
        run_predictability_matrix(
            build_synthetic_datasets(),
            lambda_cost=0.05,
            bootstrap_resamples=20,
            bootstrap_confidence=0.95,
            bootstrap_seed=17,
            call_rates=(0.0, 1.0),
            seeds=(17,),
            predictor_levels=("l0_uncertainty",),
            target_families=("direct_gain",),
            formal_claim_eligible=True,
        )


def test_formal_matrix_rejects_unfrozen_strong_baseline_seed() -> None:
    with pytest.raises(ValueError, match="complete frozen"):
        run_predictability_matrix(
            build_synthetic_datasets(),
            lambda_cost=0.05,
            bootstrap_resamples=20,
            bootstrap_confidence=0.95,
            bootstrap_seed=17,
            call_rates=(0.0, 1.0),
            strong_baseline_random_seed=29,
            formal_claim_eligible=True,
        )
