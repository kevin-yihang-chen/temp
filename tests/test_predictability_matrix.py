from __future__ import annotations

from dataclasses import replace

import pytest

from beyond_entropy.predictability_matrix import (
    BenchmarkTestData,
    _majority_calls,
    _select_validation_cell_keys,
    evaluate_frozen_predictability_matrix,
    fit_predictability_matrix,
    frozen_predictability_matrix_report,
    load_frozen_predictability_matrix,
    run_predictability_matrix,
    save_frozen_predictability_matrix,
)
from beyond_entropy.predictability_modeling import FrozenAuditCell
from beyond_entropy.predictability_matrix_smoke import build_synthetic_datasets


def test_registered_selection_and_seed_majority_are_deterministic() -> None:
    cells = [
        FrozenAuditCell(
            level=level,
            target=target,
            seed=seed,
            variant="entropy",
            raw_model=None,
            calibrators=None,
            threshold=0.0,
            validation_metrics={
                "validation_utility": utility,
                "validation_calls": calls,
            },
        )
        for seed in (17, 29, 47)
        for level, target, utility, calls in (
            ("l0_uncertainty", "direct_gain", 0.1, 2),
            ("l1_shallow", "rescue_harm", 0.1, 1),
            ("l2_semantic", "factorized", 0.1, 1),
        )
    ]
    assert _select_validation_cell_keys(
        cells,
        seeds=(17, 29, 47),
        predictor_levels=("l0_uncertainty", "l1_shallow", "l2_semantic"),
        target_families=("direct_gain", "rescue_harm", "factorized"),
    ) == (
        ("l1_shallow", "rescue_harm", 17),
        ("l1_shallow", "rescue_harm", 29),
        ("l1_shallow", "rescue_harm", 47),
    )
    assert _majority_calls(
        ((True, False, True), (False, False, True), (True, True, False))
    ) == (True, False, True)
    assert _majority_calls(((True,), (False,))) == (False,)


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
    assert set(report["oracle_headroom"]) == {"chartqa", "docvqa", "hrbench"}
    assert set(report["primary_deployable"]) == {
        "chartqa",
        "docvqa",
        "hrbench",
    }
    assert all(
        len(value["selected_cell_keys"]) == 1
        and value["aggregation"] == "strict_seed_majority_with_even_tie_as_no_call"
        for value in report["primary_deployable"].values()
    )
    assert all(
        value["available"] is False
        for value in report["representation_diagnostic"].values()
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


def test_separate_freeze_round_trip_precedes_held_out_evaluation(tmp_path) -> None:
    complete = build_synthetic_datasets()
    frozen = fit_predictability_matrix(
        {name: data.development() for name, data in complete.items()},
        lambda_cost=0.05,
        seeds=(17,),
        predictor_levels=("l0_uncertainty",),
        target_families=("direct_gain",),
        provenance={"protocol_sha256": "a" * 64},
    )
    inventory = frozen_predictability_matrix_report(frozen)
    assert inventory["test_data_present"] is False
    assert inventory["format_version"] == 2
    assert all(len(item["cells"]) == 1 for item in inventory["benchmarks"].values())
    assert all(
        len(item["frozen_policy_selection"]["selected_deployable_cell_keys"]) == 1
        for item in inventory["benchmarks"].values()
    )

    model_path = tmp_path / "frozen.pkl"
    report_path = tmp_path / "freeze.json"
    saved = save_frozen_predictability_matrix(
        frozen, model_path=model_path, report_path=report_path
    )
    restored = load_frozen_predictability_matrix(
        model_path, expected_sha256=saved["model_sha256"]
    )
    report = evaluate_frozen_predictability_matrix(
        restored,
        {name: data.held_out_test() for name, data in complete.items()},
        bootstrap_resamples=20,
        bootstrap_confidence=0.95,
        bootstrap_seed=17,
        call_rates=(0.0, 1.0),
    )
    assert report["frozen_before_test"] is True
    assert report["schema"] == "predictability_matrix_report_v3"
    assert report["matrix"]["completed_cells"] == 3
    assert all(value["passed"] for value in report["split_audits"].values())
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_frozen_predictability_matrix(model_path, expected_sha256="0" * 64)


def test_separate_evaluation_rejects_development_rgb_leakage() -> None:
    complete = build_synthetic_datasets()
    frozen = fit_predictability_matrix(
        {name: data.development() for name, data in complete.items()},
        lambda_cost=0.05,
        seeds=(17,),
        predictor_levels=("l0_uncertainty",),
        target_families=("direct_gain",),
    )
    held_out = {name: data.held_out_test() for name, data in complete.items()}
    chartqa = held_out["chartqa"]
    leaked_hash = complete["chartqa"].train[0].image_rgb_sha256
    held_out["chartqa"] = BenchmarkTestData(
        test=(replace(chartqa.test[0], image_rgb_sha256=leaked_hash),)
        + tuple(chartqa.test[1:]),
        post_action_test=(
            replace(chartqa.post_action_test[0], image_rgb_sha256=leaked_hash),
        )
        + tuple(chartqa.post_action_test[1:]),
        test_siblings=chartqa.test_siblings,
    )
    with pytest.raises(ValueError, match="development/test source or decoded-RGB"):
        evaluate_frozen_predictability_matrix(
            frozen,
            held_out,
            bootstrap_resamples=20,
            bootstrap_confidence=0.95,
            bootstrap_seed=17,
            call_rates=(0.0, 1.0),
        )


def test_formal_one_shot_wrapper_is_forbidden_after_complete_grid_check() -> None:
    with pytest.raises(ValueError, match="formal one-shot evaluation is forbidden"):
        run_predictability_matrix(
            build_synthetic_datasets(),
            lambda_cost=0.05,
            bootstrap_resamples=20,
            bootstrap_confidence=0.95,
            bootstrap_seed=17,
            call_rates=(0.0, 1.0),
            formal_claim_eligible=True,
        )
