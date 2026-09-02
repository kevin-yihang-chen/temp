from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from beyond_entropy.infographicvqa_decar_evaluation import (
    DECAR_ACTION_IDS,
    build_decar_outcomes,
    complete_tie_top_keys,
    evaluate_decar_oof,
    evaluate_entropy_oracle_where_factorization,
    evaluate_entropy_where_hybrid,
    parse_decar_predictions,
)
from beyond_entropy.infographicvqa_attention_where_evaluation import (  # noqa: E402
    evaluate_attention_where,
)
from beyond_entropy.infographicvqa_relative_where import RELATIVE_WHERE_VARIANTS
from beyond_entropy.infographicvqa_relative_where_evaluation import (
    _first_exact_difference,
    evaluate_relative_where_oof,
)
from beyond_entropy.schema import ActionRecord, BBox


def _decision(index: int) -> list[ActionRecord]:
    state_id = f"state-{index:03d}"
    source_id = f"source-{index // 2:03d}"
    deltas = (0.0, 0.5 if index == 0 else 0.3 if index == 1 else 0.1, -0.2, 0.1)
    common: dict[str, Any] = {
        "state_id": state_id,
        "image_id": f"image-{index:03d}",
        "source_id": source_id,
        "question": "Question?",
        "original_image": f"image-{index:03d}.png",
        "replicate_id": "replicate-000",
        "generation_seed": 0,
        "entropy_before": 1.0 - 0.01 * index,
        "answer_before": "before",
        "correct_before": 0.2,
    }
    records = [
        ActionRecord(
            **common,
            action_id="answer-now",
            action_type="ANSWER",
            candidate_bbox=None,
            entropy_after=1.0 - 0.01 * index,
            answer_after="before",
            correct_after=0.2,
            tool_cost=0.0,
        )
    ]
    for action_index, (action_id, delta) in enumerate(
        zip(DECAR_ACTION_IDS, deltas, strict=True)
    ):
        records.append(
            ActionRecord(
                **common,
                action_id=action_id,
                action_type="ZOOM",
                candidate_bbox=BBox(0.0, 0.0, 0.5, 0.5),
                entropy_after=0.1 + 0.1 * action_index,
                answer_after=f"after-{action_index}",
                correct_after=0.2 + delta,
                tool_cost=1.0,
            )
        )
    return records


def _variant(
    name: str,
    *,
    action_id: str,
    score: float,
    eligible: bool = True,
) -> dict[str, Any]:
    common: dict[str, Any] = {
        "selected_action_id": action_id,
        "predicted_gap": score + 0.01,
        "predicted_margin": 0.02,
        "score": score,
        "eligible": eligible,
    }
    if name in {"decar", "task_value_only"}:
        return {
            **common,
            "rescue_probability": 0.6,
            "neutral_probability": 0.3,
            "harm_probability": 0.1,
            "predicted_delta": 0.2,
        }
    if name == "no_harm_head":
        return {
            **common,
            "rescue_probability": 0.6,
            "other_probability": 0.4,
            "predicted_delta": 0.2,
        }
    return common


def _prediction(index: int) -> dict[str, Any]:
    score = 1.0 - 0.1 * index
    return {
        "schema": "infographicvqa_decar_oof_prediction_v1",
        "state_id": f"state-{index:03d}",
        "replicate_id": "replicate-000",
        "image_id": f"image-{index:03d}",
        "source_id": f"source-{index // 2:03d}",
        "outer_fold": (index // 2) % 5,
        "variants": {
            "decar": _variant("decar", action_id="ug-grid-01", score=score),
            "task_value_only": _variant(
                "task_value_only", action_id="ug-grid-00", score=score
            ),
            "loss_only": _variant("loss_only", action_id="ug-grid-03", score=score),
            "no_harm_head": _variant(
                "no_harm_head", action_id="ug-grid-00", score=score
            ),
        },
    }


def test_complete_tie_top_keys_retains_boundary_ties() -> None:
    scores = {
        ("a", "r"): 1.0,
        ("b", "r"): 0.5,
        ("c", "r"): 0.5,
        ("d", "r"): 0.1,
    }
    selected, audit = complete_tie_top_keys(scores, target_calls=2)
    assert selected == {("a", "r"), ("b", "r"), ("c", "r")}
    assert audit["actual_calls"] == 3
    assert audit["boundary_ties"] == 2
    assert audit["ties_preserved"] is True


def test_prediction_join_rejects_outcome_fields() -> None:
    records = [record for index in range(2) for record in _decision(index)]
    outcomes = build_decar_outcomes(records, expected_decisions=2, expected_sources=1)
    rows = [_prediction(0), _prediction(1)]
    rows[0]["correct_after"] = 1.0
    with pytest.raises(ValueError, match="forbidden outcomes"):
        parse_decar_predictions(rows, outcomes)


def test_registered_evaluation_reports_paired_metrics_and_costs() -> None:
    records = [record for index in range(8) for record in _decision(index)]
    predictions = [_prediction(index) for index in range(8)]
    indices = np.tile(np.arange(4, dtype=np.int32), (100, 1))
    result = evaluate_decar_oof(
        records,
        predictions,
        bootstrap_indices=indices,
        target_call_rates=(0.25,),
        expected_decisions=8,
        expected_sources=4,
    )
    assert result["population"] == {"decisions": 8, "sources": 4, "images": 8}
    assert result["decision"] == "decar_not_advanced"
    point = result["operating_points"][0]
    assert point["primary_actual_calls"] == 2
    assert (
        point["selection_audits"]["entropy_gate_random_and_fixed"]["matched_call_count"]
        is True
    )
    primary = point["policies"]["decar"]
    assert primary["question_balanced"]["call"] == pytest.approx(0.25)
    assert primary["question_balanced"]["executed_crops"] == pytest.approx(0.25)
    assert primary["question_balanced"]["anls_gain"] == pytest.approx(0.1)
    assert primary["question_balanced"]["utility"] == pytest.approx(0.0875)
    assert primary["question_balanced"]["baseline_exact_accuracy"] == 0.0
    assert primary["question_balanced"]["final_exact_accuracy"] == 0.0
    assert primary["question_balanced"]["helpful_call_precision"] == pytest.approx(1.0)
    random = point["policies"]["entropy_random"]["question_balanced"]
    assert random["executed_crops"] == pytest.approx(0.25)
    assert random["utility"] < primary["question_balanced"]["utility"]
    gated_ug = point["policies"]["entropy_gated_ug"]["question_balanced"]
    assert gated_ug["executed_crops"] == 0.0
    assert point["source_bootstrap"]["decar"]["additive"]["utility"][
        "ci_low"
    ] == pytest.approx(primary["source_balanced"]["utility"])
    assert "task_value_only" in point["paired_source_utility_differences"]
    decomposition = point["failure_decomposition"]
    assert decomposition["question_balanced"]["action_choice_regret"] >= 0.0
    assert decomposition["question_balanced"]["gate_false_positive_mass"] == 0.0
    assert decomposition["source_concentration"]["sources"] == 4
    assert (
        0.0
        <= decomposition["source_concentration"]["top_10pct_sources_call_fraction"]
        <= 1.0
    )
    exhaustive = result["static_references"]["charged_exhaustive_ug"]
    assert exhaustive["question_balanced"]["executed_crops"] == 4.0


def test_entropy_when_oof_where_hybrid_reuses_formal_identities() -> None:
    records = [record for index in range(8) for record in _decision(index)]
    predictions = [_prediction(index) for index in range(8)]
    indices = np.tile(np.arange(4, dtype=np.int32), (100, 1))
    formal = evaluate_decar_oof(
        records,
        predictions,
        bootstrap_indices=indices,
        expected_decisions=8,
        expected_sources=4,
    )
    result = evaluate_entropy_where_hybrid(
        records,
        predictions,
        formal,
        bootstrap_indices=indices,
        expected_decisions=8,
        expected_sources=4,
        expected_bootstrap_resamples=100,
    )
    assert result["decision"] == "hybrid_train_not_supported"
    assert result["validation_or_test_inputs_used"] is False
    assert result["population"] == {"decisions": 8, "sources": 4, "images": 8}
    point = result["operating_points"][0]
    assert point["selection_audits"]["matches_formal_identities"] is True
    assert (
        point["actual_calls"] == formal["operating_points"][0]["primary_actual_calls"]
    )
    primary = point["policies"]["entropy_when_decar_where"]
    assert (
        primary["question_balanced"]["executed_crops"]
        == primary["question_balanced"]["call"]
    )
    assert (
        primary["question_balanced"]["utility"]
        > point["policies"]["entropy_when_task_value_where"]["question_balanced"][
            "utility"
        ]
    )
    assert "original_decar" in point["paired_source_utility_differences"]
    assert point["qualification_rules"]["minimum_calls_and_sources"] is False


def test_entropy_oracle_where_factorizes_crop_selection_on_same_states() -> None:
    records = [record for index in range(8) for record in _decision(index)]
    predictions = [_prediction(index) for index in range(8)]
    for prediction in predictions:
        prediction["variants"]["decar"]["selected_action_id"] = "ug-grid-02"
    indices = np.tile(np.arange(4, dtype=np.int32), (100, 1))
    formal = evaluate_decar_oof(
        records,
        predictions,
        bootstrap_indices=indices,
        expected_decisions=8,
        expected_sources=4,
    )
    hybrid = evaluate_entropy_where_hybrid(
        records,
        predictions,
        formal,
        bootstrap_indices=indices,
        expected_decisions=8,
        expected_sources=4,
        expected_bootstrap_resamples=100,
    )
    result = evaluate_entropy_oracle_where_factorization(
        records,
        predictions,
        hybrid,
        bootstrap_indices=indices,
        expected_decisions=8,
        expected_sources=4,
        expected_bootstrap_resamples=100,
    )
    assert result["decision"] == "where_bottleneck_not_supported"
    assert result["outcome_oracle_used"] is True
    assert result["deployable_method_evidence"] is False
    assert result["validation_or_test_inputs_used"] is False
    point = result["operating_points"][0]
    oracle = point["policies"]["entropy_when_task_oracle_where"]
    learned = point["policies"]["entropy_when_decar_where"]
    assert (
        oracle["question_balanced"]["utility"] > learned["question_balanced"]["utility"]
    )
    assert point["qualification_rules"]["exact_arithmetic_consistency"] is True
    assert point["per_state_regret_identity_passed"] is True
    assert "entropy_when_decar_where" in point["paired_source_utility_differences"]
    assert point["qualification_rules"]["minimum_calls_and_sources"] is False


def test_relative_where_evaluation_reuses_frozen_states_and_bootstrap() -> None:
    records = [record for index in range(8) for record in _decision(index)]
    decar_predictions = [_prediction(index) for index in range(8)]
    indices = np.tile(np.arange(4, dtype=np.int32), (100, 1))
    formal = evaluate_decar_oof(
        records,
        decar_predictions,
        bootstrap_indices=indices,
        expected_decisions=8,
        expected_sources=4,
    )
    hybrid = evaluate_entropy_where_hybrid(
        records,
        decar_predictions,
        formal,
        bootstrap_indices=indices,
        expected_decisions=8,
        expected_sources=4,
        expected_bootstrap_resamples=100,
    )
    oracle = evaluate_entropy_oracle_where_factorization(
        records,
        decar_predictions,
        hybrid,
        bootstrap_indices=indices,
        expected_decisions=8,
        expected_sources=4,
        expected_bootstrap_resamples=100,
    )
    oracle["decision"] = "where_bottleneck_supported"
    relative_predictions = []
    nll_rows = []
    for index in range(8):
        scores = (0.0, 1.0, -1.0, -0.5)
        probabilities = (0.15, 0.55, 0.1, 0.2)
        relative_predictions.append(
            {
                "schema": "infographicvqa_relative_where_oof_prediction_v1",
                "state_id": f"state-{index:03d}",
                "replicate_id": "replicate-000",
                "image_id": f"image-{index:03d}",
                "source_id": f"source-{index // 2:03d}",
                "outer_fold": (index // 2) % 5,
                "variants": {
                    name: {
                        "action_scores": list(scores),
                        "action_probabilities": list(probabilities),
                        "selected_action_id": "ug-grid-01",
                        "predicted_margin": 1.0,
                    }
                    for name in RELATIVE_WHERE_VARIANTS
                },
            }
        )
        for action_index, action_id in enumerate(("answer-now", *DECAR_ACTION_IDS)):
            nll_rows.append(
                {
                    "schema": "visual_action_answer_nll_v1",
                    "state_id": f"state-{index:03d}",
                    "replicate_id": "replicate-000",
                    "image_id": f"image-{index:03d}",
                    "source_id": f"source-{index // 2:03d}",
                    "action_id": action_id,
                    "answer_mean_nll": (
                        1.0
                        if action_id == "answer-now"
                        else {
                            "ug-grid-00": 0.1,
                            "ug-grid-01": 0.0,
                            "ug-grid-02": 0.2,
                            "ug-grid-03": 0.3,
                        }[action_id]
                    ),
                }
            )
    result = evaluate_relative_where_oof(
        records,
        relative_predictions,
        decar_predictions,
        nll_rows,
        hybrid,
        oracle,
        bootstrap_indices=indices,
        expected_decisions=8,
        expected_sources=4,
        expected_bootstrap_resamples=100,
    )
    assert result["decision"] == "relative_where_train_not_supported"
    assert result["validation_or_test_inputs_used"] is False
    assert result["relative_prediction_outcomes_included"] is False
    point = result["operating_points"][0]
    assert point["selection_audit"]["matched_call_count"] is True
    assert (
        point["policies"]["relative_teacher_entropy"]["teacher_agreement"][
            "question_balanced"
        ]
        == 1.0
    )
    assert "old_decar_where" in point["paired_source_utility_differences"]
    assert point["qualification_rules"]["minimum_calls_and_sources"] is False


def test_relative_where_frozen_difference_reports_first_exact_path() -> None:
    assert (
        _first_exact_difference(
            {"source_balanced": {"utility": 0.1, "call": 0.2}},
            {"source_balanced": {"utility": 0.3, "call": 0.2}},
        )
        == "$.source_balanced.utility: recomputed=0.1 frozen=0.3"
    )
    assert _first_exact_difference({"value": [1, 2]}, {"value": [1, 2]}) is None


def test_attention_where_evaluation_reuses_all_frozen_comparators() -> None:
    torch = pytest.importorskip("torch")
    records = [record for index in range(8) for record in _decision(index)]
    decar_predictions = [_prediction(index) for index in range(8)]
    indices = np.tile(np.arange(4, dtype=np.int32), (100, 1))
    formal = evaluate_decar_oof(
        records,
        decar_predictions,
        bootstrap_indices=indices,
        expected_decisions=8,
        expected_sources=4,
    )
    hybrid = evaluate_entropy_where_hybrid(
        records,
        decar_predictions,
        formal,
        bootstrap_indices=indices,
        expected_decisions=8,
        expected_sources=4,
        expected_bootstrap_resamples=100,
    )
    oracle = evaluate_entropy_oracle_where_factorization(
        records,
        decar_predictions,
        hybrid,
        bootstrap_indices=indices,
        expected_decisions=8,
        expected_sources=4,
        expected_bootstrap_resamples=100,
    )
    oracle["decision"] = "where_bottleneck_supported"
    relative_predictions = []
    nll_rows = []
    attention_decisions = []
    for index in range(8):
        scores = (0.0, 1.0, -1.0, -0.5)
        probabilities = (0.15, 0.55, 0.1, 0.2)
        relative_predictions.append(
            {
                "schema": "infographicvqa_relative_where_oof_prediction_v1",
                "state_id": f"state-{index:03d}",
                "replicate_id": "replicate-000",
                "image_id": f"image-{index:03d}",
                "source_id": f"source-{index // 2:03d}",
                "outer_fold": (index // 2) % 5,
                "variants": {
                    name: {
                        "action_scores": list(scores),
                        "action_probabilities": list(probabilities),
                        "selected_action_id": "ug-grid-01",
                        "predicted_margin": 1.0,
                    }
                    for name in RELATIVE_WHERE_VARIANTS
                },
            }
        )
        siblings = _decision(index)
        zooms = siblings[1:]
        attention_decisions.append(
            {
                "state_id": f"state-{index:03d}",
                "replicate_id": "replicate-000",
                "image_id": f"image-{index:03d}",
                "source_id": f"source-{index // 2:03d}",
                "question": "Question?",
                "action_ids": list(DECAR_ACTION_IDS),
                "tool_costs": torch.ones(4, dtype=torch.float32),
                "bboxes": torch.tensor(
                    [record.candidate_bbox.to_list() for record in zooms],
                    dtype=torch.float32,
                ),
                "state_signals": torch.tensor(
                    [1.0 - 0.01 * index], dtype=torch.float32
                ),
                "question_region_attention": torch.tensor(
                    [0.1, 0.7, 0.1, 0.1], dtype=torch.float32
                ),
                "question_image_attention_mass": 1.0,
            }
        )
        for action_id in ("answer-now", *DECAR_ACTION_IDS):
            nll_rows.append(
                {
                    "schema": "visual_action_answer_nll_v1",
                    "state_id": f"state-{index:03d}",
                    "replicate_id": "replicate-000",
                    "image_id": f"image-{index:03d}",
                    "source_id": f"source-{index // 2:03d}",
                    "action_id": action_id,
                    "answer_mean_nll": (
                        1.0
                        if action_id == "answer-now"
                        else {
                            "ug-grid-00": 0.1,
                            "ug-grid-01": 0.0,
                            "ug-grid-02": 0.2,
                            "ug-grid-03": 0.3,
                        }[action_id]
                    ),
                }
            )
    relative = evaluate_relative_where_oof(
        records,
        relative_predictions,
        decar_predictions,
        nll_rows,
        hybrid,
        oracle,
        bootstrap_indices=indices,
        expected_decisions=8,
        expected_sources=4,
        expected_bootstrap_resamples=100,
    )
    attention_payload = {
        "format_version": 1,
        "metadata": {
            "outcomes_included": False,
            "question_region_attention": {
                "source_features_sha256": "features",
                "source_rollouts_sha256": "rollouts",
                "model_revision": "model",
                "attention_implementation": "eager",
                "top_layers": 4,
                "head_pooling": "mean",
                "question_token_pooling": "mean",
                "candidate_pooling": "ROI mean then normalize across candidates",
                "candidate_actions_executed": False,
                "replace_question_embedding": False,
                "code_revision": "code",
                "completed_decisions": 8,
                "total_decisions": 8,
            },
        },
        "decisions": attention_decisions,
    }
    result = evaluate_attention_where(
        records,
        attention_payload,
        decar_predictions,
        relative_predictions,
        nll_rows,
        hybrid,
        oracle,
        relative,
        expected_attention_code_revision="code",
        expected_model_revision="model",
        expected_source_features_sha256="features",
        expected_rollouts_sha256="rollouts",
        bootstrap_indices=indices,
        expected_decisions=8,
        expected_sources=4,
        expected_bootstrap_resamples=100,
    )
    assert result["validation_or_test_inputs_used"] is False
    assert result["attention_features_outcomes_included"] is False
    assert result["all_state_attention_localization"]["exact_nll_teacher"][
        "question_balanced"
    ] == 1.0
    point = result["operating_points"][0]
    assert point["frozen_comparators_exact_match"] is True
    assert set(point["paired_source_utility_differences"]) == {
        "entropy_fixed_ug_grid_00",
        "entropy_random",
        "old_decar_where",
        "relative_where",
    }
    assert len(result["attention_max_score_deciles"]) == 8
    assert len(result["attention_margin_deciles"]) == 8


def test_decar_evaluation_script_freezes_train_only_full_protocol() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts/evaluate_infographicvqa_decar_oof.py").read_text()
    assert "DECAR_BOOTSTRAP_RESAMPLES" in script
    assert "DECAR_BOOTSTRAP_SEED" in script
    assert "EXPECTED_DECISIONS = 23_946" in script
    assert "EXPECTED_SOURCES = 2_204" in script
    assert "EXPECTED_IMAGES = 4_406" in script
    assert "bootstrap-indices.npy" in script
    assert "validation_or_test_inputs_used" in script
    assert "download" not in script.lower()


def test_entropy_where_hybrid_script_reuses_formal_bootstrap_and_seals_eval() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root / "scripts/evaluate_infographicvqa_decar_entropy_where_hybrid.py"
    ).read_text()
    assert "EXPECTED_DECISIONS = 23_946" in script
    assert "EXPECTED_SOURCES = 2_204" in script
    assert "EXPECTED_IMAGES = 4_406" in script
    assert '"task_value_only": 17_446' in script
    assert 'mmap_mode="r"' in script
    assert '"formal_bootstrap_reused": True' in script
    assert '"validation_opened": False' in script
    assert '"test_opened": False' in script
    assert "download" not in script.lower()


def test_entropy_where_hybrid_slurm_contract_hides_required_gpu_and_notifies() -> None:
    root = Path(__file__).resolve().parents[1]
    worker = (
        root / "scripts/slurm_infographicvqa_decar_entropy_where_hybrid.sh"
    ).read_text()
    submitter = (
        root / "scripts/submit_infographicvqa_decar_entropy_where_hybrid.sh"
    ).read_text()
    assert "#SBATCH --partition=debug" in worker
    assert "#SBATCH --gres=gpu:rtx_4090:1" in worker
    assert "#SBATCH --cpus-per-task=4" in worker
    assert "#SBATCH --mem=64G" in worker
    assert "#SBATCH --time=00:45:00" in worker
    assert "#SBATCH --mail-user=yihangc@connect.hku.hk" in worker
    assert "#SBATCH --mail-type=ALL" in worker
    assert 'export CUDA_VISIBLE_DEVICES=""' in worker
    assert "resource-amendment" in worker
    assert "unset HF_TOKEN HUGGINGFACE_HUB_TOKEN" in worker
    assert "-lt 45" in submitter
    assert "-lt 180" in submitter
    assert "--test-only --export=NONE" in submitter
    assert "--parsable --export=NONE" in submitter
    assert "resource_amendment_sha256" in submitter
    assert "git push" not in submitter


def test_entropy_oracle_where_runner_marks_outcome_oracle_and_keeps_seal() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root / "scripts/evaluate_infographicvqa_entropy_oracle_where_factorization.py"
    ).read_text()
    assert "EXPECTED_DECISIONS" in script
    assert "EXPECTED_SOURCES" in script
    assert "EXPECTED_IMAGES" in script
    assert 'mmap_mode="r"' in script
    assert '"outcome_oracle_used": True' in script
    assert '"deployable_method_evidence": False' in script
    assert '"formal_bootstrap_reused": True' in script
    assert '"validation_opened": False' in script
    assert '"test_opened": False' in script
    assert "download" not in script.lower()


def test_entropy_oracle_where_slurm_contract_hides_gpu_and_notifies() -> None:
    root = Path(__file__).resolve().parents[1]
    worker = (
        root / "scripts/slurm_infographicvqa_entropy_oracle_where_factorization.sh"
    ).read_text()
    submitter = (
        root / "scripts/submit_infographicvqa_entropy_oracle_where_factorization.sh"
    ).read_text()
    assert "#SBATCH --partition=debug" in worker
    assert "#SBATCH --gres=gpu:rtx_4090:1" in worker
    assert "#SBATCH --cpus-per-task=4" in worker
    assert "#SBATCH --mem=64G" in worker
    assert "#SBATCH --time=00:45:00" in worker
    assert "#SBATCH --mail-user=yihangc@connect.hku.hk" in worker
    assert "#SBATCH --mail-type=ALL" in worker
    assert 'export CUDA_VISIBLE_DEVICES=""' in worker
    assert "outcome_oracle_used" in worker
    assert "deployable_method_evidence" in worker
    assert "unset HF_TOKEN HUGGINGFACE_HUB_TOKEN" in worker
    assert "-lt 45" in submitter
    assert "-lt 180" in submitter
    assert "--test-only --export=NONE" in submitter
    assert "--parsable --export=NONE" in submitter
    assert "git push" not in submitter


def test_relative_where_runners_keep_oof_predictions_outcome_free() -> None:
    root = Path(__file__).resolve().parents[1]
    fit_runner = (root / "scripts/fit_infographicvqa_relative_where_oof.py").read_text()
    eval_runner = (
        root / "scripts/evaluate_infographicvqa_relative_where_oof.py"
    ).read_text()
    assert "EXPECTED_DECISIONS = 23_946" in fit_runner
    assert "EXPECTED_SOURCES = 2_204" in fit_runner
    assert "fit_relative_where_oof" in fit_runner
    assert '"scientific_endpoints_computed": False' in fit_runner
    assert '"prediction_outcomes_included": False' in fit_runner
    assert 'mmap_mode="r"' in eval_runner
    assert '"formal_bootstrap_reused": True' in eval_runner
    assert '"validation_opened": False' in eval_runner
    assert '"test_opened": False' in eval_runner
    assert "download" not in fit_runner.lower()
    assert "download" not in eval_runner.lower()


def test_relative_where_h800_worker_binds_code_quota_and_notifications() -> None:
    root = Path(__file__).resolve().parents[1]
    worker = (
        root / "scripts/slurm_infographicvqa_relative_where_oof_h800.sh"
    ).read_text()
    submitter = (
        root / "scripts/submit_infographicvqa_relative_where_oof_h800.sh"
    ).read_text()
    assert "#SBATCH --partition=q-h800" in worker
    assert "#SBATCH --gres=gpu:h800:1" in worker
    assert "#SBATCH --cpus-per-task=12" in worker
    assert "#SBATCH --mem=192G" in worker
    assert "#SBATCH --time=01:00:00" in worker
    assert "#SBATCH --mail-user=yihangc@connect.hku.hk" in worker
    assert "#SBATCH --mail-type=ALL" in worker
    assert 'gpu_name}" != "NVIDIA H800"' in worker
    assert "--epochs 200" in worker
    assert "relative_prediction_outcomes_included" in worker
    assert "unset HF_TOKEN HUGGINGFACE_HUB_TOKEN" in worker
    assert "expected_train_module_sha256" in worker
    assert "expected_eval_module_sha256" in worker
    assert "expected_decar_eval_sha256" in worker
    assert "-lt 60" in submitter
    assert "-lt 720" in submitter
    assert "--test-only --export=NONE" in submitter
    assert "--parsable --export=NONE" in submitter
    assert "git push" not in submitter


def test_relative_where_recovery_reuses_predictions_and_keeps_gate_frozen() -> None:
    root = Path(__file__).resolve().parents[1]
    worker = (
        root / "scripts/slurm_infographicvqa_relative_where_evaluation_recovery.sh"
    ).read_text()
    submitter = (
        root / "scripts/submit_infographicvqa_relative_where_evaluation_recovery.sh"
    ).read_text()
    assert "#SBATCH --partition=debug" in worker
    assert "#SBATCH --gres=gpu:rtx_4090:1" in worker
    assert "#SBATCH --mail-user=yihangc@connect.hku.hk" in worker
    assert "#SBATCH --mail-type=ALL" in worker
    assert 'export CUDA_VISIBLE_DEVICES=""' in worker
    assert "94e292d33fe9e55e9d50d53eef5f2590c4d5c751567feb2fe736867fc7dd6b6b" in worker
    assert "1277c06f98a14ffbd8cfddb4a833c87a1a2a38de149cf786c87c6621e6f00def" in worker
    assert "evaluation-recovery-v1" in worker
    assert "fit_infographicvqa_relative_where_oof.py" not in worker
    assert "frozen_comparators_exact_match" in worker
    assert "unset HF_TOKEN HUGGINGFACE_HUB_TOKEN" in worker
    assert "-lt 45" in submitter
    assert "-lt 180" in submitter
    assert "--test-only --export=NONE" in submitter
    assert "--parsable --export=NONE" in submitter
    assert "git push" not in submitter


def test_decar_oof_worker_and_submitter_freeze_h800_and_notifications() -> None:
    root = Path(__file__).resolve().parents[1]
    worker = (root / "scripts/slurm_infographicvqa_decar_oof_h800.sh").read_text()
    submitter = (root / "scripts/submit_infographicvqa_decar_oof_h800.sh").read_text()
    assert "#SBATCH --gres=gpu:h800:1" in worker
    assert "#SBATCH --cpus-per-task=12" in worker
    assert "#SBATCH --mem=192G" in worker
    assert "#SBATCH --time=04:00:00" in worker
    assert "#SBATCH --mail-user=yihangc@connect.hku.hk" in worker
    assert "#SBATCH --mail-type=ALL" in worker
    assert "--epochs 200" in worker
    assert "--device cuda:0" in worker
    assert "--expected-rollouts-sha256" in worker
    assert "--expected-predictions-sha256" in worker
    assert "scientific_endpoints_used_for_selection == false" in worker
    assert "validation_or_test_inputs_used" in worker
    assert "unset HF_TOKEN HUGGINGFACE_HUB_TOKEN" in worker
    assert "resource-amendment" in worker
    assert "startup-hash-correction" in worker
    assert "7f0f23e65d155e728b592a96b5d5a463d67cfed9742977532535e6a232b0837a" in worker
    assert (
        "7f0f23e65d155e728b592a96b5d5a463d67cfed9742977532535e6b8cdb6d0af5a4da60"
        not in worker
    )
    assert "resource_amendment_sha256" in submitter
    assert "startup_hash_correction_sha256" in submitter
    assert "-lt 240" in submitter
    assert "--test-only --export=NONE" in submitter
    assert "--parsable --export=NONE" in submitter
    assert "git push" not in submitter
