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
    parse_decar_predictions,
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
