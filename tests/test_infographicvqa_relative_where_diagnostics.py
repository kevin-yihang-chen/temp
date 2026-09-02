from __future__ import annotations

from pathlib import Path

from beyond_entropy.infographicvqa_relative_where import RELATIVE_WHERE_VARIANTS
from beyond_entropy.infographicvqa_relative_where_diagnostics import (
    ACTION_GENERALIZATION_SCHEMA,
    audit_relative_where_action_generalization,
)


def _inputs() -> tuple[list[dict], list[dict], list[dict]]:
    predictions = []
    nll_rows = []
    answer_rows = []
    teachers = (0, 1, 2, 3, 0)
    predicted = (0, 0, 2, 1, 0)
    for index, (teacher, chosen) in enumerate(zip(teachers, predicted, strict=True)):
        key = f"state-{index:03d}"
        scores = [-2.0, -2.0, -2.0, -2.0]
        scores[chosen] = 2.0
        scores[teacher] = max(scores[teacher], 1.0)
        probabilities = [0.05, 0.05, 0.05, 0.05]
        probabilities[chosen] = 0.85
        predictions.append(
            {
                "schema": "infographicvqa_relative_where_oof_prediction_v1",
                "state_id": key,
                "replicate_id": "replicate-000",
                "image_id": f"image-{index:03d}",
                "source_id": f"source-{index // 2:03d}",
                "outer_fold": index,
                "variants": {
                    name: {
                        "action_scores": scores,
                        "action_probabilities": probabilities,
                        "selected_action_id": f"ug-grid-{chosen:02d}",
                        "predicted_margin": 1.0,
                    }
                    for name in RELATIVE_WHERE_VARIANTS
                },
            }
        )
        answer_rows.append(
            {
                "state_id": key,
                "replicate_id": "replicate-000",
                "image_id": f"image-{index:03d}",
                "source_id": f"source-{index // 2:03d}",
                "action_id": "answer-now",
                "action_type": "ANSWER",
                "entropy_before": index / 10.0,
            }
        )
        for action_index, action_id in enumerate(
            ("answer-now", "ug-grid-00", "ug-grid-01", "ug-grid-02", "ug-grid-03")
        ):
            if action_id == "answer-now":
                value = 1.0
            else:
                crop_index = action_index - 1
                value = 0.0 if crop_index == teacher else 0.1 + crop_index / 100.0
            nll_rows.append(
                {
                    "schema": "visual_action_answer_nll_v1",
                    "state_id": key,
                    "replicate_id": "replicate-000",
                    "image_id": f"image-{index:03d}",
                    "source_id": f"source-{index // 2:03d}",
                    "action_id": action_id,
                    "action_type": "ANSWER" if action_id == "answer-now" else "ZOOM",
                    "answer_mean_nll": value,
                }
            )
    return predictions, nll_rows, answer_rows


def test_action_generalization_reports_factorized_and_regret_metrics() -> None:
    predictions, nll_rows, answer_rows = _inputs()
    result = audit_relative_where_action_generalization(
        predictions,
        nll_rows,
        answer_rows,
        expected_decisions=5,
        expected_sources=3,
    )
    assert result["schema"] == ACTION_GENERALIZATION_SCHEMA
    assert result["population"] == {"decisions": 5, "sources": 3, "images": 5}
    assert result["validation_or_test_inputs_used"] is False
    primary = result["variants"]["relative_teacher_entropy"]
    metrics = primary["overall"]["metrics"]
    assert metrics["exact_agreement"]["question_balanced"] == 0.6
    assert metrics["row_agreement"]["question_balanced"] == 0.8
    assert metrics["column_agreement"]["question_balanced"] == 0.8
    assert metrics["nll_regret"]["question_balanced"] > 0.0
    assert len(primary["by_outer_fold"]) == 5
    assert len(primary["by_confidence_decile"]) == 5
    assert sum(primary["predicted_action_counts"].values()) == 5
    assert result["teacher_label_audit"]["exact_tie_rate"] == 0.0


def test_action_generalization_counts_exact_teacher_ties() -> None:
    predictions, nll_rows, answer_rows = _inputs()
    first = [
        row
        for row in nll_rows
        if row["state_id"] == "state-000" and row["action_id"] == "ug-grid-01"
    ][0]
    first["answer_mean_nll"] = 0.0
    result = audit_relative_where_action_generalization(
        predictions,
        nll_rows,
        answer_rows,
        expected_decisions=5,
        expected_sources=3,
    )
    assert result["teacher_label_audit"]["exact_tie_rate"] == 0.2
    assert result["teacher_label_audit"]["near_tie_rate"]["atol_0"] == 0.2


def test_action_generalization_worker_binds_parent_and_notifications() -> None:
    root = Path(__file__).resolve().parents[1]
    worker = (
        root
        / "scripts/slurm_infographicvqa_relative_where_action_generalization_audit.sh"
    ).read_text()
    submitter = (
        root
        / "scripts/submit_infographicvqa_relative_where_action_generalization_audit.sh"
    ).read_text()
    assert "#SBATCH --partition=debug" in worker
    assert "#SBATCH --gres=gpu:rtx_4090:1" in worker
    assert "#SBATCH --mail-user=yihangc@connect.hku.hk" in worker
    assert "#SBATCH --mail-type=ALL" in worker
    assert 'export CUDA_VISIBLE_DEVICES=""' in worker
    assert "94e292d33fe9e55e9d50d53eef5f2590c4d5c751567feb2fe736867fc7dd6b6b" in worker
    assert "1c51131d6b8599a3733c3018e0a53570552ff09fff19aa07bcb7bf61b984e61c" in worker
    assert "relative-where-action-generalization-audit-v1" in worker
    assert "changes_parent_train_gate" in worker
    assert "unset HF_TOKEN HUGGINGFACE_HUB_TOKEN" in worker
    assert "-lt 30" in submitter
    assert "-lt 120" in submitter
    assert "--test-only --export=NONE" in submitter
    assert "--parsable --export=NONE" in submitter
    assert "git push" not in submitter
