from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from beyond_entropy.infographicvqa_decar_evaluation import (
    DECAR_ACTION_IDS,
    evaluate_decar_oof,
)
from beyond_entropy.schema import ActionRecord, BBox


def _records(decisions: int) -> list[ActionRecord]:
    result: list[ActionRecord] = []
    crop_deltas = (0.20, 0.10, -0.10, 0.00)
    for index in range(decisions):
        common: dict[str, Any] = {
            "state_id": f"state-{index:03d}",
            "image_id": f"image-{index:03d}",
            "source_id": f"source-{index // 2:03d}",
            "question": "Question?",
            "original_image": f"image-{index:03d}.png",
            "replicate_id": "replicate-000",
            "generation_seed": index,
            "entropy_before": 2.0 - 0.01 * index,
            "answer_before": "before",
            "correct_before": 0.40,
        }
        result.append(
            ActionRecord(
                **common,
                action_id="answer-now",
                action_type="ANSWER",
                candidate_bbox=None,
                entropy_after=2.0 - 0.01 * index,
                answer_after="before",
                correct_after=0.40,
                tool_cost=0.0,
            )
        )
        for action_index, (action_id, delta) in enumerate(
            zip(DECAR_ACTION_IDS, crop_deltas, strict=True)
        ):
            result.append(
                ActionRecord(
                    **common,
                    action_id=action_id,
                    action_type="ZOOM",
                    candidate_bbox=BBox(0.0, 0.0, 0.5, 0.5),
                    entropy_after=0.1 + 0.1 * action_index,
                    answer_after=f"after-{action_index}",
                    correct_after=0.40 + delta,
                    tool_cost=1.0,
                )
            )
    return result


def _variant(name: str, *, score: float) -> dict[str, Any]:
    common: dict[str, Any] = {
        "selected_action_id": "ug-grid-00",
        "predicted_gap": score + 0.01,
        "predicted_margin": 0.10,
        "score": score,
        "eligible": True,
    }
    if name in {"decar", "task_value_only"}:
        return {
            **common,
            "rescue_probability": 0.80,
            "neutral_probability": 0.15,
            "harm_probability": 0.05,
            "predicted_delta": 0.20,
        }
    if name == "no_harm_head":
        return {
            **common,
            "rescue_probability": 0.80,
            "other_probability": 0.20,
            "predicted_delta": 0.20,
        }
    return common


def _predictions(decisions: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(decisions):
        score = float(decisions - index)
        rows.append(
            {
                "schema": "infographicvqa_decar_oof_prediction_v1",
                "state_id": f"state-{index:03d}",
                "replicate_id": "replicate-000",
                "image_id": f"image-{index:03d}",
                "source_id": f"source-{index // 2:03d}",
                "outer_fold": (index // 2) % 5,
                "variants": {
                    name: _variant(name, score=score)
                    for name in (
                        "decar",
                        "task_value_only",
                        "loss_only",
                        "no_harm_head",
                    )
                },
            }
        )
    return rows


def _evaluate(records: list[ActionRecord]) -> dict[str, Any]:
    decisions = len(records) // 5
    sources = decisions // 2
    bootstrap_indices = np.tile(
        np.arange(sources, dtype=np.int32),
        (16, 1),
    )
    return evaluate_decar_oof(
        records,
        _predictions(decisions),
        bootstrap_indices=bootstrap_indices,
        target_call_rates=(0.20,),
        expected_decisions=decisions,
        expected_sources=sources,
    )


def test_one_crop_and_four_crop_policies_have_matched_execution_budget() -> None:
    result = _evaluate(_records(40))
    point = result["operating_points"][0]

    assert point["primary_actual_calls"] == 8
    assert point["selection_audits"]["entropy_gate_random_and_fixed"][
        "matched_call_count"
    ]
    assert point["selection_audits"]["entropy_gated_ug"]["matched_call_count"]

    for name in (
        "decar",
        "entropy_random",
        "entropy_fixed_ug_grid_00",
        "entropy_gated_ug",
    ):
        assert point["policies"][name]["question_balanced"][
            "executed_crops"
        ] == pytest.approx(0.20)

    assert point["policies"]["decar"]["raw_calls"] == 8
    assert point["policies"]["entropy_gated_ug"]["raw_calls"] == 2
    assert set(point["feasible_non_oracle_baselines"]) == {
        "answer_now",
        "entropy_random",
        "entropy_fixed_ug_grid_00",
        "entropy_gated_ug",
    }


def test_entropy_boundary_tie_is_retained_and_blocks_qualification() -> None:
    tied_records = [replace(row, entropy_before=1.0) for row in _records(40)]
    result = _evaluate(tied_records)
    point = result["operating_points"][0]

    random_audit = point["selection_audits"]["entropy_gate_random_and_fixed"]
    exhaustive_audit = point["selection_audits"]["entropy_gated_ug"]
    assert random_audit["target_calls"] == 8
    assert random_audit["actual_calls"] == 40
    assert random_audit["matched_call_count"] is False
    assert exhaustive_audit["target_calls"] == 2
    assert exhaustive_audit["actual_calls"] == 40
    assert exhaustive_audit["matched_call_count"] is False
    assert point["qualification_rules"]["all_audits_passed"] is False
    assert point["qualified"] is False
