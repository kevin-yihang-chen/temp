from __future__ import annotations

import json
from pathlib import Path

import pytest

from beyond_entropy.predictability_baselines import (
    FROZEN_UG_ACTION_IDS,
    STRONG_BASELINE_NAMES,
    apply_strong_baselines,
    fit_strong_baselines,
    random_gate_score,
    strong_baseline_report,
    trace_by_name,
    validate_fixed_tool_outcomes,
)
from beyond_entropy.predictability_audit import BinaryToolOutcome
from beyond_entropy.schema import ActionRecord, BBox


def _siblings(
    *,
    state_id: str,
    source_id: str,
    entropy_before: float,
    y0: float,
    crop_outcomes: tuple[float, float, float, float],
    entropy_order: tuple[int, int, int, int] = (0, 1, 2, 3),
) -> list[ActionRecord]:
    common = dict(
        state_id=state_id,
        image_id=f"image-{state_id}",
        source_id=source_id,
        question="what?",
        original_image=f"{state_id}.png",
        replicate_id="replicate-000",
        generation_seed=0,
        entropy_before=entropy_before,
        answer_before="baseline",
        correct_before=y0,
        pre_action_features={},
        metadata={},
    )
    result = [
        ActionRecord(
            **common,
            action_id="answer-now",
            action_type="ANSWER",
            candidate_bbox=None,
            entropy_after=entropy_before,
            answer_after="baseline",
            correct_after=y0,
            tool_cost=0.0,
        )
    ]
    for index, outcome in enumerate(crop_outcomes):
        result.append(
            ActionRecord(
                **common,
                action_id=f"ug-grid-0{index}",
                action_type="ZOOM",
                candidate_bbox=BBox(
                    0.25 * (index % 2),
                    0.25 * (index // 2),
                    0.5 + 0.25 * (index % 2),
                    0.5 + 0.25 * (index // 2),
                ),
                entropy_after=0.1 + 0.1 * entropy_order[index],
                answer_after=f"crop-{index}",
                correct_after=outcome,
                tool_cost=1.0,
            )
        )
    return result


def _validation_records() -> list[ActionRecord]:
    records: list[ActionRecord] = []
    for index, (entropy, y0, crops) in enumerate(
        (
            (0.9, 0.0, (1.0, 0.0, 0.0, 0.0)),
            (0.8, 0.0, (1.0, 0.0, 0.0, 0.0)),
            (0.2, 1.0, (0.0, 1.0, 1.0, 1.0)),
            (0.1, 1.0, (0.0, 1.0, 1.0, 1.0)),
        )
    ):
        records.extend(
            _siblings(
                state_id=f"validation-{index}",
                source_id=f"source-{index}",
                entropy_before=entropy,
                y0=y0,
                crop_outcomes=crops,
            )
        )
    return records


def test_strong_baselines_freeze_choices_on_validation_only() -> None:
    frozen = fit_strong_baselines(
        _validation_records(), lambda_cost=0.1, random_gate_seed=20260903
    )
    assert frozen.entropy_gate_threshold == 0.8
    assert frozen.fixed_crop_action_id == "ug-grid-00"
    assert frozen.strongest_name == "fixed_crop_with_matched_gate"
    assert tuple(trace.name for trace in frozen.validation_traces) == (
        STRONG_BASELINE_NAMES
    )

    test_records = _siblings(
        state_id="test-0",
        source_id="test-source",
        entropy_before=0.95,
        y0=0.0,
        crop_outcomes=(0.0, 0.0, 0.0, 1.0),
        entropy_order=(3, 2, 1, 0),
    )
    traces = apply_strong_baselines(frozen, test_records)
    fixed = trace_by_name(traces, "fixed_crop_with_matched_gate")
    uniform = trace_by_name(traces, "uniform_random_crop_expectation_with_matched_gate")
    exhaustive = trace_by_name(
        traces, "exhaustive_ug_entropy_search_charged_four_calls"
    )
    assert fixed.calls == (True,)
    assert fixed.outcomes[0].selected_action_id == "ug-grid-00"
    assert fixed.outcomes[0].y_tool == 0.0
    assert fixed.outcomes[0].tool_cost == 1.0
    assert uniform.outcomes[0].y_tool == 0.25
    assert uniform.outcomes[0].tool_calls == 1
    assert exhaustive.outcomes[0].selected_action_id == "ug-grid-03"
    assert exhaustive.outcomes[0].y_tool == 1.0
    assert exhaustive.outcomes[0].tool_cost == 4.0
    assert exhaustive.outcomes[0].tool_calls == 4

    report = strong_baseline_report(frozen, traces)
    assert report["selection_role"] == "validation_only"
    assert report["strongest_baseline"] == "fixed_crop_with_matched_gate"
    assert set(report["validation"]) == set(STRONG_BASELINE_NAMES)
    assert set(report["test"]) == set(STRONG_BASELINE_NAMES)


def test_random_gate_score_is_stable_and_outcome_free() -> None:
    first = random_gate_score("state", "replicate", seed=17)
    second = random_gate_score("state", "replicate", seed=17)
    assert first == second
    assert 0.0 <= first < 1.0
    assert first != random_gate_score("state", "replicate", seed=29)


def test_machine_protocol_matches_strong_baseline_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (root / "configs/predictability_audit_v1.json").read_text(encoding="utf-8")
    )
    assert tuple(protocol["strong_baselines"]) == STRONG_BASELINE_NAMES
    assert tuple(protocol["fixed_visual_tool"]["candidate_action_ids"]) == (
        FROZEN_UG_ACTION_IDS
    )
    implementation = protocol["strong_baseline_implementation"]
    assert implementation["random_gate_fixed_visual_tool"]["seed"] == 20260903
    assert implementation["test_outcomes_used_for_selection"] is False
    assert (
        implementation["fixed_crop_with_matched_gate"]["gate"]
        == "reuse_entropy_gate_fixed_visual_tool_threshold_and_call_mask"
    )


def test_evaluation_action_bank_must_match_validation() -> None:
    frozen = fit_strong_baselines(
        _validation_records(), lambda_cost=0.1, random_gate_seed=17
    )
    records = _siblings(
        state_id="test",
        source_id="source",
        entropy_before=0.9,
        y0=0.0,
        crop_outcomes=(1.0, 0.0, 0.0, 0.0),
    )
    records[-1] = ActionRecord.from_dict(
        {**records[-1].to_dict(), "action_id": "unregistered-crop"}
    )
    try:
        apply_strong_baselines(frozen, records)
    except ValueError as exc:
        assert "requires frozen UG action IDs" in str(exc)
    else:
        raise AssertionError("changed test action bank should fail closed")


def test_random_gate_seed_and_duplicate_feature_labels_fail_closed() -> None:
    with pytest.raises(ValueError, match="random_gate_seed must be an integer"):
        fit_strong_baselines(
            _validation_records(), lambda_cost=0.1, random_gate_seed=17.0  # type: ignore[arg-type]
        )

    item = BinaryToolOutcome(
        state_id="state",
        replicate_id="replicate",
        image_id="image",
        source_id="source",
        selected_action_id="ug-grid-00",
        y0=0.0,
        y_tool=1.0,
        tool_cost=4.0,
        tool_calls=4,
    )
    with pytest.raises(ValueError, match="feature outcomes contain duplicate"):
        validate_fixed_tool_outcomes([item, item], [item])
