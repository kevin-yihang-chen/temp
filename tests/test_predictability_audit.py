from __future__ import annotations

import pytest

from beyond_entropy.predictability_audit import (
    AUDIT_BENCHMARKS,
    AuditVerdict,
    BenchmarkVerdictEvidence,
    PreActionInputs,
    SplitIdentity,
    assign_disjoint_split_roles,
    audit_split_disjointness,
    classify_completed_audit,
    collapse_fixed_entropy_tool,
    expected_matrix_cells,
    fixed_tool_headroom_summary,
    matrix_completion_report,
)
from beyond_entropy.schema import ActionRecord, BBox


def _siblings(
    *, state_id: str = "s0", source_id: str = "source-0"
) -> list[ActionRecord]:
    records = [
        ActionRecord(
            state_id=state_id,
            image_id=f"image-{source_id}",
            source_id=source_id,
            question="what?",
            original_image="image.png",
            replicate_id="r0",
            generation_seed=17,
            action_id="answer",
            action_type="ANSWER",
            candidate_bbox=None,
            entropy_before=1.0,
            entropy_after=1.0,
            answer_before="wrong",
            answer_after="wrong",
            correct_before=0.0,
            correct_after=0.0,
            tool_cost=0.0,
            pre_action_features={},
            metadata={},
        )
    ]
    for index, (entropy, correct) in enumerate(
        ((0.8, 0.0), (0.2, 1.0), (0.2, 0.0), (0.5, 0.0))
    ):
        records.append(
            ActionRecord(
                state_id=state_id,
                image_id=f"image-{source_id}",
                source_id=source_id,
                question="what?",
                original_image="image.png",
                replicate_id="r0",
                generation_seed=17,
                action_id=f"zoom-{index}",
                action_type="ZOOM",
                candidate_bbox=BBox(0.0, 0.0, 0.5, 0.5),
                entropy_before=1.0,
                entropy_after=entropy,
                answer_before="wrong",
                answer_after="candidate",
                correct_before=0.0,
                correct_after=correct,
                tool_cost=1.0,
                pre_action_features={},
                metadata={},
            )
        )
    return records


def test_pre_action_view_discards_untrusted_top_level_outcomes() -> None:
    row = {
        "state_id": "s",
        "image_id": "i",
        "source_id": "g",
        "success_after": [1.0, 0.0],
        "entropy_after": 0.01,
        "pre_action": {
            "entropy_before": 0.8,
            "max_probability": 0.6,
            "top1_top2_margin": 0.2,
            "shallow_question_features": [1.0, 2.0],
            "question_embedding": [1.0, 2.0],
            "global_visual_embedding": [3.0, 4.0],
            "pooled_language_state": [1.0],
            "pooled_visual_state": [2.0],
            "fused_multimodal_state": [3.0],
        },
    }
    view = PreActionInputs.from_untrusted_mapping(row)
    exported = view.to_feature_dict()
    assert "success_after" not in exported
    assert "entropy_after" not in exported
    assert view.feature_vector("l2_semantic") == (
        0.8,
        0.6,
        0.2,
        1.0,
        2.0,
        3.0,
        4.0,
        3.0,
        8.0,
    )


def test_pre_action_nested_unknown_or_target_field_fails_closed() -> None:
    row = {
        "state_id": "s",
        "image_id": "i",
        "source_id": "g",
        "pre_action": {
            "entropy_before": 0.8,
            "max_probability": 0.6,
            "top1_top2_margin": 0.2,
            "success_after": 1.0,
        },
    }
    try:
        PreActionInputs.from_untrusted_mapping(row)
    except ValueError as exc:
        assert "unknown pre_action fields" in str(exc)
    else:
        raise AssertionError("target-derived nested feature should be rejected")


def test_fixed_entropy_tool_charges_all_four_calls_and_uses_tie_break() -> None:
    outcome = collapse_fixed_entropy_tool(_siblings())[0]
    assert outcome.selected_action_id == "zoom-1"
    assert outcome.y0 == 0.0
    assert outcome.y_tool == 1.0
    assert outcome.rescue is True
    assert outcome.harm is False
    assert outcome.tool_calls == 4
    assert outcome.tool_cost == 4.0
    assert outcome.incremental_utility(0.05) == 0.8


def test_fixed_tool_headroom_is_source_balanced() -> None:
    outcomes = collapse_fixed_entropy_tool(_siblings(state_id="a", source_id="shared"))
    outcomes += collapse_fixed_entropy_tool(_siblings(state_id="b", source_id="shared"))
    outcomes += collapse_fixed_entropy_tool(_siblings(state_id="c", source_id="other"))
    report = fixed_tool_headroom_summary(outcomes, lambda_cost=0.05)
    assert report["decisions"] == 3
    assert report["sources"] == 2
    assert report["privileged_binary_oracle"]["utility"] == 0.8


def test_hard_matrix_has_exactly_36_cells_and_reports_gaps() -> None:
    cells = expected_matrix_cells()
    assert len(cells) == 36
    empty = matrix_completion_report([])
    assert empty["completed_cells"] == 0
    assert len(empty["missing"]) == 36
    complete = matrix_completion_report(cells)
    assert complete["complete"] is True


def test_split_assignment_uses_source_rgb_connected_components() -> None:
    digest = lambda value: f"{value:064x}"
    identities = [
        SplitIdentity("a", "source-a", digest(1)),
        SplitIdentity("b", "source-a", digest(2)),
        SplitIdentity("c", "source-c", digest(2)),
        SplitIdentity("d", "source-d", digest(4)),
        SplitIdentity("e", "source-e", digest(5)),
        SplitIdentity("f", "source-f", digest(6)),
    ]
    assignments, audit = assign_disjoint_split_roles(identities, seed=17)
    assert assignments["a"] == assignments["b"] == assignments["c"]
    assert set(assignments.values()) == {"train", "validation", "test"}
    assert audit["connected_components"] == 4
    assert audit["passed"] is True
    assert audit_split_disjointness(identities, assignments)["passed"] is True


def test_split_audit_rejects_rgb_leakage() -> None:
    digest = "a" * 64
    identities = [
        SplitIdentity("a", "source-a", digest),
        SplitIdentity("b", "source-b", digest),
        SplitIdentity("c", "source-c", "b" * 64),
    ]
    try:
        audit_split_disjointness(
            identities, {"a": "train", "b": "test", "c": "validation"}
        )
    except ValueError as exc:
        assert "split leakage" in str(exc)
    else:
        raise AssertionError("duplicate RGB content across roles should fail")


def _verdict_rows(**updates: float) -> list[BenchmarkVerdictEvidence]:
    defaults = dict(
        oracle_utility=0.02,
        primary_deployable_beats_strongest_baseline_lower_ci=-0.001,
        maximum_lower_ci_across_all_deployable_policies=-0.001,
        deployable_accuracy_cost_pareto=True,
        deployable_rescue_precision_higher=True,
        deployable_harm_rate_not_higher=True,
        post_action_probe_utility_lower_ci=0.001,
        l3_in_domain_improvement_lower_ci=-0.001,
        l3_image_or_cross_domain_improvement_upper_ci=0.001,
    )
    defaults.update(updates)
    return [
        BenchmarkVerdictEvidence(benchmark=name, **defaults)
        for name in AUDIT_BENCHMARKS
    ]


def test_final_verdict_rules_are_deterministic() -> None:
    assert classify_completed_audit(_verdict_rows()) == AuditVerdict.PIVOT
    assert (
        classify_completed_audit(
            _verdict_rows(primary_deployable_beats_strongest_baseline_lower_ci=0.001)
        )
        == AuditVerdict.GO
    )
    assert (
        classify_completed_audit(
            _verdict_rows(
                primary_deployable_beats_strongest_baseline_lower_ci=0.001,
                deployable_accuracy_cost_pareto=False,
                l3_in_domain_improvement_lower_ci=0.001,
                l3_image_or_cross_domain_improvement_upper_ci=-0.001,
            )
        )
        == AuditVerdict.REPRESENTATION
    )
    assert (
        classify_completed_audit(
            _verdict_rows(
                l3_in_domain_improvement_lower_ci=0.001,
                l3_image_or_cross_domain_improvement_upper_ci=-0.001,
            )
        )
        == AuditVerdict.REPRESENTATION
    )
    assert (
        classify_completed_audit(_verdict_rows(oracle_utility=0.001))
        == AuditVerdict.STOP
    )


def test_pivot_requires_every_deployable_policy_to_fail() -> None:
    with pytest.raises(ValueError, match="does not support"):
        classify_completed_audit(
            _verdict_rows(maximum_lower_ci_across_all_deployable_policies=0.001)
        )
