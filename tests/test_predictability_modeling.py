from __future__ import annotations

from beyond_entropy.predictability_audit import (
    BinaryToolOutcome,
    PreActionInputs,
)
from beyond_entropy.predictability_modeling import (
    AuditExample,
    evaluate_frozen_audit_cell,
    fit_frozen_audit_cell,
    registered_model_variants,
)


def _examples(start: int, stop: int, *, role_offset: int = 0) -> list[AuditExample]:
    result = []
    for index in range(start, stop):
        signal = float((index + role_offset) % 4) / 3.0
        rescued = signal > 0.5
        inputs = PreActionInputs(
            state_id=f"state-{role_offset}-{index}",
            image_id=f"image-{role_offset}-{index}",
            source_id=f"source-{role_offset}-{index // 2}",
            entropy_before=signal + 0.1,
            max_probability=1.0 - signal,
            top1_top2_margin=1.0 - signal,
            shallow_question_features=(signal, 1.0 - signal),
            question_embedding=(signal, 1.0),
            global_visual_embedding=(1.0, signal),
            pooled_language_state=(signal, 1.0),
            pooled_visual_state=(1.0, signal),
            fused_multimodal_state=(signal, signal),
        )
        outcome = BinaryToolOutcome(
            state_id=inputs.state_id,
            replicate_id="r0",
            image_id=inputs.image_id,
            source_id=inputs.source_id,
            selected_action_id="zoom-0",
            y0=0.0,
            y_tool=1.0 if rescued else 0.0,
            tool_cost=4.0,
            tool_calls=4,
        )
        result.append(AuditExample(inputs, outcome, f"{role_offset + index + 1:064x}"))
    return result


def test_registered_model_ladder_is_fixed() -> None:
    assert registered_model_variants("l0_uncertainty") == (
        "entropy",
        "max_probability",
        "top1_top2_margin",
    )
    assert registered_model_variants("l1_shallow") == ("linear", "small_mlp")
    assert registered_model_variants("l2_semantic") == ("small_mlp",)
    assert registered_model_variants("l3_frozen_qwen") == ("linear", "two_layer_mlp")


def test_linear_direct_gain_cell_fits_on_train_val_then_evaluates_test() -> None:
    train = _examples(0, 24)
    validation = _examples(0, 12, role_offset=100)
    test = _examples(0, 12, role_offset=200)
    cell = fit_frozen_audit_cell(
        train,
        validation,
        level="l0_uncertainty",
        target="direct_gain",
        seed=17,
        lambda_cost=0.05,
    )
    predictions, metrics = evaluate_frozen_audit_cell(cell, test, lambda_cost=0.05)
    assert len(predictions) == len(test)
    assert cell.variant in registered_model_variants("l0_uncertainty")
    utility = metrics["incremental_utility"]
    assert isinstance(utility, (float, int)) and utility > 0.0
    assert metrics["calls"] == 6
