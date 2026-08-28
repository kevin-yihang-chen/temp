import json
from dataclasses import replace

import pytest

from beyond_entropy.schema import BBox
from beyond_entropy.stopping import FrozenWhenToCallGate, PreActionGateInput


def _constant_model(*, threshold: float) -> dict[str, object]:
    return {
        "model_type": "factorized_context_cross_benchmark_transfer",
        "threshold": threshold,
        "error_scaler_mean": [0.0] * 27,
        "error_scaler_scale": [1.0] * 27,
        "error_coefficient": [0.0] * 27,
        "error_intercept": 0.0,
        "rescue_scaler_mean": [0.0] * 27,
        "rescue_scaler_scale": [1.0] * 27,
        "rescue_coefficient": [0.0] * 27,
        "rescue_intercept": 0.0,
    }


def _runtime_state() -> PreActionGateInput:
    return PreActionGateInput(
        state_id="chartqa-1",
        question="What is the total?",
        answer_before="42",
        entropy_before=0.3,
        normalized_token_entropies=(0.2, 0.4),
    )


def test_when_to_call_gate_has_no_spatial_action_surface():
    gate = FrozenWhenToCallGate(_constant_model(threshold=0.2))
    state = _runtime_state()
    decision = gate.decide(state)

    assert decision.action == "CALL_VISUAL_TOOL"
    assert decision.should_call_tool
    assert decision.score == pytest.approx(0.25)
    assert decision.spatial_action_id is None
    assert not hasattr(state, "correct_before")
    assert not hasattr(state, "correct_after")
    assert not hasattr(state, "candidate_bbox")


def test_when_to_call_gate_can_stop():
    gate = FrozenWhenToCallGate(_constant_model(threshold=0.3))
    state = _runtime_state()
    decision = gate.decide(state)

    assert decision.action == "ANSWER"
    assert not decision.should_call_tool
    assert decision.score == pytest.approx(0.25)


def test_when_to_call_gate_verifies_frozen_model_hash(tmp_path):
    path = tmp_path / "model.json"
    path.write_text(json.dumps(_constant_model(threshold=0.2)) + "\n")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        FrozenWhenToCallGate.load(path, expected_sha256="0" * 64)

    gate = FrozenWhenToCallGate.load(path)
    assert len(gate.model_sha256) == 64


def test_when_to_call_gate_copies_model_and_rejects_invalid_threshold():
    model = _constant_model(threshold=0.2)
    gate = FrozenWhenToCallGate(model)
    model["threshold"] = 0.9
    coefficients = model["error_coefficient"]
    assert isinstance(coefficients, list)
    coefficients[0] = 1000.0

    assert gate.decide(_runtime_state()).action == "CALL_VISUAL_TOOL"
    with pytest.raises(ValueError, match=r"threshold must be in \[0, 1\]"):
        FrozenWhenToCallGate(_constant_model(threshold=1.1))


def test_pre_action_gate_input_rejects_invalid_uncertainty():
    with pytest.raises(ValueError, match="entropy_before"):
        PreActionGateInput(
            state_id="chartqa-1",
            question="What is the total?",
            answer_before="42",
            entropy_before=float("nan"),
        )


def test_pre_action_gate_input_from_answer_record_uses_only_runtime_fields():
    state = _runtime_state()
    record = state._feature_record()
    rebuilt = PreActionGateInput.from_answer_record(record)

    assert rebuilt == state
    with pytest.raises(ValueError, match="requires an ANSWER"):
        PreActionGateInput.from_answer_record(
            replace(
                record,
                action_id="crop-0",
                action_type="ZOOM",
                candidate_bbox=BBox(0.0, 0.0, 0.5, 0.5),
                tool_cost=1.0,
            )
        )
