import json

import pytest

from beyond_entropy.stopping import StoppingDecision
from beyond_entropy.vtool_adapter import (
    VTOOL_GATE_METADATA_KEY,
    VToolGateControl,
)


def _decision() -> StoppingDecision:
    return StoppingDecision(
        state_id="chartqa-1",
        action="CALL_VISUAL_TOOL",
        score=0.61,
        threshold=0.45,
        registered_lambda_cost=0.05,
        model_sha256="a" * 64,
    )


def test_vtool_gate_metadata_round_trip_preserves_existing_bbox_metadata():
    control = VToolGateControl.from_stopping_decision(_decision())
    metadata = control.merge_tools_metadata(
        {"source": "chartqa_v_bar", "x_values_bbox": {"2019": [1, 2, 3, 4]}}
    )

    assert metadata["source"] == "chartqa_v_bar"
    restored = VToolGateControl.from_tools_metadata(json.dumps(metadata))
    assert restored == control
    assert restored.should_call_tool
    assert restored.spatial_action_id is None


def test_vtool_gate_metadata_rejects_spatial_selection():
    metadata = VToolGateControl.from_stopping_decision(_decision()).merge_tools_metadata()
    payload = metadata[VTOOL_GATE_METADATA_KEY]
    assert isinstance(payload, dict)
    payload["spatial_action_id"] = "crop-2"

    with pytest.raises(ValueError, match="must not select a spatial action"):
        VToolGateControl.from_tools_metadata(metadata)


def test_vtool_gate_metadata_rejects_inconsistent_action_flag():
    metadata = VToolGateControl.from_stopping_decision(_decision()).merge_tools_metadata()
    payload = metadata[VTOOL_GATE_METADATA_KEY]
    assert isinstance(payload, dict)
    payload["should_call_tool"] = False

    with pytest.raises(ValueError, match="disagree"):
        VToolGateControl.from_tools_metadata(metadata)

    payload["should_call_tool"] = "false"
    with pytest.raises(ValueError, match="must be a boolean"):
        VToolGateControl.from_tools_metadata(metadata)


def test_vtool_gate_metadata_refuses_silent_overwrite():
    control = VToolGateControl.from_stopping_decision(_decision())
    with pytest.raises(ValueError, match="already contains"):
        control.merge_tools_metadata({VTOOL_GATE_METADATA_KEY: {}})
