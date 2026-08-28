"""Beyond Entropy: pre-action visual value-of-information research tools."""

from .schema import ActionRecord, BBox
from .rollout import ActionSpec, AgentState, GroundTruth, TaskExample
from .stopping import FrozenWhenToCallGate, PreActionGateInput, StoppingDecision

__all__ = [
    "ActionRecord",
    "ActionSpec",
    "AgentState",
    "BBox",
    "GroundTruth",
    "FrozenWhenToCallGate",
    "PreActionGateInput",
    "StoppingDecision",
    "TaskExample",
]
__version__ = "0.2.0"
