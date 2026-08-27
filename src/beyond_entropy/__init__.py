"""Beyond Entropy: pre-action visual value-of-information research tools."""

from .schema import ActionRecord, BBox
from .rollout import ActionSpec, AgentState, GroundTruth, TaskExample

__all__ = [
    "ActionRecord",
    "ActionSpec",
    "AgentState",
    "BBox",
    "GroundTruth",
    "TaskExample",
]
__version__ = "0.2.0"
