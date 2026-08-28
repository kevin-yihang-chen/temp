from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, cast

from .dataset import group_by_decision
from .schema import ActionRecord
from .stopping import (
    FrozenWhenToCallGate,
    PreActionGateInput,
    StoppingAction,
    StoppingDecision,
)


VTOOL_GATE_METADATA_KEY = "beyond_entropy_gate"
VTOOL_GATE_SCHEMA_VERSION = 1


def normalize_question_for_vtool_join(question: str) -> str:
    """Normalize harmless formatting while retaining the question semantics."""

    normalized = unicodedata.normalize("NFKC", question)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"\s+answer\s*:\s*$", "", normalized, flags=re.IGNORECASE)
    normalized = normalized.strip()
    if not normalized:
        raise ValueError("question identity must be non-empty")
    return normalized.casefold()


def vtool_identity_join_key(image_rgb_sha256: str, question: str) -> str:
    """Bind a VTool row to a rollout by decoded RGB identity and question text."""

    if len(image_rgb_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in image_rgb_sha256
    ):
        raise ValueError("image_rgb_sha256 must be a lowercase SHA-256 digest")
    question_digest = hashlib.sha256(
        normalize_question_for_vtool_join(question).encode()
    ).hexdigest()
    return f"rgb256:{image_rgb_sha256}:question256:{question_digest}"


@dataclass(frozen=True)
class VToolGateControl:
    """JSON-safe binary gate control for VTool-R1 ``tools_kwargs.metadata``."""

    state_id: str
    action: StoppingAction
    score: float
    threshold: float
    registered_lambda_cost: float
    model_sha256: str
    spatial_action_id: None = None

    def __post_init__(self) -> None:
        if not self.state_id:
            raise ValueError("state_id must be non-empty")
        if self.action not in ("ANSWER", "CALL_VISUAL_TOOL"):
            raise ValueError(f"unsupported stopping action: {self.action!r}")
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be finite and in [0, 1]")
        if not math.isfinite(self.threshold) or not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be finite and in [0, 1]")
        if (
            not math.isfinite(self.registered_lambda_cost)
            or self.registered_lambda_cost < 0.0
        ):
            raise ValueError(
                "registered_lambda_cost must be finite and non-negative"
            )
        if len(self.model_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.model_sha256
        ):
            raise ValueError("model_sha256 must be a lowercase SHA-256 digest")
        if self.spatial_action_id is not None:
            raise ValueError("when-to-call control must not contain a spatial action")

    @property
    def should_call_tool(self) -> bool:
        return self.action == "CALL_VISUAL_TOOL"

    @classmethod
    def from_stopping_decision(
        cls,
        decision: StoppingDecision,
    ) -> "VToolGateControl":
        return cls(
            state_id=decision.state_id,
            action=decision.action,
            score=decision.score,
            threshold=decision.threshold,
            registered_lambda_cost=decision.registered_lambda_cost,
            model_sha256=decision.model_sha256,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VTOOL_GATE_SCHEMA_VERSION,
            "state_id": self.state_id,
            "action": self.action,
            "should_call_tool": self.should_call_tool,
            "score": self.score,
            "threshold": self.threshold,
            "registered_lambda_cost": self.registered_lambda_cost,
            "model_sha256": self.model_sha256,
            "spatial_action_id": None,
        }

    def merge_tools_metadata(
        self,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = deepcopy(dict(metadata or {}))
        if VTOOL_GATE_METADATA_KEY in result:
            raise ValueError(
                f"tools metadata already contains {VTOOL_GATE_METADATA_KEY!r}"
            )
        result[VTOOL_GATE_METADATA_KEY] = self.to_dict()
        return result

    @classmethod
    def from_tools_metadata(
        cls,
        metadata: Mapping[str, Any] | str,
    ) -> "VToolGateControl":
        value: Any = json.loads(metadata) if isinstance(metadata, str) else metadata
        if not isinstance(value, Mapping):
            raise ValueError("VTool tools metadata must be a mapping or JSON object")
        payload = value.get(VTOOL_GATE_METADATA_KEY)
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"VTool tools metadata is missing {VTOOL_GATE_METADATA_KEY!r}"
            )
        if int(payload.get("schema_version", -1)) != VTOOL_GATE_SCHEMA_VERSION:
            raise ValueError("unsupported VTool gate metadata schema version")
        action = str(payload["action"])
        if action not in ("ANSWER", "CALL_VISUAL_TOOL"):
            raise ValueError(f"unsupported stopping action: {action!r}")
        should_call_tool = payload.get("should_call_tool")
        expected_should_call = action == "CALL_VISUAL_TOOL"
        if should_call_tool is not None:
            if not isinstance(should_call_tool, bool):
                raise ValueError("VTool should_call_tool must be a boolean")
            if should_call_tool != expected_should_call:
                raise ValueError("VTool gate action and should_call_tool disagree")
        if payload.get("spatial_action_id") is not None:
            raise ValueError("VTool when-to-call metadata must not select a spatial action")
        return cls(
            state_id=str(payload["state_id"]),
            action=cast(StoppingAction, action),
            score=float(payload["score"]),
            threshold=float(payload["threshold"]),
            registered_lambda_cost=float(payload["registered_lambda_cost"]),
            model_sha256=str(payload["model_sha256"]),
        )


def build_vtool_gate_manifest_rows(
    records: Sequence[ActionRecord],
    gate: FrozenWhenToCallGate,
) -> list[dict[str, Any]]:
    """Build label-free gate rows keyed by rollout decision for parquet merging."""

    rows: list[dict[str, Any]] = []
    for (state_id, replicate_id), siblings in sorted(group_by_decision(records).items()):
        answers = [record for record in siblings if record.action_type == "ANSWER"]
        if len(answers) != 1:
            raise ValueError(
                f"decision {(state_id, replicate_id)!r} must contain one ANSWER"
            )
        decision = gate.decide(PreActionGateInput.from_answer_record(answers[0]))
        control = VToolGateControl.from_stopping_decision(decision)
        rows.append(
            {
                "state_id": state_id,
                "replicate_id": replicate_id,
                "tools_kwargs_metadata": control.merge_tools_metadata(),
            }
        )
    return rows
