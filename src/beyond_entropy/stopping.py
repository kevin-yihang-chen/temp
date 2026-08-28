from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from .schema import ActionRecord
from .transfer_gate import score_frozen_factorized_context_baseline


StoppingAction = Literal["ANSWER", "CALL_VISUAL_TOOL"]


@dataclass(frozen=True)
class PreActionGateInput:
    """Minimal runtime state for deciding whether to acquire more visual evidence."""

    state_id: str
    question: str
    answer_before: str
    entropy_before: float
    normalized_token_entropies: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.state_id:
            raise ValueError("state_id must be non-empty")
        if not self.question.strip():
            raise ValueError("question must be non-empty")
        if not math.isfinite(self.entropy_before) or self.entropy_before < 0.0:
            raise ValueError("entropy_before must be finite and non-negative")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in self.normalized_token_entropies
        ):
            raise ValueError(
                "normalized_token_entropies must be finite and non-negative"
            )

    def _feature_record(self) -> ActionRecord:
        """Build a label-free feature carrier for the existing frozen scorer."""

        return ActionRecord(
            state_id=self.state_id,
            image_id=self.state_id,
            source_id=self.state_id,
            question=self.question,
            original_image="<runtime-pre-action-state>",
            replicate_id="runtime",
            generation_seed=None,
            action_id="answer-now",
            action_type="ANSWER",
            candidate_bbox=None,
            entropy_before=self.entropy_before,
            entropy_after=self.entropy_before,
            answer_before=self.answer_before,
            answer_after=self.answer_before,
            correct_before=0.0,
            correct_after=0.0,
            tool_cost=0.0,
            metadata={
                "baseline_backend": {
                    "normalized_token_entropies": list(
                        self.normalized_token_entropies
                    )
                }
            },
        )


@dataclass(frozen=True)
class StoppingDecision:
    state_id: str
    action: StoppingAction
    score: float
    threshold: float
    registered_lambda_cost: float
    model_sha256: str
    spatial_action_id: None = None

    @property
    def should_call_tool(self) -> bool:
        return self.action == "CALL_VISUAL_TOOL"


class FrozenWhenToCallGate:
    """Frozen stopping adapter that deliberately cannot select a spatial action."""

    def __init__(
        self,
        model: Mapping[str, Any],
        *,
        registered_lambda_cost: float = 0.05,
        model_sha256: str | None = None,
    ) -> None:
        if model.get("model_type") != "factorized_context_cross_benchmark_transfer":
            raise ValueError("unsupported frozen stopping model type")
        if "threshold" not in model:
            raise ValueError("frozen stopping model is missing its threshold")
        threshold = float(model["threshold"])
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("frozen stopping threshold must be in [0, 1]")
        if not math.isfinite(registered_lambda_cost) or registered_lambda_cost < 0.0:
            raise ValueError("registered_lambda_cost must be finite and non-negative")
        self._model = deepcopy(dict(model))
        self.threshold = threshold
        self.registered_lambda_cost = registered_lambda_cost
        self.model_sha256 = model_sha256 or hashlib.sha256(
            json.dumps(self._model, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
        registered_lambda_cost: float = 0.05,
    ) -> "FrozenWhenToCallGate":
        payload = Path(path).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError(
                f"frozen stopping model SHA-256 mismatch: {digest} != {expected_sha256}"
            )
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError("frozen stopping model must be a JSON object")
        return cls(
            value,
            registered_lambda_cost=registered_lambda_cost,
            model_sha256=digest,
        )

    def decide(self, state: PreActionGateInput) -> StoppingDecision:
        score = score_frozen_factorized_context_baseline(
            self._model,
            state._feature_record(),
        )
        action: StoppingAction = (
            "CALL_VISUAL_TOOL" if score >= self.threshold else "ANSWER"
        )
        return StoppingDecision(
            state_id=state.state_id,
            action=action,
            score=score,
            threshold=self.threshold,
            registered_lambda_cost=self.registered_lambda_cost,
            model_sha256=self.model_sha256,
        )
