from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

from .schema import ActionRecord, BBox


@dataclass(frozen=True)
class TaskSample:
    state_id: str
    image_path: str
    question: str
    target: Any


@dataclass(frozen=True)
class CandidateProposal:
    action_id: str
    bbox: BBox
    tool_cost: float = 1.0
    pre_action_features: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelOutput:
    answer: str
    entropy: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


class VisualBackend(Protocol):
    """Model-specific bridge; implementations may wrap lmms-eval or another VLM."""

    def infer(
        self,
        *,
        image_path: str,
        question: str,
        bbox: BBox | None,
    ) -> ModelOutput: ...


CorrectnessScorer = Callable[[str, Any], float]


def collect_sibling_rollouts(
    samples: Sequence[TaskSample],
    *,
    proposals: Callable[[TaskSample], Sequence[CandidateProposal]],
    backend: VisualBackend,
    scorer: CorrectnessScorer,
) -> list[ActionRecord]:
    """Execute ANSWER and ZOOM siblings and normalize them to the project contract.

    Candidate proposal features are frozen before any candidate crop is sent to
    the backend. The collector deliberately never copies post-action metadata
    into ``pre_action_features``.
    """

    records: list[ActionRecord] = []
    for sample in samples:
        baseline = backend.infer(
            image_path=sample.image_path,
            question=sample.question,
            bbox=None,
        )
        correct_before = float(scorer(baseline.answer, sample.target))
        if not 0.0 <= correct_before <= 1.0:
            raise ValueError("scorer must return a value in [0, 1]")
        records.append(
            ActionRecord(
                state_id=sample.state_id,
                question=sample.question,
                original_image=sample.image_path,
                action_id="answer-now",
                action_type="ANSWER",
                candidate_bbox=None,
                entropy_before=baseline.entropy,
                entropy_after=baseline.entropy,
                answer_before=baseline.answer,
                answer_after=baseline.answer,
                correct_before=correct_before,
                correct_after=correct_before,
                tool_cost=0.0,
                metadata={"baseline_backend": dict(baseline.metadata)},
            )
        )
        candidate_actions = list(proposals(sample))
        if not candidate_actions:
            raise ValueError(f"proposal function returned no candidates for {sample.state_id!r}")
        action_ids: set[str] = {"answer-now"}
        for proposal in candidate_actions:
            if proposal.action_id in action_ids:
                raise ValueError(
                    f"duplicate action_id {proposal.action_id!r} in state {sample.state_id!r}"
                )
            action_ids.add(proposal.action_id)
            output = backend.infer(
                image_path=sample.image_path,
                question=sample.question,
                bbox=proposal.bbox,
            )
            correct_after = float(scorer(output.answer, sample.target))
            if not 0.0 <= correct_after <= 1.0:
                raise ValueError("scorer must return a value in [0, 1]")
            records.append(
                ActionRecord(
                    state_id=sample.state_id,
                    question=sample.question,
                    original_image=sample.image_path,
                    action_id=proposal.action_id,
                    action_type="ZOOM",
                    candidate_bbox=proposal.bbox,
                    entropy_before=baseline.entropy,
                    entropy_after=output.entropy,
                    answer_before=baseline.answer,
                    answer_after=output.answer,
                    correct_before=correct_before,
                    correct_after=correct_after,
                    tool_cost=proposal.tool_cost,
                    pre_action_features=dict(proposal.pre_action_features),
                    metadata={
                        "baseline_backend": dict(baseline.metadata),
                        "action_backend": dict(output.metadata),
                    },
                )
            )
    return records


def exact_match(answer: str, target: Any) -> float:
    normalized_answer = " ".join(answer.strip().casefold().split())
    normalized_target = " ".join(str(target).strip().casefold().split())
    return float(normalized_answer == normalized_target)
