from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence

from .schema import ActionRecord, BBox


@dataclass(frozen=True)
class AgentState:
    """Information that an agent may legally use before taking an action."""

    state_id: str
    image_id: str
    source_id: str
    image_path: str
    question: str
    trajectory: tuple[str, ...] = ()
    model_prompt: str | None = None

    @property
    def backend_prompt(self) -> str:
        """Text shown to the VLM, distinct from gate-visible question context."""

        return self.model_prompt if self.model_prompt is not None else self.question


@dataclass(frozen=True)
class GroundTruth:
    """Evaluation-only payload that is never passed to proposal/model code."""

    target: Any


@dataclass(frozen=True)
class TaskExample:
    state: AgentState
    ground_truth: GroundTruth


@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    bbox: BBox
    visual_cost: float = 1.0
    pre_action_features: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("action_id must be non-empty")
        if self.visual_cost < 0.0:
            raise ValueError("visual_cost must be non-negative")


@dataclass(frozen=True)
class VisualObservation:
    """An observation shown to the backend; zooms supplement the original image."""

    kind: Literal["ORIGINAL", "ZOOM"]
    image_path: str
    action_id: str
    bbox: BBox | None

    def __post_init__(self) -> None:
        if self.kind == "ORIGINAL" and self.bbox is not None:
            raise ValueError("ORIGINAL observation must not have a bbox")
        if self.kind == "ZOOM" and self.bbox is None:
            raise ValueError("ZOOM observation requires a bbox")


@dataclass(frozen=True)
class ModelOutput:
    answer: str
    entropy: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


class VisualBackend(Protocol):
    """Bridge to a VLM agent that consumes an additive observation history."""

    def infer(
        self,
        *,
        state: AgentState,
        observations: Sequence[VisualObservation],
        generation_seed: int | None,
    ) -> ModelOutput: ...


@dataclass(frozen=True)
class InferenceRequest:
    state: AgentState
    observations: tuple[VisualObservation, ...]
    generation_seed: int | None

    def cache_key(self) -> str:
        payload = {
            "state": {
                "state_id": self.state.state_id,
                "image_id": self.state.image_id,
                "source_id": self.state.source_id,
                "image_path": self.state.image_path,
                "question": self.state.question,
                "trajectory": list(self.state.trajectory),
                "model_prompt": self.state.model_prompt,
            },
            "observations": [
                {
                    "kind": observation.kind,
                    "image_path": observation.image_path,
                    "action_id": observation.action_id,
                    "bbox": (
                        None
                        if observation.bbox is None
                        else observation.bbox.to_list()
                    ),
                }
                for observation in self.observations
            ],
            "generation_seed": self.generation_seed,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode()).hexdigest()


class BatchVisualBackend(Protocol):
    def infer_batch(self, requests: Sequence[InferenceRequest]) -> Sequence[ModelOutput]: ...


def infer_many(
    backend: VisualBackend | BatchVisualBackend,
    requests: Sequence[InferenceRequest],
) -> list[ModelOutput]:
    """Use backend batching when present, otherwise provide a serial fallback."""

    batch_method = getattr(backend, "infer_batch", None)
    if callable(batch_method):
        outputs = list(batch_method(requests))
    else:
        infer_method = getattr(backend, "infer", None)
        if not callable(infer_method):
            raise TypeError("backend must implement infer or infer_batch")
        outputs = [
            infer_method(
                state=request.state,
                observations=request.observations,
                generation_seed=request.generation_seed,
            )
            for request in requests
        ]
    if len(outputs) != len(requests):
        raise ValueError("backend returned a different number of outputs than requests")
    return outputs


class CachedVisualBackend:
    """In-memory cache wrapper keyed only by pre-action request content and seed."""

    def __init__(self, backend: VisualBackend | BatchVisualBackend) -> None:
        self.backend = backend
        self.cache: dict[str, ModelOutput] = {}

    def infer_batch(self, requests: Sequence[InferenceRequest]) -> list[ModelOutput]:
        keys = [request.cache_key() for request in requests]
        missing_positions = [
            position for position, key in enumerate(keys) if key not in self.cache
        ]
        if missing_positions:
            missing_requests = [requests[position] for position in missing_positions]
            missing_outputs = infer_many(self.backend, missing_requests)
            for position, output in zip(missing_positions, missing_outputs):
                self.cache[keys[position]] = output
        return [self.cache[key] for key in keys]

    def clear(self) -> None:
        self.cache.clear()


CorrectnessScorer = Callable[[str, GroundTruth], float]
ProposalFunction = Callable[[AgentState], Sequence[ActionSpec]]


def collect_sibling_rollouts(
    examples: Sequence[TaskExample],
    *,
    proposals: ProposalFunction,
    backend: VisualBackend | BatchVisualBackend,
    scorer: CorrectnessScorer,
    generation_seeds: Sequence[int | None] = (None,),
) -> list[ActionRecord]:
    """Execute paired ANSWER/ZOOM siblings without exposing ground truth.

    Proposals are generated once from ``AgentState``. For a ZOOM action the
    backend receives both the original observation and the new crop observation,
    making the operation additive information acquisition rather than image
    replacement. Repeated seeds produce paired stochastic replicates.
    """

    if not generation_seeds:
        raise ValueError("generation_seeds must contain at least one seed")
    records: list[ActionRecord] = []
    for example in examples:
        state = example.state
        candidate_actions = list(proposals(state))
        if not candidate_actions:
            raise ValueError(f"proposal function returned no candidates for {state.state_id!r}")
        action_ids: set[str] = {"answer-now"}
        for proposal in candidate_actions:
            if proposal.action_id in action_ids:
                raise ValueError(
                    f"duplicate action_id {proposal.action_id!r} in state {state.state_id!r}"
                )
            action_ids.add(proposal.action_id)

        original = VisualObservation(
            kind="ORIGINAL",
            image_path=state.image_path,
            action_id="original",
            bbox=None,
        )
        for replicate_number, generation_seed in enumerate(generation_seeds):
            replicate_id = f"replicate-{replicate_number:03d}"
            zoom_observations = [
                VisualObservation(
                    kind="ZOOM",
                    image_path=state.image_path,
                    action_id=proposal.action_id,
                    bbox=proposal.bbox,
                )
                for proposal in candidate_actions
            ]
            requests = [
                InferenceRequest(state, (original,), generation_seed),
                *(
                    InferenceRequest(state, (original, zoom), generation_seed)
                    for zoom in zoom_observations
                ),
            ]
            outputs = infer_many(backend, requests)
            baseline = outputs[0]
            correct_before = float(scorer(baseline.answer, example.ground_truth))
            if not 0.0 <= correct_before <= 1.0:
                raise ValueError("scorer must return a value in [0, 1]")
            records.append(
                ActionRecord(
                    state_id=state.state_id,
                    image_id=state.image_id,
                    source_id=state.source_id,
                    question=state.question,
                    original_image=state.image_path,
                    replicate_id=replicate_id,
                    generation_seed=generation_seed,
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
            for proposal, output in zip(candidate_actions, outputs[1:]):
                correct_after = float(scorer(output.answer, example.ground_truth))
                if not 0.0 <= correct_after <= 1.0:
                    raise ValueError("scorer must return a value in [0, 1]")
                records.append(
                    ActionRecord(
                        state_id=state.state_id,
                        image_id=state.image_id,
                        source_id=state.source_id,
                        question=state.question,
                        original_image=state.image_path,
                        replicate_id=replicate_id,
                        generation_seed=generation_seed,
                        action_id=proposal.action_id,
                        action_type="ZOOM",
                        candidate_bbox=proposal.bbox,
                        entropy_before=baseline.entropy,
                        entropy_after=output.entropy,
                        answer_before=baseline.answer,
                        answer_after=output.answer,
                        correct_before=correct_before,
                        correct_after=correct_after,
                        tool_cost=proposal.visual_cost,
                        pre_action_features=dict(proposal.pre_action_features),
                        metadata={
                            "baseline_backend": dict(baseline.metadata),
                            "action_backend": dict(output.metadata),
                        },
                    )
                )
    return records


def exact_match(answer: str, ground_truth: GroundTruth) -> float:
    normalized_answer = " ".join(answer.strip().casefold().split())
    normalized_target = " ".join(str(ground_truth.target).strip().casefold().split())
    return float(normalized_answer == normalized_target)
