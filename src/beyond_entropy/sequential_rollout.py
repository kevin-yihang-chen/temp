"""Counterfactual STOP/CONTINUE collection from a shared visual prefix."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Sequence

from .crops import UGGridProposer
from .rollout import (
    ActionSpec,
    AgentState,
    BatchVisualBackend,
    CorrectnessScorer,
    InferenceRequest,
    TaskExample,
    VisualBackend,
    VisualObservation,
    infer_many,
)
from .sequential_schema import AcquiredObservationSpec, SequentialRolloutRecord


@dataclass(frozen=True)
class SequentialPrefix:
    """Deployable stopping state and its single fixed next acquisition."""

    acquired: tuple[ActionSpec, ...]
    proposed: ActionSpec
    proposer_id: str

    def __post_init__(self) -> None:
        if not self.acquired:
            raise ValueError("sequential prefix requires partial visual evidence")
        ids = [item.action_id for item in self.acquired]
        if self.proposed.action_id in ids or len(ids) != len(set(ids)):
            raise ValueError("prefix actions must be unique")
        if not self.proposer_id:
            raise ValueError("proposer_id must be non-empty")


PrefixFunction = Callable[[AgentState], SequentialPrefix]


@dataclass(frozen=True)
class FixedOppositeUGPrefix:
    """Outcome-blind one-crop prefix followed by its farthest UG crop.

    A stable hash spreads the already-acquired crop over spatial positions.  The
    next crop is fixed by geometry only, so this experiment studies *when* to
    acquire and contains no learned or exhaustive candidate-ranking decision.
    """

    candidate_count: int = 4
    visual_crop_ratio: float = 2.0
    visual_cost: float = 1.0
    namespace: str = "sequential-opposite-ug-v1"

    def __post_init__(self) -> None:
        if self.candidate_count < 2:
            raise ValueError("sequential prefix requires at least two UG actions")
        if not self.namespace:
            raise ValueError("namespace must be non-empty")

    @staticmethod
    def _center(action: ActionSpec) -> tuple[float, float]:
        return (
            (action.bbox.x1 + action.bbox.x2) / 2.0,
            (action.bbox.y1 + action.bbox.y2) / 2.0,
        )

    def __call__(self, state: AgentState) -> SequentialPrefix:
        actions = UGGridProposer(
            candidate_count=self.candidate_count,
            visual_crop_ratio=self.visual_crop_ratio,
            visual_cost=self.visual_cost,
        )(state)
        digest = hashlib.sha256(
            f"{self.namespace}:{state.state_id}".encode("utf-8")
        ).digest()
        acquired = actions[int.from_bytes(digest[:8], "big") % len(actions)]
        x0, y0 = self._center(acquired)
        proposed = min(
            (item for item in actions if item.action_id != acquired.action_id),
            key=lambda item: (
                -((self._center(item)[0] - x0) ** 2 + (self._center(item)[1] - y0) ** 2),
                item.action_id,
            ),
        )
        return SequentialPrefix((acquired,), proposed, self.namespace)


def prefix_observations(
    state: AgentState, prefix: SequentialPrefix
) -> tuple[VisualObservation, ...]:
    return (
        VisualObservation("ORIGINAL", state.image_path, "original", None),
        *(
            VisualObservation("ZOOM", state.image_path, item.action_id, item.bbox)
            for item in prefix.acquired
        ),
    )


def _confidence(metadata: object, name: str) -> float:
    if not isinstance(metadata, dict) and not hasattr(metadata, "get"):
        raise ValueError("backend metadata must be a mapping")
    value = metadata.get(name)  # type: ignore[union-attr]
    if value is None:
        raise ValueError(f"backend metadata is missing {name}")
    return float(value)


def collect_counterfactual_prefixes(
    examples: Sequence[TaskExample],
    *,
    prefixes: PrefixFunction,
    backend: VisualBackend | BatchVisualBackend,
    scorer: CorrectnessScorer,
    generation_seeds: Sequence[int | None] = (0,),
) -> list[SequentialRolloutRecord]:
    """Execute paired STOP/CONTINUE branches with identical prefix and seed."""

    if not generation_seeds or len(set(generation_seeds)) != len(generation_seeds):
        raise ValueError("generation seeds must be non-empty and unique")
    records: list[SequentialRolloutRecord] = []
    for example in examples:
        # GroundTruth is intentionally unavailable to the prefix function.
        prefix = prefixes(example.state)
        current = prefix_observations(example.state, prefix)
        proposed = VisualObservation(
            "ZOOM",
            example.state.image_path,
            prefix.proposed.action_id,
            prefix.proposed.bbox,
        )
        for replicate_number, seed in enumerate(generation_seeds):
            stop_request = InferenceRequest(example.state, current, seed)
            continue_request = InferenceRequest(example.state, current + (proposed,), seed)
            stop, continued = infer_many(backend, (stop_request, continue_request))
            stop_correct = float(scorer(stop.answer, example.ground_truth))
            continue_correct = float(scorer(continued.answer, example.ground_truth))
            if not 0.0 <= stop_correct <= 1.0 or not 0.0 <= continue_correct <= 1.0:
                raise ValueError("scorer must return values in [0, 1]")
            records.append(
                SequentialRolloutRecord(
                    state_id=example.state.state_id,
                    image_id=example.state.image_id,
                    source_id=example.state.source_id,
                    question=example.state.question,
                    original_image=example.state.image_path,
                    step_index=len(prefix.acquired),
                    acquired_observations=tuple(
                        AcquiredObservationSpec(
                            item.action_id, item.bbox, item.visual_cost
                        )
                        for item in prefix.acquired
                    ),
                    proposed_action_id=prefix.proposed.action_id,
                    proposed_bbox=prefix.proposed.bbox,
                    proposed_visual_cost=prefix.proposed.visual_cost,
                    replicate_id=f"replicate-{replicate_number:03d}",
                    generation_seed=seed,
                    stop_answer=stop.answer,
                    stop_correct=stop_correct,
                    stop_entropy=stop.entropy,
                    stop_max_probability=_confidence(
                        stop.metadata, "mean_maximum_token_probability"
                    ),
                    stop_top1_top2_margin=_confidence(
                        stop.metadata, "mean_top1_top2_token_probability_margin"
                    ),
                    continue_answer=continued.answer,
                    continue_correct=continue_correct,
                    continue_entropy=continued.entropy,
                    continue_max_probability=_confidence(
                        continued.metadata, "mean_maximum_token_probability"
                    ),
                    continue_top1_top2_margin=_confidence(
                        continued.metadata,
                        "mean_top1_top2_token_probability_margin",
                    ),
                    stop_backend=dict(stop.metadata),
                    continue_backend=dict(continued.metadata),
                    metadata={
                        "proposer_id": prefix.proposer_id,
                        "paired_prefix_cache_key": stop_request.cache_key(),
                        "stop_observation_count": len(current),
                        "continue_observation_count": len(current) + 1,
                    },
                )
            )
    return records
