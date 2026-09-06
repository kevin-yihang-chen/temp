"""Paired utility labels with a separate, strictly outcome-free inference view."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from .dataset import group_by_decision, group_by_state, validate_sibling_groups
from .predictability_audit import SplitIdentity, audit_split_disjointness
from .rollout import AgentState
from .schema import ActionRecord, BBox
from .spatial_action_space import SpatialAction, SpatialActionSpace


@dataclass(frozen=True)
class UtilityInputs:
    """The ONLY public input to the trainable selector. Never tokenize IDs."""

    state: AgentState
    action_space: SpatialActionSpace

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state.state_id, "image_id": self.state.image_id,
            "source_id": self.state.source_id, "image_path": self.state.image_path,
            "question": self.state.question, "model_prompt": self.state.model_prompt,
            "actions": [
                {"index": a.index, "action_id": a.action_id,
                 "bbox": None if a.bbox is None else a.bbox.to_list(),
                 "visual_cost": a.visual_cost}
                for a in self.action_space.actions
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> UtilityInputs:
        expected = {"state_id", "image_id", "source_id", "image_path", "question", "model_prompt", "actions"}
        if set(value) != expected:
            raise ValueError("inference input keys differ from strict allowlist")
        actions = []
        for a in value["actions"]:
            if set(a) != {"index", "action_id", "bbox", "visual_cost"}:
                raise ValueError("action contains unexpected fields")
            if type(a["index"]) is not int:
                raise ValueError("action index must be an integer")
            actions.append(SpatialAction(a["index"], str(a["action_id"]), BBox.from_value(a["bbox"]), float(a["visual_cost"])))
        fields = ("state_id", "image_id", "source_id", "image_path", "question")
        if any(not isinstance(value[k], str) or not value[k].strip() for k in fields):
            raise ValueError("state input fields must be nonempty strings")
        prompt = value["model_prompt"]
        if prompt is not None and (not isinstance(prompt, str) or not prompt.strip()):
            raise ValueError("invalid model prompt")
        return cls(AgentState(**{k: value[k] for k in fields}, model_prompt=prompt), SpatialActionSpace(tuple(actions)))


@dataclass(frozen=True)
class UtilitySample:
    inputs: UtilityInputs
    benchmark: str
    role: str
    image_rgb_sha256: str
    rewards: tuple[float, ...]
    gains: tuple[float, ...]
    generation_seeds: tuple[int | None, ...]
    replicate_ids: tuple[str, ...]
    outcomes: tuple[ActionRecord, ...]

    def __post_init__(self) -> None:
        n = len(self.inputs.action_space.actions)
        if self.role not in ("train", "validation", "test") or not self.benchmark:
            raise ValueError("invalid dataset role/benchmark")
        if len(self.image_rgb_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.image_rgb_sha256):
            raise ValueError("decoded RGB SHA-256 required")
        if len(self.rewards) != n or len(self.gains) != n:
            raise ValueError("labels must match complete action support")
        if not all(math.isfinite(r) and 0 <= r <= 1 for r in self.rewards):
            raise ValueError("rewards must be finite official scores in [0,1]")
        if any(not math.isfinite(g) or abs(g - (r-self.rewards[0])) > 1e-9 for g, r in zip(self.gains, self.rewards)):
            raise ValueError("gain must be paired reward minus ANSWER, without cost")
        if not self.replicate_ids or len(self.replicate_ids) != len(self.generation_seeds):
            raise ValueError("replicate provenance required")
        if len(set(self.generation_seeds)) != len(self.generation_seeds):
            raise ValueError("duplicate seeds are not independent paired replicates")

    @property
    def best_action(self) -> int:
        return max(range(len(self.gains)), key=lambda i: (self.gains[i], -i))

    @property
    def support_action(self) -> int:
        # Gain-free null control, NOT a task-optimal target or an inference feature.
        key = f"utility-support-v1:{self.benchmark}:{self.inputs.state.state_id}"
        return int(hashlib.sha256(key.encode()).hexdigest(), 16) % len(self.gains)

    def to_dict(self) -> dict[str, Any]:
        return {
            "inputs": self.inputs.to_dict(), "benchmark": self.benchmark,
            "role": self.role, "image_rgb_sha256": self.image_rgb_sha256,
            "labels": {"reward": list(self.rewards), "gain": list(self.gains)},
            "generation_seeds": list(self.generation_seeds),
            "replicate_ids": list(self.replicate_ids),
            "outcomes": [r.to_dict() for r in self.outcomes],
        }


def build_utility_samples(
    records: Sequence[ActionRecord], *, states: Mapping[str, AgentState],
    rgb_hashes: Mapping[str, str], benchmark: str, role: str,
    aggregation: str = "single",
) -> list[UtilitySample]:
    """Aggregate paired replicates only AFTER grouping each full sibling set.

    Caller supplies manifest states so MCQ choices/model prompts are not lost.
    No ground truth is retained in the inference view; outcomes stay diagnostics.
    """
    if aggregation not in ("single", "mean"):
        raise ValueError("aggregation must be single or mean")
    validate_sibling_groups(records)
    result = []
    for state_id, rows in sorted(group_by_state(records).items()):
        state = states[state_id]
        exemplar = rows[0]
        if (state.image_id, state.source_id, state.question) != (exemplar.image_id, exemplar.source_id, exemplar.question):
            raise ValueError("manifest/rollout identity or question mismatch")
        if Path(state.image_path).resolve() != Path(exemplar.original_image).resolve():
            raise ValueError("manifest/rollout image mismatch")
        decisions = sorted(group_by_decision(rows).items())
        if aggregation == "single" and len(decisions) != 1:
            raise ValueError("single-seed MVP requires exactly one paired replicate")
        space = SpatialActionSpace.from_siblings(decisions[0][1])
        reward_vectors = []
        seeds, replicate_ids, ordered_outcomes = [], [], []
        for (_, replicate_id), siblings in decisions:
            if SpatialActionSpace.from_siblings(siblings) != space:
                raise ValueError("action mapping changed across seeds")
            by_id = {r.action_id: r for r in siblings}
            ordered = [by_id[a.action_id] for a in space.actions]
            baseline = ordered[0]
            if any(r.answer_before != baseline.answer_before for r in ordered):
                raise ValueError("paired siblings disagree on baseline answer")
            if baseline.answer_before != baseline.answer_after:
                raise ValueError("ANSWER must preserve baseline answer")
            reward_vectors.append([r.correct_after for r in ordered])
            seeds.append(baseline.generation_seed)
            replicate_ids.append(replicate_id)
            ordered_outcomes.extend(ordered)
        rewards = tuple(mean(v[i] for v in reward_vectors) for i in range(len(space.actions)))
        # Strip trajectories as well: only original state/prompt is authorized.
        clean_state = AgentState(state.state_id, state.image_id, state.source_id, state.image_path, state.question, model_prompt=state.model_prompt)
        result.append(UtilitySample(
            UtilityInputs(clean_state, space), benchmark, role, rgb_hashes[state_id],
            rewards, tuple(r-rewards[0] for r in rewards), tuple(seeds),
            tuple(replicate_ids), tuple(ordered_outcomes),
        ))
    return result


def audit_utility_splits(samples: Sequence[UtilitySample]) -> dict[str, Any]:
    identities, assignments = [], {}
    image_roles: dict[str, set[str]] = {}
    for sample in samples:
        state = sample.inputs.state
        item_id = f"{sample.benchmark}:{state.state_id}"
        if item_id in assignments:
            raise ValueError("state appears more than once (possibly across roles)")
        assignments[item_id] = sample.role
        identities.append(SplitIdentity(item_id, state.source_id, sample.image_rgb_sha256))
        image_roles.setdefault(f"{sample.benchmark}:{state.image_id}", set()).add(sample.role)
    if not samples or any(len(roles) > 1 for roles in image_roles.values()):
        raise ValueError("empty dataset or image-ID split leakage")
    return audit_split_disjointness(identities, assignments)


def load_utility_development(path: str | Path, *, role: str) -> list[UtilitySample]:
    """Training API deliberately has no test authorization or test-role option."""
    if role not in ("train", "validation"):
        raise ValueError("development loader cannot open test")
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != "utility_sft_dataset_v1" or payload.get("role") != role:
        raise ValueError("wrong utility dataset schema/role")
    samples = []
    for row in payload["samples"]:
        if row["role"] != role or row["benchmark"] != payload["benchmark"]:
            raise ValueError("sample role/benchmark mismatch")
        inputs = UtilityInputs.from_dict(row["inputs"])
        raw = [ActionRecord.from_dict(r) for r in row["outcomes"]]
        rebuilt = build_utility_samples(
            raw, states={inputs.state.state_id: inputs.state},
            rgb_hashes={inputs.state.state_id: row["image_rgb_sha256"]},
            benchmark=row["benchmark"], role=role, aggregation=payload["aggregation"],
        )
        if len(rebuilt) != 1 or rebuilt[0].to_dict() != row:
            raise ValueError("serialized labels/provenance differ from paired outcomes")
        samples.extend(rebuilt)
    audit_utility_splits(samples)
    return samples
