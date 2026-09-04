from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
from statistics import mean
from typing import Any, Iterable, Literal, Mapping, Sequence

from .dataset import group_by_decision, validate_sibling_groups
from .policies import EntropySearchPolicy
from .schema import ActionRecord


AUDIT_BENCHMARKS = ("chartqa", "docvqa", "hrbench")
PREDICTOR_LEVELS = ("l0_uncertainty", "l1_shallow", "l2_semantic", "l3_frozen_qwen")
TARGET_FAMILIES = ("direct_gain", "rescue_harm", "factorized")
AUDIT_SEEDS = (17, 29, 47)
SplitRole = Literal["train", "validation", "test"]

_PRE_ACTION_FIELDS = frozenset(
    {
        "entropy_before",
        "max_probability",
        "top1_top2_margin",
        "shallow_question_features",
        "question_embedding",
        "global_visual_embedding",
        "pooled_language_state",
        "pooled_visual_state",
        "fused_multimodal_state",
    }
)
_FORBIDDEN_FEATURE_FRAGMENTS = (
    "answer_after",
    "correct_after",
    "entropy_after",
    "gain",
    "harm",
    "label",
    "outcome",
    "rescue",
    "success_after",
    "tool_answer",
    "y_tool",
)


def _finite_float(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_vector(value: Any, *, name: str) -> tuple[float, ...]:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().reshape(-1).tolist()
    elif hasattr(value, "tolist") and callable(value.tolist):
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a numeric sequence")
    result = tuple(_finite_float(item, name=name) for item in value)
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _validate_feature_name(name: str) -> None:
    lowered = name.lower()
    if any(fragment in lowered for fragment in _FORBIDDEN_FEATURE_FRAGMENTS):
        raise ValueError(
            f"post-action or target-derived feature is forbidden: {name!r}"
        )


@dataclass(frozen=True)
class PreActionInputs:
    """Typed, allowlisted view of information available before a tool call.

    The constructor deliberately accepts a nested ``pre_action`` object instead
    of a complete rollout/feature row. Unknown top-level fields are never copied.
    This permits safe adaptation of legacy bundles that also store labels while
    preventing those labels from reaching a predictor.
    """

    state_id: str
    image_id: str
    source_id: str
    entropy_before: float
    max_probability: float
    top1_top2_margin: float
    shallow_question_features: tuple[float, ...] | None = None
    question_embedding: tuple[float, ...] | None = None
    global_visual_embedding: tuple[float, ...] | None = None
    pooled_language_state: tuple[float, ...] | None = None
    pooled_visual_state: tuple[float, ...] | None = None
    fused_multimodal_state: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not self.state_id or not self.image_id or not self.source_id:
            raise ValueError("state_id, image_id, and source_id must be non-empty")
        for name in ("entropy_before", "max_probability", "top1_top2_margin"):
            _finite_float(getattr(self, name), name=name)
        if self.entropy_before < 0.0:
            raise ValueError("entropy_before must be non-negative")
        if not 0.0 <= self.max_probability <= 1.0:
            raise ValueError("max_probability must be in [0, 1]")
        if not 0.0 <= self.top1_top2_margin <= 1.0:
            raise ValueError("top1_top2_margin must be in [0, 1]")
        for name in _PRE_ACTION_FIELDS - {
            "entropy_before",
            "max_probability",
            "top1_top2_margin",
        }:
            value = getattr(self, name)
            if value is not None:
                _finite_vector(value, name=name)

    @classmethod
    def from_untrusted_mapping(cls, value: Mapping[str, Any]) -> "PreActionInputs":
        """Copy only the frozen pre-action allowlist from an untrusted row."""

        raw = value.get("pre_action")
        if not isinstance(raw, Mapping):
            raise ValueError("row must contain a pre_action mapping")
        unknown = set(raw) - _PRE_ACTION_FIELDS
        if unknown:
            raise ValueError(f"unknown pre_action fields: {sorted(unknown)}")
        for name in raw:
            _validate_feature_name(str(name))
        required = {"entropy_before", "max_probability", "top1_top2_margin"}
        missing = required - set(raw)
        if missing:
            raise ValueError(f"missing L0 fields: {sorted(missing)}")

        vectors: dict[str, tuple[float, ...] | None] = {}
        for name in _PRE_ACTION_FIELDS - required:
            item = raw.get(name)
            vectors[name] = None if item is None else _finite_vector(item, name=name)
        return cls(
            state_id=str(value["state_id"]),
            image_id=str(value["image_id"]),
            source_id=str(value["source_id"]),
            entropy_before=_finite_float(raw["entropy_before"], name="entropy_before"),
            max_probability=_finite_float(
                raw["max_probability"], name="max_probability"
            ),
            top1_top2_margin=_finite_float(
                raw["top1_top2_margin"], name="top1_top2_margin"
            ),
            **vectors,
        )

    def feature_vector(self, level: str) -> tuple[float, ...]:
        if level not in PREDICTOR_LEVELS:
            raise ValueError(f"unsupported predictor level: {level}")
        uncertainty = (
            self.entropy_before,
            self.max_probability,
            self.top1_top2_margin,
        )
        if level == "l0_uncertainty":
            return uncertainty
        if level == "l1_shallow":
            if self.shallow_question_features is None:
                raise ValueError("L1 requires shallow_question_features")
            return uncertainty + self.shallow_question_features
        if level == "l2_semantic":
            question = self.question_embedding
            visual = self.global_visual_embedding
            if question is None or visual is None:
                raise ValueError("L2 requires question and global visual embeddings")
            if len(question) != len(visual):
                raise ValueError(
                    "L2 question and visual embeddings must have equal size"
                )
            interaction = tuple(left * right for left, right in zip(question, visual))
            return uncertainty + question + visual + interaction
        states = (
            self.pooled_language_state,
            self.pooled_visual_state,
            self.fused_multimodal_state,
        )
        if any(item is None for item in states):
            raise ValueError(
                "L3 requires language, visual, and fused multimodal states"
            )
        language, visual, fused = states
        assert language is not None and visual is not None and fused is not None
        return uncertainty + language + visual + fused

    def to_feature_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if any(
            fragment in key.lower()
            for key in result
            for fragment in _FORBIDDEN_FEATURE_FRAGMENTS
        ):
            raise AssertionError(
                "typed pre-action view unexpectedly contains a forbidden key"
            )
        return result


@dataclass(frozen=True)
class BinaryToolOutcome:
    state_id: str
    replicate_id: str
    image_id: str
    source_id: str
    selected_action_id: str
    y0: float
    y_tool: float
    tool_cost: float
    tool_calls: int

    def __post_init__(self) -> None:
        if not all(
            (
                self.state_id,
                self.replicate_id,
                self.image_id,
                self.source_id,
                self.selected_action_id,
            )
        ):
            raise ValueError("binary tool outcome identities must be non-empty")
        for name in ("y0", "y_tool"):
            value = _finite_float(getattr(self, name), name=name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        cost = _finite_float(self.tool_cost, name="tool_cost")
        if cost < 0.0:
            raise ValueError("tool_cost must be non-negative")
        if not isinstance(self.tool_calls, int) or self.tool_calls < 0:
            raise ValueError("tool_calls must be a non-negative integer")
        if self.tool_calls == 0 and cost != 0.0:
            raise ValueError("zero-call outcome must have zero tool cost")

    @property
    def decision_id(self) -> tuple[str, str]:
        return self.state_id, self.replicate_id

    @property
    def gain(self) -> float:
        return self.y_tool - self.y0

    @property
    def rescue(self) -> bool:
        return self.gain > 0.0

    @property
    def harm(self) -> bool:
        return self.gain < 0.0

    def incremental_utility(self, lambda_cost: float) -> float:
        if lambda_cost < 0.0:
            raise ValueError("lambda_cost must be non-negative")
        return self.gain - lambda_cost * self.tool_cost


@dataclass(frozen=True)
class SplitIdentity:
    item_id: str
    source_id: str
    image_rgb_sha256: str

    def __post_init__(self) -> None:
        if not self.item_id or not self.source_id:
            raise ValueError("split item_id and source_id must be non-empty")
        if len(self.image_rgb_sha256) != 64:
            raise ValueError("image_rgb_sha256 must contain 64 hexadecimal characters")
        try:
            int(self.image_rgb_sha256, 16)
        except ValueError as exc:
            raise ValueError("image_rgb_sha256 is not hexadecimal") from exc


class _DisjointSets:
    def __init__(self, item_ids: Iterable[str]) -> None:
        self.parent = {item_id: item_id for item_id in item_ids}

    def find(self, item_id: str) -> str:
        parent = self.parent[item_id]
        if parent != item_id:
            self.parent[item_id] = self.find(parent)
        return self.parent[item_id]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def assign_disjoint_split_roles(
    identities: Sequence[SplitIdentity],
    *,
    seed: int,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> tuple[dict[str, SplitRole], dict[str, Any]]:
    """Assign connected source/RGB components to deterministic data roles."""

    if not identities:
        raise ValueError("split assignment requires at least one identity")
    if len({item.item_id for item in identities}) != len(identities):
        raise ValueError("split item_id values must be unique")
    if len(fractions) != 3 or any(value <= 0.0 for value in fractions):
        raise ValueError("train/validation/test fractions must be positive")
    if not math.isclose(sum(fractions), 1.0, abs_tol=1e-12):
        raise ValueError("train/validation/test fractions must sum to one")

    sets = _DisjointSets(item.item_id for item in identities)
    by_source: dict[str, str] = {}
    by_rgb: dict[str, str] = {}
    for item in identities:
        for key, registry in (
            (item.source_id, by_source),
            (item.image_rgb_sha256, by_rgb),
        ):
            previous = registry.get(key)
            if previous is None:
                registry[key] = item.item_id
            else:
                sets.union(previous, item.item_id)
    components: dict[str, list[SplitIdentity]] = defaultdict(list)
    for item in identities:
        components[sets.find(item.item_id)].append(item)
    if len(components) < 3:
        raise ValueError("at least three disjoint source/RGB components are required")

    roles: tuple[SplitRole, ...] = ("train", "validation", "test")
    targets = {
        role: fraction * len(identities) for role, fraction in zip(roles, fractions)
    }
    counts = {role: 0 for role in roles}
    component_values = list(components.values())
    component_values.sort(
        key=lambda items: hashlib.sha256(
            f"{seed}:{min(item.item_id for item in items)}".encode("utf-8")
        ).hexdigest()
    )
    assignments: dict[str, SplitRole] = {}
    for index, items in enumerate(component_values):
        unfilled = [role for role in roles if counts[role] == 0]
        remaining_components = len(component_values) - index
        if unfilled and remaining_components == len(unfilled):
            selected_role = unfilled[0]
        else:
            selected_role = max(
                roles,
                key=lambda role: (
                    (targets[role] - counts[role]) / targets[role],
                    -roles.index(role),
                ),
            )
        for item in items:
            assignments[item.item_id] = selected_role
        counts[selected_role] += len(items)

    audit = audit_split_disjointness(identities, assignments)
    audit.update(
        {
            "seed": seed,
            "fractions": dict(zip(roles, fractions)),
            "connected_components": len(components),
            "role_counts": counts,
        }
    )
    return assignments, audit


def audit_split_disjointness(
    identities: Sequence[SplitIdentity], assignments: Mapping[str, SplitRole]
) -> dict[str, Any]:
    expected_ids = {item.item_id for item in identities}
    if set(assignments) != expected_ids:
        missing = sorted(expected_ids - set(assignments))
        extra = sorted(set(assignments) - expected_ids)
        raise ValueError(
            f"split assignment identity mismatch; missing={missing[:3]}, extra={extra[:3]}"
        )
    roles: tuple[SplitRole, ...] = ("train", "validation", "test")
    if set(assignments.values()) - set(roles):
        raise ValueError("split assignments contain an unsupported role")
    sources: dict[SplitRole, set[str]] = {role: set() for role in roles}
    images: dict[SplitRole, set[str]] = {role: set() for role in roles}
    for item in identities:
        role = assignments[item.item_id]
        sources[role].add(item.source_id)
        images[role].add(item.image_rgb_sha256)
    overlaps: dict[str, dict[str, int]] = {}
    for left_index, left in enumerate(roles):
        for right in roles[left_index + 1 :]:
            overlaps[f"{left}_vs_{right}"] = {
                "source_overlap": len(sources[left] & sources[right]),
                "image_rgb_sha256_overlap": len(images[left] & images[right]),
            }
    passed = all(value == 0 for pair in overlaps.values() for value in pair.values())
    if not passed:
        raise ValueError(f"source/RGB split leakage detected: {overlaps}")
    return {
        "schema": "predictability_split_disjointness_v1",
        "passed": True,
        "pairwise_overlaps": overlaps,
    }


def collapse_fixed_entropy_tool(
    records: Iterable[ActionRecord],
    *,
    expected_zoom_actions: int = 4,
) -> list[BinaryToolOutcome]:
    """Collapse siblings to the fixed ANSWER_NOW versus exhaustive-UG task.

    ``Y_tool`` is the outcome of evaluating every registered crop and selecting
    the lowest post-action entropy, with action ID as the deterministic tie
    breaker. The predictor itself never receives any post-action entropy.
    """

    materialized = list(records)
    validate_sibling_groups(materialized)
    result: list[BinaryToolOutcome] = []
    for (state_id, replicate_id), siblings in sorted(
        group_by_decision(materialized).items()
    ):
        answer = next(record for record in siblings if record.action_type == "ANSWER")
        zooms = [record for record in siblings if record.action_type == "ZOOM"]
        if len(zooms) != expected_zoom_actions:
            raise ValueError(
                f"decision {(state_id, replicate_id)!r} has {len(zooms)} ZOOM actions; "
                f"expected {expected_zoom_actions}"
            )
        decision = EntropySearchPolicy().select(siblings)
        result.append(
            BinaryToolOutcome(
                state_id=state_id,
                replicate_id=replicate_id,
                image_id=answer.image_id,
                source_id=answer.source_id,
                selected_action_id=decision.selected.action_id,
                y0=answer.correct_before,
                y_tool=decision.selected.correct_after,
                tool_cost=decision.visual_cost,
                tool_calls=decision.tool_calls,
            )
        )
    return result


def _source_balanced_mean(
    outcomes: Sequence[BinaryToolOutcome], values: Sequence[float]
) -> float:
    if not outcomes or len(outcomes) != len(values):
        raise ValueError("source-balanced mean requires aligned non-empty values")
    by_source: dict[str, list[float]] = defaultdict(list)
    for outcome, value in zip(outcomes, values):
        by_source[outcome.source_id].append(float(value))
    return mean(mean(items) for items in by_source.values())


def fixed_tool_headroom_summary(
    outcomes: Sequence[BinaryToolOutcome], *, lambda_cost: float
) -> dict[str, Any]:
    """Report fixed-tool headroom without fitting or tuning a predictor."""

    if lambda_cost < 0.0:
        raise ValueError("lambda_cost must be non-negative")
    utilities = [outcome.incremental_utility(lambda_cost) for outcome in outcomes]
    oracle_calls = [utility > 0.0 for utility in utilities]
    oracle_values = [
        utility if call else 0.0 for utility, call in zip(utilities, oracle_calls)
    ]
    return {
        "decisions": len(outcomes),
        "sources": len({outcome.source_id for outcome in outcomes}),
        "lambda_cost": lambda_cost,
        "mean_tool_cost": mean(outcome.tool_cost for outcome in outcomes),
        "always_call": {
            "utility": _source_balanced_mean(outcomes, utilities),
            "call_rate": 1.0,
        },
        "privileged_binary_oracle": {
            "utility": _source_balanced_mean(outcomes, oracle_values),
            "call_rate": _source_balanced_mean(
                outcomes, [float(value) for value in oracle_calls]
            ),
        },
        "raw_targets": {
            "gain": _source_balanced_mean(outcomes, [item.gain for item in outcomes]),
            "rescue_rate": _source_balanced_mean(
                outcomes, [float(item.rescue) for item in outcomes]
            ),
            "harm_rate": _source_balanced_mean(
                outcomes, [float(item.harm) for item in outcomes]
            ),
        },
    }


def expected_matrix_cells() -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (benchmark, level, target)
        for benchmark in AUDIT_BENCHMARKS
        for level in PREDICTOR_LEVELS
        for target in TARGET_FAMILIES
    )


def matrix_completion_report(
    completed: Iterable[tuple[str, str, str]],
) -> dict[str, Any]:
    expected = set(expected_matrix_cells())
    supplied = list(completed)
    completed_set = set(supplied)
    if len(completed_set) != len(supplied):
        raise ValueError("duplicate matrix cells are not allowed")
    unexpected = sorted(completed_set - expected)
    if unexpected:
        raise ValueError(f"unexpected matrix cells: {unexpected}")
    missing = sorted(expected - completed_set)
    return {
        "schema": "predictability_matrix_completion_v1",
        "expected_cells": len(expected),
        "completed_cells": len(completed_set),
        "complete": not missing,
        "missing": [
            {"benchmark": benchmark, "predictor_level": level, "target": target}
            for benchmark, level, target in missing
        ],
    }


class AuditVerdict(str, Enum):
    GO = "GO"
    PIVOT = "PIVOT"
    REPRESENTATION = "REPRESENTATION"
    STOP = "STOP"


@dataclass(frozen=True)
class BenchmarkVerdictEvidence:
    benchmark: str
    oracle_utility: float
    primary_deployable_beats_strongest_baseline_lower_ci: float
    maximum_lower_ci_across_all_deployable_policies: float
    deployable_accuracy_cost_pareto: bool
    deployable_rescue_precision_higher: bool
    deployable_harm_rate_not_higher: bool
    post_action_probe_utility_lower_ci: float
    l3_in_domain_improvement_lower_ci: float
    l3_image_or_cross_domain_improvement_upper_ci: float


def classify_completed_audit(
    evidence: Sequence[BenchmarkVerdictEvidence],
    *,
    small_oracle_utility: float = 0.005,
) -> AuditVerdict:
    """Apply the frozen, mutually exclusive final decision hierarchy.

    STOP has priority when the task lacks headroom; otherwise GO has priority
    when deployable evidence succeeds. REPRESENTATION is the specific failure
    case before the more general PIVOT diagnosis. An inconclusive audit raises
    instead of manufacturing a verdict.
    """

    if {item.benchmark for item in evidence} != set(AUDIT_BENCHMARKS):
        raise ValueError(
            "final classification requires exactly the three frozen benchmarks"
        )
    if len(evidence) != len(AUDIT_BENCHMARKS):
        raise ValueError("duplicate benchmark evidence is not allowed")
    count = lambda predicate: sum(bool(predicate(item)) for item in evidence)
    if count(lambda item: item.oracle_utility <= small_oracle_utility) >= 2:
        return AuditVerdict.STOP
    if (
        count(
            lambda item: (
                item.primary_deployable_beats_strongest_baseline_lower_ci > 0.0
            )
            and item.deployable_accuracy_cost_pareto
            and item.deployable_rescue_precision_higher
            and item.deployable_harm_rate_not_higher
        )
        >= 2
    ):
        return AuditVerdict.GO
    if (
        count(lambda item: item.l3_in_domain_improvement_lower_ci > 0.0) >= 2
        and count(
            lambda item: item.l3_image_or_cross_domain_improvement_upper_ci <= 0.0
        )
        >= 2
    ):
        return AuditVerdict.REPRESENTATION
    if (
        count(lambda item: item.oracle_utility > small_oracle_utility) >= 2
        and count(lambda item: item.post_action_probe_utility_lower_ci > 0.0) >= 2
        and count(
            lambda item: item.maximum_lower_ci_across_all_deployable_policies <= 0.0
        )
        == 3
    ):
        return AuditVerdict.PIVOT
    raise ValueError("completed matrix does not support one of the frozen verdicts")
