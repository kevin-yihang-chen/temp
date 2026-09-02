from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Mapping, Sequence


PAIR_SCHEMA = "counterfactual_action_credit_pair_v1"
MASK_SCHEMA = "counterfactual_action_credit_masks_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_finite(name: str, value: float) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _require_sha256(name: str, value: str) -> str:
    normalized = str(value)
    if not _SHA256_RE.fullmatch(normalized):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


@dataclass(frozen=True)
class TokenSpan:
    """Half-open token span inside the valid (unpadded) response."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        if type(self.start) is not int or type(self.stop) is not int:
            raise ValueError("token span boundaries must be integers")
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("token span must be non-negative and have positive length")


@dataclass(frozen=True)
class TokenRoleMasks:
    """Auditable, exhaustive response-token roles for one trajectory."""

    action: tuple[int, ...]
    answer: tuple[int, ...]
    observation: tuple[int, ...]
    padding: tuple[int, ...]

    def __post_init__(self) -> None:
        lengths = {
            len(self.action),
            len(self.answer),
            len(self.observation),
            len(self.padding),
        }
        if len(lengths) != 1 or not self.action:
            raise ValueError("all token-role masks must have the same positive length")
        for name, mask in (
            ("action", self.action),
            ("answer", self.answer),
            ("observation", self.observation),
            ("padding", self.padding),
        ):
            if any(value not in (0, 1) for value in mask):
                raise ValueError(f"{name} mask must be binary")
        for index, roles in enumerate(
            zip(self.action, self.answer, self.observation, self.padding)
        ):
            if sum(roles) != 1:
                raise ValueError(
                    f"token role masks must partition the response at index {index}"
                )
        if not any(self.answer):
            raise ValueError(
                "every trainable trajectory must contain final-answer tokens"
            )
        if bool(any(self.action)) != bool(any(self.observation)):
            raise ValueError(
                "action and observation roles must either both be present or both be absent"
            )
        seen_padding = False
        for value in self.padding:
            if value:
                seen_padding = True
            elif seen_padding:
                raise ValueError("padding mask must be a contiguous response suffix")

    @property
    def response_length(self) -> int:
        return len(self.action)

    @property
    def valid_response_length(self) -> int:
        return self.response_length - sum(self.padding)

    @property
    def policy(self) -> tuple[int, ...]:
        return tuple(
            int(action or answer) for action, answer in zip(self.action, self.answer)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MASK_SCHEMA,
            "action": list(self.action),
            "answer": list(self.answer),
            "observation": list(self.observation),
            "padding": list(self.padding),
            "policy": list(self.policy),
            "response_length": self.response_length,
            "valid_response_length": self.valid_response_length,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TokenRoleMasks":
        if value.get("schema") != MASK_SCHEMA:
            raise ValueError(f"unsupported token-role schema: {value.get('schema')!r}")
        result = cls(
            action=tuple(int(item) for item in value["action"]),
            answer=tuple(int(item) for item in value["answer"]),
            observation=tuple(int(item) for item in value["observation"]),
            padding=tuple(int(item) for item in value["padding"]),
        )
        expected = result.to_dict()
        for field in ("policy", "response_length", "valid_response_length"):
            if field in value and value[field] != expected[field]:
                raise ValueError(
                    f"serialized {field} does not match reconstructed masks"
                )
        return result


def build_token_role_masks(
    *,
    response_length: int,
    valid_response_length: int,
    action_spans: Sequence[TokenSpan] = (),
    answer_spans: Sequence[TokenSpan] = (),
    observation_spans: Sequence[TokenSpan] = (),
) -> TokenRoleMasks:
    """Build exhaustive role masks, rejecting overlaps, gaps, and padding leakage."""

    if type(response_length) is not int or type(valid_response_length) is not int:
        raise ValueError("response lengths must be integers")
    if response_length <= 0:
        raise ValueError("response_length must be positive")
    if valid_response_length <= 0 or valid_response_length > response_length:
        raise ValueError("valid_response_length must be in [1, response_length]")
    roles: list[str | None] = [None] * valid_response_length

    def assign(name: str, spans: Sequence[TokenSpan]) -> None:
        for span in spans:
            if span.stop > valid_response_length:
                raise ValueError(f"{name} span extends beyond the valid response")
            for index in range(span.start, span.stop):
                if roles[index] is not None:
                    raise ValueError(
                        f"token role overlap at index {index}: {roles[index]} and {name}"
                    )
                roles[index] = name

    assign("action", action_spans)
    assign("observation", observation_spans)
    assign("answer", answer_spans)
    missing = [index for index, role in enumerate(roles) if role is None]
    if missing:
        raise ValueError(f"token role gap at valid indices {missing[:8]}")

    padding = [0] * valid_response_length + [1] * (
        response_length - valid_response_length
    )

    def mask_for(name: str) -> tuple[int, ...]:
        return tuple(int(role == name) for role in roles) + (0,) * (
            response_length - valid_response_length
        )

    return TokenRoleMasks(
        action=mask_for("action"),
        answer=mask_for("answer"),
        observation=mask_for("observation"),
        padding=tuple(padding),
    )


@dataclass(frozen=True)
class CounterfactualArmOutcome:
    """One fully identified continuation arm for a shared visual-action prefix."""

    branch_id: str
    prefix_sha256: str
    action_sha256: str
    observation_sha256: str
    target_sha256: str
    policy_sha256: str
    decoding_sha256: str
    scorer_sha256: str
    continuation_seed: int
    task_score: float
    action_cost: float

    def __post_init__(self) -> None:
        if not self.branch_id:
            raise ValueError("branch_id must be non-empty")
        for name in (
            "prefix_sha256",
            "action_sha256",
            "observation_sha256",
            "target_sha256",
            "policy_sha256",
            "decoding_sha256",
            "scorer_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if not isinstance(self.continuation_seed, int) or self.continuation_seed < 0:
            raise ValueError("continuation_seed must be a non-negative integer")
        task_score = _require_finite("task_score", self.task_score)
        action_cost = _require_finite("action_cost", self.action_cost)
        if not 0.0 <= task_score <= 1.0:
            raise ValueError("task_score must be in [0, 1]")
        if action_cost < 0.0:
            raise ValueError("action_cost must be non-negative")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CounterfactualArmOutcome":
        return cls(
            branch_id=str(value["branch_id"]),
            prefix_sha256=str(value["prefix_sha256"]),
            action_sha256=str(value["action_sha256"]),
            observation_sha256=str(value["observation_sha256"]),
            target_sha256=str(value["target_sha256"]),
            policy_sha256=str(value["policy_sha256"]),
            decoding_sha256=str(value["decoding_sha256"]),
            scorer_sha256=str(value["scorer_sha256"]),
            continuation_seed=int(value["continuation_seed"]),
            task_score=float(value["task_score"]),
            action_cost=float(value["action_cost"]),
        )


@dataclass(frozen=True)
class CounterfactualActionPair:
    """A factual/no-op pair whose net-utility contrast credits one action."""

    trajectory_id: str
    factual: CounterfactualArmOutcome
    counterfactual: CounterfactualArmOutcome
    lambda_cost: float = 0.05

    def __post_init__(self) -> None:
        if not self.trajectory_id:
            raise ValueError("trajectory_id must be non-empty")
        if self.factual.branch_id == self.counterfactual.branch_id:
            raise ValueError("factual and counterfactual branch_id must differ")
        lambda_cost = _require_finite("lambda_cost", self.lambda_cost)
        if lambda_cost < 0.0:
            raise ValueError("lambda_cost must be non-negative")
        matched_fields = (
            "prefix_sha256",
            "action_sha256",
            "target_sha256",
            "policy_sha256",
            "decoding_sha256",
            "scorer_sha256",
            "continuation_seed",
        )
        for field in matched_fields:
            if getattr(self.factual, field) != getattr(self.counterfactual, field):
                raise ValueError(f"counterfactual pair mismatch in {field}")

    @property
    def raw_score_effect(self) -> float:
        return self.factual.task_score - self.counterfactual.task_score

    @property
    def action_credit(self) -> float:
        factual_utility = (
            self.factual.task_score - self.lambda_cost * self.factual.action_cost
        )
        counterfactual_utility = (
            self.counterfactual.task_score
            - self.lambda_cost * self.counterfactual.action_cost
        )
        return factual_utility - counterfactual_utility

    def swapped(self) -> "CounterfactualActionPair":
        return CounterfactualActionPair(
            trajectory_id=self.trajectory_id,
            factual=self.counterfactual,
            counterfactual=self.factual,
            lambda_cost=self.lambda_cost,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PAIR_SCHEMA,
            "trajectory_id": self.trajectory_id,
            "factual": asdict(self.factual),
            "counterfactual": asdict(self.counterfactual),
            "lambda_cost": self.lambda_cost,
            "raw_score_effect": self.raw_score_effect,
            "action_credit": self.action_credit,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CounterfactualActionPair":
        if value.get("schema") != PAIR_SCHEMA:
            raise ValueError(f"unsupported action-pair schema: {value.get('schema')!r}")
        result = cls(
            trajectory_id=str(value["trajectory_id"]),
            factual=CounterfactualArmOutcome.from_dict(value["factual"]),
            counterfactual=CounterfactualArmOutcome.from_dict(value["counterfactual"]),
            lambda_cost=float(value["lambda_cost"]),
        )
        for field, expected in (
            ("raw_score_effect", result.raw_score_effect),
            ("action_credit", result.action_credit),
        ):
            if field in value and not math.isclose(
                float(value[field]), expected, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"serialized {field} does not match reconstructed pair"
                )
        return result


def compose_token_local_advantages(
    *,
    outcome_advantage: float,
    action_credit: float,
    masks: TokenRoleMasks,
    beta: float = 1.0,
) -> tuple[float, ...]:
    """Apply outcome credit only to answers and signed visual credit only to actions."""

    outcome = _require_finite("outcome_advantage", outcome_advantage)
    credit = _require_finite("action_credit", action_credit)
    multiplier = _require_finite("beta", beta)
    if multiplier < 0.0:
        raise ValueError("beta must be non-negative")
    if not any(masks.action) and credit != 0.0:
        raise ValueError(
            "a trajectory without action tokens must have zero action credit"
        )
    return tuple(
        outcome * answer + multiplier * credit * action
        for action, answer in zip(masks.action, masks.answer)
    )


@dataclass(frozen=True)
class ShuffledCreditAssignment:
    target_trajectory_id: str
    donor_trajectory_id: str
    action_credit: float


def cyclically_derange_action_credits(
    trajectory_ids: Sequence[str], action_credits: Sequence[float]
) -> tuple[ShuffledCreditAssignment, ...]:
    """Deterministically rotate credits over sorted ids with no self-donor."""

    if len(trajectory_ids) != len(action_credits):
        raise ValueError("trajectory_ids and action_credits must have the same length")
    if len(trajectory_ids) < 2:
        raise ValueError("at least two trajectories are required for shuffled credit")
    if any(not trajectory_id for trajectory_id in trajectory_ids):
        raise ValueError("trajectory ids must be non-empty")
    if len(set(trajectory_ids)) != len(trajectory_ids):
        raise ValueError("trajectory ids must be unique")
    credits = [_require_finite("action_credit", value) for value in action_credits]
    indexed = sorted(
        range(len(trajectory_ids)), key=lambda index: trajectory_ids[index]
    )
    donor_by_target: dict[int, int] = {}
    for offset, target_index in enumerate(indexed):
        donor_by_target[target_index] = indexed[(offset + 1) % len(indexed)]
    return tuple(
        ShuffledCreditAssignment(
            target_trajectory_id=trajectory_ids[target_index],
            donor_trajectory_id=trajectory_ids[donor_by_target[target_index]],
            action_credit=credits[donor_by_target[target_index]],
        )
        for target_index in range(len(trajectory_ids))
    )
