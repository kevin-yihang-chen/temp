from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Callable, Literal, Mapping, Sequence

from .counterfactual_action_credit import (
    TokenSpan,
    build_token_role_masks,
    compose_token_local_advantages,
    cyclically_derange_action_credits,
)


CreditMode = Literal["signed", "zero", "shuffled"]

TRAJECTORY_ID_KEY = "vtool_action_credit_trajectory_id"
ACTION_TOKEN_COUNT_KEY = "vtool_action_token_count"
OBSERVATION_TOKEN_COUNT_KEY = "vtool_observation_token_count"
ANSWER_TOKEN_COUNT_KEY = "vtool_answer_token_count"
ACTION_CREDIT_KEY = "vtool_action_credit"
PAIR_VALID_KEY = "vtool_action_credit_pair_valid"


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def deterministic_rollout_seeds(trajectory_info: object) -> tuple[int, int]:
    digest = canonical_sha256(trajectory_info)
    action_seed = int(digest[:8], 16) % (2**31 - 2)
    return action_seed, action_seed + 1


def extract_vtool_answer(response: str) -> str:
    cleaned = response.replace("||", "|")
    patterns = (
        r"FINAL ANSWER:\s*(.*?)\s*TERMINATE",
        r"FINAL ANSWER:\s*(.*?)(?=\s*(?:\||$))",
        r"ANSWER:\s*(.*?)(?=\s*(?:\||$))",
    )
    for pattern in patterns:
        match = re.search(pattern, cleaned, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return cleaned.strip()


def _finite(name: str, value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(name: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result != value or result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def _scalar(value: Any) -> Any:
    """Unbox numpy/object scalar wrappers without importing numpy."""

    if hasattr(value, "shape") and getattr(value, "shape") == ():
        return value.item()
    if hasattr(value, "size") and getattr(value, "size") == 1:
        reshaped = value.reshape(-1)
        return reshaped[0]
    return value


@dataclass(frozen=True)
class ActionCreditTrajectory:
    """Role boundaries and signed credit exported by one paired agent rollout."""

    trajectory_id: str
    action_token_count: int
    observation_token_count: int
    answer_token_count: int
    action_credit: float
    pair_valid: bool

    def __post_init__(self) -> None:
        if not self.trajectory_id:
            raise ValueError("trajectory_id must be non-empty")
        action = _integer("action_token_count", self.action_token_count)
        observation = _integer("observation_token_count", self.observation_token_count)
        answer = _integer("answer_token_count", self.answer_token_count, minimum=1)
        credit = _finite("action_credit", self.action_credit)
        if bool(action) != bool(observation):
            raise ValueError(
                "action and observation token counts must both be present or absent"
            )
        if action and not self.pair_valid:
            raise ValueError("tool trajectories require a valid counterfactual pair")
        if not action and (self.pair_valid or credit != 0.0):
            raise ValueError(
                "no-tool trajectories require pair_valid=false and zero action credit"
            )

    @property
    def valid_response_length(self) -> int:
        return (
            self.action_token_count
            + self.observation_token_count
            + self.answer_token_count
        )

    @property
    def tool_attempted(self) -> bool:
        return self.action_token_count > 0

    def masks(self, response_length: int):
        action_stop = self.action_token_count
        observation_stop = action_stop + self.observation_token_count
        answer_stop = observation_stop + self.answer_token_count
        return build_token_role_masks(
            response_length=response_length,
            valid_response_length=self.valid_response_length,
            action_spans=(TokenSpan(0, action_stop),) if action_stop else (),
            observation_spans=(
                (TokenSpan(action_stop, observation_stop),)
                if self.observation_token_count
                else ()
            ),
            answer_spans=(TokenSpan(observation_stop, answer_stop),),
        )


@dataclass(frozen=True)
class PreparedActionCreditBatch:
    advantages: tuple[tuple[float, ...], ...]
    action_masks: tuple[tuple[int, ...], ...]
    answer_masks: tuple[tuple[int, ...], ...]
    policy_masks: tuple[tuple[int, ...], ...]
    applied_action_credits: tuple[float, ...]
    donor_trajectory_ids: tuple[str | None, ...]


def _credits_for_mode(
    trajectories: Sequence[ActionCreditTrajectory], mode: CreditMode
) -> tuple[tuple[float, ...], tuple[str | None, ...]]:
    if mode not in ("signed", "zero", "shuffled"):
        raise ValueError(f"unsupported action-credit mode: {mode!r}")
    if mode == "zero":
        return (tuple(0.0 for _ in trajectories), tuple(None for _ in trajectories))
    if mode == "signed":
        return (
            tuple(item.action_credit for item in trajectories),
            tuple(
                item.trajectory_id if item.tool_attempted else None
                for item in trajectories
            ),
        )

    tool_indices = [
        index for index, item in enumerate(trajectories) if item.tool_attempted
    ]
    assignments = cyclically_derange_action_credits(
        [trajectories[index].trajectory_id for index in tool_indices],
        [trajectories[index].action_credit for index in tool_indices],
    )
    credits = [0.0] * len(trajectories)
    donors: list[str | None] = [None] * len(trajectories)
    for target_index, assignment in zip(tool_indices, assignments, strict=True):
        credits[target_index] = assignment.action_credit
        donors[target_index] = assignment.donor_trajectory_id
    return tuple(credits), tuple(donors)


def prepare_action_credit_batch(
    *,
    trajectories: Sequence[ActionCreditTrajectory],
    outcome_advantages: Sequence[float],
    response_length: int,
    mode: CreditMode,
    beta: float = 1.0,
) -> PreparedActionCreditBatch:
    """Build token-local GRPO advantages before converting them to tensors."""

    if len(trajectories) != len(outcome_advantages):
        raise ValueError("trajectories and outcome_advantages must have equal length")
    if not trajectories:
        raise ValueError("action-credit batch must be non-empty")
    ids = [item.trajectory_id for item in trajectories]
    if len(ids) != len(set(ids)):
        raise ValueError("trajectory ids must be unique within an action-credit batch")
    credits, donors = _credits_for_mode(trajectories, mode)
    masks = [item.masks(response_length) for item in trajectories]
    advantages = tuple(
        compose_token_local_advantages(
            outcome_advantage=_finite("outcome_advantage", outcome),
            action_credit=credit,
            masks=mask,
            beta=beta,
        )
        for outcome, credit, mask in zip(
            outcome_advantages, credits, masks, strict=True
        )
    )
    return PreparedActionCreditBatch(
        advantages=advantages,
        action_masks=tuple(mask.action for mask in masks),
        answer_masks=tuple(mask.answer for mask in masks),
        policy_masks=tuple(mask.policy for mask in masks),
        applied_action_credits=credits,
        donor_trajectory_ids=donors,
    )


def trajectories_from_non_tensor_batch(
    non_tensor_batch: Mapping[str, Sequence[object]],
) -> tuple[ActionCreditTrajectory, ...]:
    required = (
        TRAJECTORY_ID_KEY,
        ACTION_TOKEN_COUNT_KEY,
        OBSERVATION_TOKEN_COUNT_KEY,
        ANSWER_TOKEN_COUNT_KEY,
        ACTION_CREDIT_KEY,
        PAIR_VALID_KEY,
    )
    missing = [key for key in required if key not in non_tensor_batch]
    if missing:
        raise ValueError(f"missing action-credit rollout fields: {missing}")
    lengths = {len(non_tensor_batch[key]) for key in required}
    if len(lengths) != 1:
        raise ValueError("action-credit rollout fields must have equal batch length")
    return tuple(
        ActionCreditTrajectory(
            trajectory_id=str(_scalar(non_tensor_batch[TRAJECTORY_ID_KEY][index])),
            action_token_count=_integer(
                ACTION_TOKEN_COUNT_KEY,
                _scalar(non_tensor_batch[ACTION_TOKEN_COUNT_KEY][index]),
            ),
            observation_token_count=_integer(
                OBSERVATION_TOKEN_COUNT_KEY,
                _scalar(non_tensor_batch[OBSERVATION_TOKEN_COUNT_KEY][index]),
            ),
            answer_token_count=_integer(
                ANSWER_TOKEN_COUNT_KEY,
                _scalar(non_tensor_batch[ANSWER_TOKEN_COUNT_KEY][index]),
                minimum=1,
            ),
            action_credit=_finite(
                ACTION_CREDIT_KEY,
                _scalar(non_tensor_batch[ACTION_CREDIT_KEY][index]),
            ),
            pair_valid=bool(_scalar(non_tensor_batch[PAIR_VALID_KEY][index])),
        )
        for index in range(next(iter(lengths)))
    )


def inject_token_local_action_credit(
    data: Any,
    *,
    mode: CreditMode,
    beta: float = 1.0,
    constant_atol: float = 1e-6,
) -> tuple[Any, dict[str, float]]:
    """Replace GRPO's broadcast advantage with answer/action-local advantages.

    This deliberately uses a duck-typed ``DataProto`` so the research package does
    not depend on the much larger pinned VTool/verl runtime. The rollout must have
    already exposed action tokens in ``response_mask``; otherwise this fails closed.
    """

    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised in minimal envs
        raise RuntimeError("PyTorch is required for the VTool runtime adapter") from exc

    for key in ("advantages", "returns", "response_mask"):
        if key not in data.batch.keys():
            raise ValueError(f"training batch is missing {key!r}")
    current = data.batch["advantages"]
    response_mask = data.batch["response_mask"]
    if current.ndim != 2 or response_mask.shape != current.shape:
        raise ValueError(
            "advantages and response_mask must be equal-shape rank-2 tensors"
        )

    trajectories = trajectories_from_non_tensor_batch(data.non_tensor_batch)
    if len(trajectories) != current.shape[0]:
        raise ValueError("rollout metadata batch size does not match tensor batch size")
    response_length = int(current.shape[1])
    role_masks = [item.masks(response_length) for item in trajectories]
    expected_policy = torch.as_tensor(
        [mask.policy for mask in role_masks],
        dtype=response_mask.dtype,
        device=response_mask.device,
    )
    if not torch.equal(response_mask, expected_policy):
        raise ValueError(
            "response_mask must equal action|answer; action tokens are still masked "
            "or observation/padding tokens leaked into the policy loss"
        )

    outcome_advantages: list[float] = []
    for row_index, mask in enumerate(role_masks):
        selected = current[row_index][
            torch.as_tensor(mask.policy, dtype=torch.bool, device=current.device)
        ]
        if selected.numel() == 0:
            raise ValueError("every rollout must have trainable policy tokens")
        scalar = selected[0]
        if not torch.allclose(
            selected,
            scalar.expand_as(selected),
            rtol=0.0,
            atol=constant_atol,
        ):
            raise ValueError(
                "incoming GRPO advantage must be constant over policy tokens"
            )
        outcome_advantages.append(float(scalar.detach().cpu().item()))

    prepared = prepare_action_credit_batch(
        trajectories=trajectories,
        outcome_advantages=outcome_advantages,
        response_length=response_length,
        mode=mode,
        beta=beta,
    )
    advantages = torch.as_tensor(
        prepared.advantages, dtype=current.dtype, device=current.device
    )
    data.batch["advantages"] = advantages
    data.batch["returns"] = advantages.clone()
    data.batch["action_mask"] = torch.as_tensor(
        prepared.action_masks, dtype=response_mask.dtype, device=response_mask.device
    )
    data.batch["answer_mask"] = torch.as_tensor(
        prepared.answer_masks, dtype=response_mask.dtype, device=response_mask.device
    )
    data.non_tensor_batch["vtool_action_credit_donor_trajectory_id"] = list(
        prepared.donor_trajectory_ids
    )
    tool_count = sum(item.tool_attempted for item in trajectories)
    metrics = {
        "action_credit/tool_trajectory_count": float(tool_count),
        "action_credit/tool_trajectory_rate": float(tool_count / len(trajectories)),
        "action_credit/mean_applied_credit": float(
            sum(prepared.applied_action_credits) / len(trajectories)
        ),
        "action_credit/mean_abs_applied_credit": float(
            sum(abs(value) for value in prepared.applied_action_credits)
            / len(trajectories)
        ),
    }
    return data, metrics


def wrap_verl_compute_advantage(
    original: Callable[..., Any], *, mode: CreditMode, beta: float = 1.0
) -> Callable[..., Any]:
    """Wrap pinned verl's GRPO function without vendoring or mutating upstream."""

    def wrapped(data: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(data, *args, **kwargs)
        result, metrics = inject_token_local_action_credit(result, mode=mode, beta=beta)
        result.meta_info["action_credit_metrics"] = metrics
        return result

    return wrapped
