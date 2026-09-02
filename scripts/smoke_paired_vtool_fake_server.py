#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

from beyond_entropy.counterfactual_action_credit import CounterfactualActionPair
from beyond_entropy.refocus_chart_audit import canonical_sha256
from beyond_entropy.vtool_action_credit import (
    ACTION_CREDIT_KEY,
    ACTION_TOKEN_COUNT_KEY,
    ANSWER_TOKEN_COUNT_KEY,
    OBSERVATION_TOKEN_COUNT_KEY,
    PAIR_VALID_KEY,
    deterministic_rollout_seeds,
)
from integrations.vtool_action_credit.paired_vtool import (
    CounterfactualCreditVToolAgentLoop,
    ROLLOUT_AUDIT_FIELDS,
    ROLLOUT_AUDIT_JSON_KEY,
    ROLLOUT_AUDIT_SCHEMA,
)


REPORT_SCHEMA = "paired_vtool_fake_server_contract_v1"
ACTION_IDS = [10, 11]
OBSERVATION_IDS = [40, 41]
FACTUAL_ANSWER_IDS = [20, 21]
COUNTERFACTUAL_ANSWER_IDS = [30, 31]
DIRECT_ANSWER_IDS = [50]


@dataclass(frozen=True)
class FakeTokenOutput:
    token_ids: list[int]
    log_probs: list[float]


class FakeTokenizer:
    def __init__(self, *, direct: bool = False) -> None:
        self.direct = direct

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        del skip_special_tokens
        if token_ids == ACTION_IDS:
            return (
                "```python\n"
                "focused = focus_on_y_values_with_highlight("
                "image_1, ['Alpha'], y_values_bbox)\n"
                "display(focused)\n```"
            )
        if token_ids == FACTUAL_ANSWER_IDS:
            return "FINAL ANSWER: 7 TERMINATE"
        if token_ids == COUNTERFACTUAL_ANSWER_IDS:
            return "FINAL ANSWER: 3 TERMINATE"
        if token_ids == DIRECT_ANSWER_IDS:
            return "FINAL ANSWER: 7 TERMINATE"
        raise AssertionError(f"unexpected fake token ids: {token_ids}")


class FakeCodeParser:
    def __init__(self, *, direct: bool = False) -> None:
        self.direct = direct

    def parse(self, text: str) -> Any:
        del text
        if self.direct:
            return SimpleNamespace(error_code="NOTOOL", status=False, code="")
        return SimpleNamespace(error_code="OK", status=True, code="display(image_1)")


class InlineExecutorLoop:
    """Resolve tiny fake decode calls without leaving executor threads behind."""

    def run_in_executor(self, executor: Any, function: Any) -> asyncio.Future[Any]:
        del executor
        future = asyncio.get_running_loop().create_future()
        try:
            future.set_result(function())
        except Exception as exc:
            future.set_exception(exc)
        return future


def _pixel_key(image: Image.Image) -> tuple[int, ...]:
    pixel = image.convert("RGB").getpixel((0, 0))
    if not isinstance(pixel, tuple):
        raise TypeError("RGB pixel must be a tuple")
    return tuple(int(value) for value in pixel)


def _make_agent(
    *,
    mode: str,
    mismatched_observation_ids: bool = False,
) -> tuple[CounterfactualCreditVToolAgentLoop, list[dict[str, Any]]]:
    agent = object.__new__(CounterfactualCreditVToolAgentLoop)
    original = Image.new("RGB", (8, 8), (10, 20, 30))
    edited = Image.new("RGB", (8, 8), (200, 10, 10))
    calls: list[dict[str, Any]] = []
    agent.processor = object()
    agent.response_length = 32
    agent.loop = InlineExecutorLoop()
    agent.tokenizer = FakeTokenizer(direct=mode == "direct")
    agent.code_parser = FakeCodeParser(direct=mode == "direct")
    agent._policy_sha256 = lambda: "a" * 64  # type: ignore[method-assign]

    async def process_vision_info(messages: list[dict[str, Any]]) -> dict[str, Any]:
        del messages
        return {"images": [original], "videos": []}

    async def apply_chat_template(
        messages: list[dict[str, Any]],
        *,
        images: list[Image.Image],
        videos: list[Any] | None = None,
        remove_system_prompt: bool = False,
    ) -> list[int]:
        del messages, videos
        if not remove_system_prompt:
            return [1, 2, 3]
        if mismatched_observation_ids and _pixel_key(images[0]) == _pixel_key(edited):
            return [40, 99]
        return list(OBSERVATION_IDS)

    async def run_tool_round(
        **kwargs: Any,
    ) -> tuple[list[int], Image.Image | None, bool]:
        del kwargs
        if mode == "tool_failure":
            return [], None, False
        return [], edited, True

    async def generate(
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        images: list[Image.Image],
        videos: list[Any],
    ) -> FakeTokenOutput:
        del videos
        calls.append(
            {
                "prompt_ids": list(prompt_ids),
                "sampling_params": dict(sampling_params),
                "image_pixels": [_pixel_key(image) for image in images],
            }
        )
        if len(images) == 1:
            ids = DIRECT_ANSWER_IDS if mode == "direct" else ACTION_IDS
            return FakeTokenOutput(ids, [-0.1] * len(ids))
        if _pixel_key(images[-1]) == _pixel_key(edited):
            if mode == "harm":
                return FakeTokenOutput(COUNTERFACTUAL_ANSWER_IDS, [-0.2, -0.3])
            return FakeTokenOutput(FACTUAL_ANSWER_IDS, [-0.2, -0.3])
        if mode == "harm":
            return FakeTokenOutput(FACTUAL_ANSWER_IDS, [-0.4, -0.5])
        return FakeTokenOutput(COUNTERFACTUAL_ANSWER_IDS, [-0.4, -0.5])

    agent.process_vision_info = process_vision_info  # type: ignore[method-assign]
    agent.apply_chat_template = apply_chat_template  # type: ignore[method-assign]
    agent._run_tool_round = run_tool_round  # type: ignore[method-assign]
    agent._generate = generate  # type: ignore[method-assign]
    return agent, calls


async def _run_case(mode: str) -> dict[str, Any]:
    agent, calls = _make_agent(mode=mode)
    trajectory_info = {"step": 2, "index": 17, "rollout": 3, "mode": mode}
    action_seed, continuation_seed = deterministic_rollout_seeds(trajectory_info)
    output = await agent.run(
        {"temperature": 0.7, "top_p": 0.9},
        _trajectory_info=trajectory_info,
        raw_prompt=[{"role": "user", "content": "ignored by fake"}],
        reward_model={"ground_truth": "7"},
        tools_kwargs={"metadata": "{}"},
    )
    reward_extra_info = output.extra_fields["reward_extra_info"]
    audit = json.loads(reward_extra_info[ROLLOUT_AUDIT_JSON_KEY])
    assert reward_extra_info["acc"] == output.reward_score
    assert audit["schema"] == ROLLOUT_AUDIT_SCHEMA
    assert set(audit) == {"schema", *ROLLOUT_AUDIT_FIELDS}
    for field in ROLLOUT_AUDIT_FIELDS:
        assert audit[field] == output.extra_fields[field]

    if mode == "direct":
        assert len(calls) == 1
        assert calls[0]["sampling_params"]["seed"] == action_seed
        assert output.response_ids == DIRECT_ANSWER_IDS
        assert output.response_mask == [1]
        assert output.reward_score == 1.0
        assert output.extra_fields[ACTION_TOKEN_COUNT_KEY] == 0
        assert output.extra_fields[OBSERVATION_TOKEN_COUNT_KEY] == 0
        assert output.extra_fields[ANSWER_TOKEN_COUNT_KEY] == 1
        assert output.extra_fields[ACTION_CREDIT_KEY] == 0.0
        assert output.extra_fields[PAIR_VALID_KEY] is False
        return {
            "mode": mode,
            "action_credit": 0.0,
            "reward_score": output.reward_score,
            "generation_calls": len(calls),
        }

    assert len(calls) == 3
    action_call = calls[0]
    branch_calls = calls[1:]
    assert action_call["sampling_params"]["seed"] == action_seed
    assert action_seed != continuation_seed
    assert branch_calls[0]["prompt_ids"] == branch_calls[1]["prompt_ids"]
    assert branch_calls[0]["sampling_params"] == branch_calls[1]["sampling_params"]
    assert branch_calls[0]["sampling_params"]["seed"] == continuation_seed
    assert branch_calls[0]["prompt_ids"] == [1, 2, 3] + ACTION_IDS + OBSERVATION_IDS
    pair = CounterfactualActionPair.from_dict(
        output.extra_fields["vtool_action_credit_pair"]
    )
    assert pair.factual.prefix_sha256 == pair.counterfactual.prefix_sha256
    assert pair.factual.action_sha256 == pair.counterfactual.action_sha256
    assert pair.factual.decoding_sha256 == pair.counterfactual.decoding_sha256
    assert pair.factual.continuation_seed == pair.counterfactual.continuation_seed
    assert output.extra_fields[PAIR_VALID_KEY] is True
    assert output.extra_fields[ACTION_TOKEN_COUNT_KEY] == len(ACTION_IDS)
    assert output.extra_fields[OBSERVATION_TOKEN_COUNT_KEY] == len(OBSERVATION_IDS)
    expected_factual_answer_ids = (
        FACTUAL_ANSWER_IDS if mode == "rescue" else COUNTERFACTUAL_ANSWER_IDS
    )
    assert output.extra_fields[ANSWER_TOKEN_COUNT_KEY] == len(
        expected_factual_answer_ids
    )
    assert output.response_mask == [1, 1, 0, 0, 1, 1]
    assert output.response_ids == (
        ACTION_IDS + OBSERVATION_IDS + expected_factual_answer_ids
    )

    expected_credit = {
        "rescue": 0.95,
        "harm": -1.05,
        "tool_failure": -0.05,
    }[mode]
    assert math.isclose(pair.action_credit, expected_credit, abs_tol=1e-12)
    if mode == "rescue":
        assert (
            branch_calls[0]["image_pixels"][-1] != branch_calls[1]["image_pixels"][-1]
        )
        assert pair.factual.task_score == 1.0
        assert pair.counterfactual.task_score == 0.0
    if mode == "tool_failure":
        assert branch_calls[0]["image_pixels"] == branch_calls[1]["image_pixels"]
        assert output.extra_fields["vtool_tool_success"] is False
    return {
        "mode": mode,
        "action_credit": pair.action_credit,
        "factual_score": pair.factual.task_score,
        "counterfactual_score": pair.counterfactual.task_score,
        "generation_calls": len(calls),
        "response_mask": output.response_mask,
        "shared_branch_prompt_sha256": canonical_sha256(branch_calls[0]["prompt_ids"]),
        "shared_branch_sampling_sha256": canonical_sha256(
            branch_calls[0]["sampling_params"]
        ),
    }


async def _assert_mismatched_observation_fails_closed() -> str:
    agent, _ = _make_agent(mode="rescue", mismatched_observation_ids=True)
    try:
        await agent.run(
            {"temperature": 0.7},
            _trajectory_info={"step": 0, "index": 1},
            raw_prompt=[{"role": "user", "content": "ignored"}],
            reward_model={"ground_truth": "7"},
            tools_kwargs={"metadata": "{}"},
        )
    except RuntimeError as exc:
        if "different token ids" not in str(exc):
            raise
        return str(exc)
    raise AssertionError("mismatched observation IDs did not fail closed")


async def _assert_missing_trajectory_fails_closed() -> str:
    agent, _ = _make_agent(mode="rescue")
    try:
        await agent.run(
            {"temperature": 0.7},
            raw_prompt=[{"role": "user", "content": "ignored"}],
            reward_model={"ground_truth": "7"},
            tools_kwargs={"metadata": "{}"},
        )
    except ValueError as exc:
        if "_trajectory_info" not in str(exc):
            raise
        return str(exc)
    raise AssertionError("missing trajectory identity did not fail closed")


async def run_contract() -> dict[str, Any]:
    cases = [
        await _run_case(mode) for mode in ("rescue", "harm", "tool_failure", "direct")
    ]
    observation_error = await _assert_mismatched_observation_fails_closed()
    trajectory_error = await _assert_missing_trajectory_fails_closed()
    return {
        "schema": REPORT_SCHEMA,
        "decision": "paired_vtool_fake_server_contract_passed",
        "cases": cases,
        "mismatched_observation_error": observation_error,
        "missing_trajectory_error": trajectory_error,
        "checks": {
            "shared_prefix": True,
            "shared_continuation_seed": True,
            "factual_only_image_delta": True,
            "action_observation_answer_mask": True,
            "rescue_credit": True,
            "harm_or_failed_action_cost": True,
            "direct_answer_no_pair": True,
            "rollout_audit_payload_exported": True,
            "mismatch_fail_closed": True,
        },
        "model_weights_loaded": False,
        "protected_split_contents_accessed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite fake-server report: {args.output}"
        )
    report = asyncio.run(run_contract())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
