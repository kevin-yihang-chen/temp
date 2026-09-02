"""Paired same-prefix VTool rollout for the frozen action-credit G1 smoke.

This module is loaded only inside the pinned VTool/verl runtime. It intentionally
keeps that heavyweight dependency outside the core ``beyond_entropy`` package.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from typing import Any
from uuid import uuid4

from PIL import Image

from beyond_entropy.benchmarks import chartqa_relaxed_match
from beyond_entropy.counterfactual_action_credit import (
    CounterfactualActionPair,
    CounterfactualArmOutcome,
)
from beyond_entropy.rollout import GroundTruth
from beyond_entropy.vtool_action_credit import (
    ACTION_CREDIT_KEY,
    ACTION_TOKEN_COUNT_KEY,
    ANSWER_TOKEN_COUNT_KEY,
    OBSERVATION_TOKEN_COUNT_KEY,
    PAIR_VALID_KEY,
    TRAJECTORY_ID_KEY,
    canonical_sha256,
    deterministic_rollout_seeds,
    extract_vtool_answer,
)
from recipe.vtool.vtool import VToolAgentLoop
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.utils.profiler import simple_timer


NEUTRAL_OBSERVATION = (
    "OBSERVATION: The visual tool returned a second image. "
    "Answer the original question using the available images."
)
SCORER_NAME = "chartqa_relaxed_match_with_vtool_answer_extraction_v1"
ROLLOUT_AUDIT_SCHEMA = "vtool_action_credit_rollout_audit_v1"
ROLLOUT_AUDIT_JSON_KEY = "vtool_action_credit_audit_json"
ROLLOUT_AUDIT_FIELDS = (
    "vtool_tool_attempted",
    "vtool_tool_success",
    "vtool_final_response_text",
    "vtool_counterfactual_response_text",
    TRAJECTORY_ID_KEY,
    ACTION_TOKEN_COUNT_KEY,
    OBSERVATION_TOKEN_COUNT_KEY,
    ANSWER_TOKEN_COUNT_KEY,
    ACTION_CREDIT_KEY,
    PAIR_VALID_KEY,
    "vtool_action_credit_pair",
    "vtool_counterfactual_generation_seconds",
)


def _image_sha256(image: Image.Image) -> str:
    digest = hashlib.sha256()
    digest.update(image.mode.encode("ascii"))
    digest.update(str(image.size).encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def local_chartqa_score(response: str, ground_truth: object) -> float:
    return chartqa_relaxed_match(
        extract_vtool_answer(response), GroundTruth(str(ground_truth))
    )


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: object,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """Deterministic reward function shared by paired and outcome-only arms."""

    del data_source, extra_info, kwargs
    score = local_chartqa_score(solution_str, ground_truth)
    return {"score": score, "acc": score}


def _attach_rollout_audit(
    extra_fields: dict[str, Any], *, score: float
) -> dict[str, Any]:
    """Put a JSON-safe, stable audit payload in upstream rollout dumps."""

    missing = [name for name in ROLLOUT_AUDIT_FIELDS if name not in extra_fields]
    if missing:
        raise ValueError(f"rollout audit fields are missing: {missing}")
    payload = {
        "schema": ROLLOUT_AUDIT_SCHEMA,
        **{name: extra_fields[name] for name in ROLLOUT_AUDIT_FIELDS},
    }
    extra_fields["reward_extra_info"] = {
        "acc": float(score),
        ROLLOUT_AUDIT_JSON_KEY: json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
    }
    return extra_fields


class CounterfactualCreditVToolAgentLoop(VToolAgentLoop):
    """Return the factual rollout while scoring a matched no-op continuation."""

    async def _neutral_observation_ids(self, second_image: Image.Image) -> list[int]:
        if self.processor is None:
            raise RuntimeError(
                "paired visual action credit requires a vision processor"
            )
        return await self.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": NEUTRAL_OBSERVATION},
                    ],
                }
            ],
            images=[second_image],
            remove_system_prompt=True,
        )

    async def _generate(
        self,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        images: list[Image.Image],
        videos: list[Any],
    ):
        return await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=images or None,
            video_data=videos or None,
        )

    @staticmethod
    def _reward_target(kwargs: dict[str, Any]) -> object:
        reward_model = kwargs.get("reward_model")
        if not isinstance(reward_model, dict) or "ground_truth" not in reward_model:
            raise ValueError("paired rollout requires reward_model.ground_truth")
        return reward_model["ground_truth"]

    def _policy_sha256(self) -> str:
        model_config = self.config.actor_rollout_ref.model
        return canonical_sha256(
            {
                "model_path": str(model_config.path),
                "upstream": "VTOOL-R1/vtool-r1@d2aa28353ec10c7f91b39f502925003a81d6982d",
            }
        )

    @staticmethod
    def _multi_modal_output(
        images: list[Image.Image], videos: list[Any]
    ) -> dict[str, Any]:
        output: dict[str, Any] = {}
        if images:
            output["images"] = images
        if videos:
            output["videos"] = videos
        return output

    async def run(
        self, sampling_params: dict[str, Any], **kwargs: Any
    ) -> AgentLoopOutput:
        trajectory_info = kwargs.get("_trajectory_info")
        if trajectory_info is None:
            raise ValueError(
                "pinned upstream patch must pass _trajectory_info into the agent loop"
            )
        action_seed, continuation_seed = deterministic_rollout_seeds(trajectory_info)
        messages = list(kwargs["raw_prompt"])
        multi_modal_data = await self.process_vision_info(messages)
        images = list(multi_modal_data.get("images") or [])
        videos = list(multi_modal_data.get("videos") or [])
        if not images:
            raise ValueError("paired visual action credit requires an input image")
        tools_kwargs = kwargs.get("tools_kwargs") or {}
        target = self._reward_target(kwargs)
        initial_prompt_ids = await self.apply_chat_template(
            messages, images=images, videos=videos or None
        )
        first_sampling = dict(sampling_params)
        first_sampling["seed"] = action_seed
        metrics: dict[str, Any] = {
            "tool_use_attempted": 0.0,
            "tool_use_success": 0.0,
        }
        with simple_timer("generate_sequences", metrics):
            first = await self._generate(
                prompt_ids=initial_prompt_ids,
                sampling_params=first_sampling,
                images=images,
                videos=videos,
            )
        action_ids = list(first.token_ids)
        if not action_ids:
            raise RuntimeError("first assistant generation returned no tokens")
        first_text = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(action_ids, skip_special_tokens=False),
        )
        parse_result = self.code_parser.parse(first_text)
        trajectory_id = canonical_sha256(
            {"trajectory_info": trajectory_info, "action_ids": action_ids}
        )

        if parse_result.error_code == "NOTOOL":
            direct_ids = action_ids[: self.response_length]
            final_text = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(direct_ids, skip_special_tokens=True),
            )
            score = local_chartqa_score(final_text, target)
            extra_fields = _attach_rollout_audit(
                {
                    "turn_scores": [],
                    "tool_rewards": [],
                    "vtool_tool_attempted": False,
                    "vtool_tool_success": False,
                    "vtool_final_response_text": final_text,
                    "vtool_counterfactual_response_text": None,
                    TRAJECTORY_ID_KEY: trajectory_id,
                    ACTION_TOKEN_COUNT_KEY: 0,
                    OBSERVATION_TOKEN_COUNT_KEY: 0,
                    ANSWER_TOKEN_COUNT_KEY: len(direct_ids),
                    ACTION_CREDIT_KEY: 0.0,
                    PAIR_VALID_KEY: False,
                    "vtool_action_credit_pair": None,
                    "vtool_counterfactual_generation_seconds": 0.0,
                },
                score=score,
            )
            return AgentLoopOutput(
                prompt_ids=initial_prompt_ids,
                response_ids=direct_ids,
                response_mask=[1] * len(direct_ids),
                response_logprobs=(
                    list(first.log_probs)[: self.response_length]
                    if first.log_probs
                    else None
                ),
                multi_modal_data=self._multi_modal_output(images, videos),
                reward_score=score,
                num_turns=2,
                metrics=metrics,
                extra_fields=extra_fields,
            )

        metrics["tool_use_attempted"] = 1.0
        with simple_timer("tool_calls", metrics):
            _, edited_image, tool_success = await self._run_tool_round(
                parse_result=parse_result,
                images=images,
                tools_kwargs=tools_kwargs,
            )
        metrics["tool_use_success"] = float(tool_success)
        factual_second_image = edited_image if edited_image is not None else images[0]
        factual_observation_ids, counterfactual_observation_ids = await asyncio.gather(
            self._neutral_observation_ids(factual_second_image),
            self._neutral_observation_ids(images[0]),
        )
        if factual_observation_ids != counterfactual_observation_ids:
            raise RuntimeError(
                "neutral factual/no-op observation templates produced different token ids"
            )
        observation_ids = factual_observation_ids
        branch_prompt_ids = initial_prompt_ids + action_ids + observation_ids
        if len(action_ids) + len(observation_ids) + 1 > self.response_length:
            raise RuntimeError(
                "action and observation leave no room for a final answer"
            )
        branch_sampling = dict(sampling_params)
        branch_sampling["seed"] = continuation_seed
        factual_images = images + [factual_second_image]
        counterfactual_images = images + [images[0]]
        with simple_timer("paired_continuations", metrics):
            factual, counterfactual = await asyncio.gather(
                self._generate(
                    prompt_ids=branch_prompt_ids,
                    sampling_params=dict(branch_sampling),
                    images=factual_images,
                    videos=videos,
                ),
                self._generate(
                    prompt_ids=branch_prompt_ids,
                    sampling_params=dict(branch_sampling),
                    images=counterfactual_images,
                    videos=videos,
                ),
            )
        counterfactual_generation_seconds = float(metrics["paired_continuations"])
        metrics["generate_sequences"] += counterfactual_generation_seconds
        factual_answer_ids = list(factual.token_ids)
        counterfactual_answer_ids = list(counterfactual.token_ids)
        if not factual_answer_ids or not counterfactual_answer_ids:
            raise RuntimeError("paired continuation returned no final-answer tokens")
        max_answer_length = (
            self.response_length - len(action_ids) - len(observation_ids)
        )
        factual_answer_ids = factual_answer_ids[:max_answer_length]
        counterfactual_answer_ids = counterfactual_answer_ids[:max_answer_length]
        factual_text, counterfactual_text = await asyncio.gather(
            self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(
                    factual_answer_ids, skip_special_tokens=True
                ),
            ),
            self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(
                    counterfactual_answer_ids, skip_special_tokens=True
                ),
            ),
        )
        factual_score = local_chartqa_score(factual_text, target)
        counterfactual_score = local_chartqa_score(counterfactual_text, target)
        decoding_sha256 = canonical_sha256(branch_sampling)
        scorer_sha256 = canonical_sha256(
            {
                "name": SCORER_NAME,
                "source": inspect.getsource(local_chartqa_score),
                "extractor_source": inspect.getsource(extract_vtool_answer),
            }
        )
        prefix_sha256 = canonical_sha256(initial_prompt_ids)
        action_sha256 = canonical_sha256(action_ids)
        target_sha256 = canonical_sha256(str(target))
        policy_sha256 = self._policy_sha256()
        pair = CounterfactualActionPair(
            trajectory_id=trajectory_id,
            factual=CounterfactualArmOutcome(
                branch_id="factual-edited-observation",
                prefix_sha256=prefix_sha256,
                action_sha256=action_sha256,
                observation_sha256=canonical_sha256(
                    {
                        "observation_ids": observation_ids,
                        "second_image_sha256": _image_sha256(factual_second_image),
                    }
                ),
                target_sha256=target_sha256,
                policy_sha256=policy_sha256,
                decoding_sha256=decoding_sha256,
                scorer_sha256=scorer_sha256,
                continuation_seed=continuation_seed,
                task_score=factual_score,
                action_cost=1.0,
            ),
            counterfactual=CounterfactualArmOutcome(
                branch_id="counterfactual-noop-observation",
                prefix_sha256=prefix_sha256,
                action_sha256=action_sha256,
                observation_sha256=canonical_sha256(
                    {
                        "observation_ids": observation_ids,
                        "second_image_sha256": _image_sha256(images[0]),
                    }
                ),
                target_sha256=target_sha256,
                policy_sha256=policy_sha256,
                decoding_sha256=decoding_sha256,
                scorer_sha256=scorer_sha256,
                continuation_seed=continuation_seed,
                task_score=counterfactual_score,
                action_cost=0.0,
            ),
            lambda_cost=0.05,
        )
        response_ids = action_ids + observation_ids + factual_answer_ids
        response_mask = (
            [1] * len(action_ids)
            + [0] * len(observation_ids)
            + [1] * len(factual_answer_ids)
        )
        response_logprobs: list[float] | None = None
        if first.log_probs or factual.log_probs:
            response_logprobs = (
                list(first.log_probs or [0.0] * len(action_ids))
                + [0.0] * len(observation_ids)
                + list(factual.log_probs or [0.0] * len(factual_answer_ids))[
                    : len(factual_answer_ids)
                ]
            )
        extra_fields = _attach_rollout_audit(
            {
                "turn_scores": [],
                "tool_rewards": [],
                "vtool_tool_attempted": True,
                "vtool_tool_success": tool_success,
                "vtool_final_response_text": factual_text,
                "vtool_counterfactual_response_text": counterfactual_text,
                TRAJECTORY_ID_KEY: trajectory_id,
                ACTION_TOKEN_COUNT_KEY: len(action_ids),
                OBSERVATION_TOKEN_COUNT_KEY: len(observation_ids),
                ANSWER_TOKEN_COUNT_KEY: len(factual_answer_ids),
                ACTION_CREDIT_KEY: pair.action_credit,
                PAIR_VALID_KEY: True,
                "vtool_action_credit_pair": pair.to_dict(),
                "vtool_counterfactual_generation_seconds": (
                    counterfactual_generation_seconds
                ),
            },
            score=factual_score,
        )
        return AgentLoopOutput(
            prompt_ids=initial_prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            multi_modal_data=self._multi_modal_output(factual_images, videos),
            reward_score=factual_score,
            num_turns=4,
            metrics=metrics,
            extra_fields=extra_fields,
        )
