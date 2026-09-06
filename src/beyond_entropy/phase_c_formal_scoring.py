"""Pre-registered semantic ablations for Phase-C selector scoring."""
from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Callable, Hashable, Mapping, Sequence

from .sequential_post_training import SequentialPolicyInput, SequentialTrainingExample


ABLATION_MODES = ("original", "question_shuffle", "image_shuffle", "region_shuffle")


def deterministic_derangement(
    examples: Sequence[SequentialTrainingExample], *, seed: int, namespace: str,
    component: Callable[[SequentialPolicyInput], Hashable] | None = None,
) -> Mapping[tuple[str, str], SequentialTrainingExample]:
    """Return an outcome-independent component-changing donor assignment.

    Donors may be reused when component multiplicities make a one-to-one
    permutation impossible. Every recipient nevertheless receives a different
    registered component, which is the scientific invariant needed by the
    semantic control.
    """

    if len(examples) < 2 or not namespace:
        raise ValueError("derangement requires at least two examples and a namespace")
    ordered = sorted(examples, key=lambda item: item.decision_id)
    component = component or (lambda inputs: inputs.state_id)
    result = {}
    for recipient in ordered:
        candidates = [
            donor for donor in ordered
            if donor.decision_id != recipient.decision_id
            and component(donor.inputs) != component(recipient.inputs)
        ]
        if not candidates:
            # A constant component cannot support a meaningful shuffle. Keep a
            # decision derangement so scoring completes; the unchanged ranking
            # and call set will deterministically fail the semantic GO gate.
            candidates = [
                donor for donor in ordered
                if donor.decision_id != recipient.decision_id
            ]
        donor = min(
            candidates,
            key=lambda item: (
                hashlib.sha256(
                    f"{namespace}:{seed}:{recipient.decision_id}:"
                    f"{item.decision_id}".encode()
                ).hexdigest(),
                item.decision_id,
            ),
        )
        result[recipient.decision_id] = donor
    return result

def ablated_policy_inputs(
    examples: Sequence[SequentialTrainingExample], *, mode: str,
    seed: int, namespace: str,
) -> Mapping[tuple[str, str], SequentialPolicyInput]:
    """Build a strict outcome-free view for one frozen semantic control."""

    if mode not in ABLATION_MODES:
        raise ValueError(f"unsupported semantic ablation: {mode}")
    if mode == "original":
        return {item.decision_id: item.inputs for item in examples}
    component = {
        "question_shuffle": lambda value: (value.question, value.model_prompt),
        "image_shuffle": lambda value: value.image_path,
        "region_shuffle": lambda value: (
            value.proposed_action_id,
            tuple(value.proposed_bbox.to_list()),
            value.proposed_visual_cost,
        ),
    }[mode]
    donors = deterministic_derangement(
        examples, seed=seed, namespace=namespace, component=component,
    )
    result = {}
    for item in examples:
        source = item.inputs
        donor = donors[item.decision_id].inputs
        if mode == "question_shuffle":
            value = replace(
                source, question=donor.question, model_prompt=donor.model_prompt,
            )
        elif mode == "image_shuffle":
            value = replace(source, image_path=donor.image_path)
        else:
            value = replace(
                source,
                proposed_action_id=donor.proposed_action_id,
                proposed_bbox=donor.proposed_bbox,
                proposed_visual_cost=donor.proposed_visual_cost,
            )
        result[item.decision_id] = value
    return result
