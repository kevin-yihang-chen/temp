from __future__ import annotations

import math
import random

from .schema import ActionRecord, BBox


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, value))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _logit(probability: float) -> float:
    probability = _clamp(probability, 1e-6, 1.0 - 1e-6)
    return math.log(probability / (1.0 - probability))


def _grid_boxes(count: int) -> list[BBox]:
    if count < 2:
        raise ValueError("num_candidates must be at least 2")
    side = math.ceil(math.sqrt(count))
    boxes: list[BBox] = []
    overlap = 0.04
    for index in range(count):
        row, column = divmod(index, side)
        x1 = max(0.0, column / side - overlap)
        y1 = max(0.0, row / side - overlap)
        x2 = min(1.0, (column + 1) / side + overlap)
        y2 = min(1.0, (row + 1) / side + overlap)
        boxes.append(BBox(x1, y1, x2, y2))
    return boxes


def simulate_counterfactual_dataset(
    *,
    n_states: int = 600,
    num_candidates: int = 4,
    seed: int = 7,
) -> list[ActionRecord]:
    """Create a controlled testbed where confidence and usefulness diverge.

    The simulator is a pipeline smoke test, not scientific evidence. It uses a
    shared latent draw per state so sibling outcomes are coupled rather than
    independently sampled, mirroring paired counterfactual evaluation.
    """

    if n_states < 2:
        raise ValueError("n_states must be at least 2")
    rng = random.Random(seed)
    boxes = _grid_boxes(num_candidates)
    records: list[ActionRecord] = []
    for state_number in range(n_states):
        state_id = f"synthetic-{state_number:06d}"
        complexity = rng.random()
        baseline_probability = _sigmoid(2.0 - 3.2 * complexity)
        coupled_draw = rng.random()
        correct_before = float(coupled_draw < baseline_probability)
        entropy_before = 0.35 + 1.35 * complexity + rng.uniform(-0.08, 0.08)
        entropy_before = max(0.05, entropy_before)
        answer_before = "correct" if correct_before else "incorrect"
        records.append(
            ActionRecord(
                state_id=state_id,
                question=f"Synthetic fine-grained visual question {state_number}",
                original_image=f"synthetic://{state_id}.png",
                action_id="answer-now",
                action_type="ANSWER",
                candidate_bbox=None,
                entropy_before=entropy_before,
                entropy_after=entropy_before,
                answer_before=answer_before,
                answer_after=answer_before,
                correct_before=correct_before,
                correct_after=correct_before,
                tool_cost=0.0,
                pre_action_features={"question_complexity": complexity},
                metadata={"synthetic": True},
            )
        )
        relevant_index = rng.randrange(num_candidates)
        for candidate_index, bbox in enumerate(boxes):
            is_relevant = candidate_index == relevant_index
            proposal_score = _clamp(
                (0.76 if is_relevant else 0.26) + rng.gauss(0.0, 0.16)
            )
            context_score = _clamp(
                (0.72 if is_relevant else 0.48) + rng.gauss(0.0, 0.18)
            )
            legibility_score = _clamp(
                (0.80 if is_relevant else 0.47) + rng.gauss(0.0, 0.15)
            )
            visual_clutter = _clamp(
                (0.18 if is_relevant else 0.65) + rng.gauss(0.0, 0.17)
            )
            alignment = proposal_score * (0.55 + 0.45 * complexity)
            action_effect = (
                3.2 * alignment
                + 0.65 * legibility_score
                + 0.35 * context_score
                - 2.15 * visual_clutter
                - 1.35
            )
            after_probability = _sigmoid(_logit(baseline_probability) + action_effect)
            correct_after = float(coupled_draw < after_probability)
            # Distracting, context-poor crops often make the simulated model
            # sharper even when their task outcome is worse.
            entropy_reduction = (
                0.08
                + 0.72 * visual_clutter
                + 0.22 * (1.0 - context_score)
                + rng.uniform(-0.06, 0.06)
            )
            entropy_after = max(0.01, entropy_before - entropy_reduction)
            records.append(
                ActionRecord(
                    state_id=state_id,
                    question=f"Synthetic fine-grained visual question {state_number}",
                    original_image=f"synthetic://{state_id}.png",
                    action_id=f"zoom-{candidate_index}",
                    action_type="ZOOM",
                    candidate_bbox=bbox,
                    entropy_before=entropy_before,
                    entropy_after=entropy_after,
                    answer_before=answer_before,
                    answer_after="correct" if correct_after else "incorrect",
                    correct_before=correct_before,
                    correct_after=correct_after,
                    tool_cost=1.0,
                    pre_action_features={
                        "question_complexity": complexity,
                        "proposal_score": proposal_score,
                        "query_region_alignment": alignment,
                        "context_score": context_score,
                        "legibility_score": legibility_score,
                        "visual_clutter": visual_clutter,
                    },
                    metadata={
                        "synthetic": True,
                        "simulated_relevant_region": is_relevant,
                        "success_probability_before": baseline_probability,
                        "success_probability_after": after_probability,
                    },
                )
            )
    return records
