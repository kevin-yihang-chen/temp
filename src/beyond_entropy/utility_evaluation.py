"""Deterministic single-acquisition evaluation with full-cost UG comparisons.

Outcomes are evaluation-only. Learned score arrays must be generated separately
from UtilityInputs; this module never fits a model or chooses a threshold.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Any, Mapping, Sequence

from .policies import EntropySearchPolicy, RandomZoomPolicy
from .predictability_audit import BinaryToolOutcome
from .predictability_evaluation import paired_source_bootstrap_policy_difference, policy_metrics
from .utility_dataset import UtilitySample


@dataclass(frozen=True)
class UtilityChoice:
    index: int
    tool_calls: int
    visual_cost: float


def _single(sample: UtilitySample, index: int) -> UtilityChoice:
    a = sample.inputs.action_space.actions[index]
    return UtilityChoice(index, int(index != 0), a.visual_cost)


def _ug(sample: UtilitySample) -> UtilityChoice:
    if len(sample.replicate_ids) != 1:
        raise ValueError("MVP evaluation requires a single paired seed, not averaged entropy")
    decision = EntropySearchPolicy().select(sample.outcomes)
    index = next(a.index for a in sample.inputs.action_space.actions if a.action_id == decision.selected.action_id)
    return UtilityChoice(index, decision.tool_calls, decision.visual_cost)


def policy_choices(
    samples: Sequence[UtilitySample], *, lambda_cost: float,
    learned_gains: Mapping[str, Mapping[str, Sequence[float]]],
    frozen_voi_calls: Mapping[str, bool], random_seed: int = 17,
) -> dict[str, list[UtilityChoice]]:
    """Frozen VOI keeps its original ANSWER/full-UG action and four-call cost."""
    if not samples or len({s.inputs.state.state_id for s in samples}) != len(samples):
        raise ValueError("nonempty unique-state single-benchmark evaluation required")
    if len({s.benchmark for s in samples}) != 1:
        raise ValueError("evaluate domains separately")
    expected = {s.inputs.state.state_id for s in samples}
    if set(learned_gains) != {"format_sft", "best_action_sft", "utility_sft"}:
        raise ValueError("all three SFT arms are mandatory")
    if any(set(rows) != expected for rows in learned_gains.values()) or set(frozen_voi_calls) != expected:
        raise ValueError("prediction coverage mismatch")
    if any(type(v) is not bool for v in frozen_voi_calls.values()):
        raise ValueError("frozen VOI must supply genuine frozen boolean decisions")
    result = {name: [] for name in ("answer_only", "random_crop", "ug", "frozen_voi", *learned_gains, "oracle")}
    for s in samples:
        space, key = s.inputs.action_space, s.inputs.state.state_id
        answer, ug = _single(s, 0), _ug(s)
        result["answer_only"].append(answer)
        random_decision = RandomZoomPolicy(random_seed).select(s.outcomes)
        random_index = next(a.index for a in space.actions if a.action_id == random_decision.selected.action_id)
        result["random_crop"].append(_single(s, random_index))
        result["ug"].append(ug)
        result["frozen_voi"].append(ug if frozen_voi_calls[key] else answer)
        for name, by_state in learned_gains.items():
            result[name].append(_single(s, space.select(by_state[key], lambda_cost=lambda_cost)))
        result["oracle"].append(_single(s, space.select(s.gains, lambda_cost=lambda_cost)))
    return result


def choice_ledgers(samples: Sequence[UtilitySample], choices: Sequence[UtilityChoice]) -> tuple[list[BinaryToolOutcome], list[bool]]:
    if len(samples) != len(choices) or not samples:
        raise ValueError("choices and states must align")
    rows, calls = [], []
    for sample, choice in zip(samples, choices):
        state = sample.inputs.state
        if type(choice.index) is not int or not 0 <= choice.index < len(sample.gains):
            raise ValueError("choice outside action support")
        action = sample.inputs.action_space.actions[choice.index]
        if choice.tool_calls == 0 and choice.index != 0:
            raise ValueError("selected crop cannot be free")
        if choice.visual_cost < action.visual_cost or choice.tool_calls < int(choice.index != 0):
            raise ValueError("undercharged selected action")
        rows.append(BinaryToolOutcome(
            state.state_id, sample.replicate_ids[0], state.image_id, state.source_id,
            action.action_id, sample.rewards[0], sample.rewards[choice.index],
            choice.visual_cost, choice.tool_calls,
        ))
        calls.append(choice.tool_calls > 0)
    return rows, calls


def choice_metrics(samples: Sequence[UtilitySample], choices: Sequence[UtilityChoice], *, lambda_cost: float) -> dict[str, Any]:
    ledgers, calls = choice_ledgers(samples, choices)
    reused = policy_metrics(ledgers, calls, lambda_cost=lambda_cost)
    by_source: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        by_source[s.inputs.state.source_id].append(i)

    def weighted(values):
        return mean(mean(values[i] for i in indices) for indices in by_source.values())

    selected_gain = [s.gains[c.index] for s, c in zip(samples, choices)]
    useful = [float(call and g > 0) for call, g in zip(calls, selected_gain)]
    opportunities = [float(max(s.gains) > 0) for s in samples]
    call_rate, opportunity_rate = weighted(calls), weighted(opportunities)
    quantities = {
        "accuracy": [s.rewards[c.index] for s, c in zip(samples, choices)],
        "accuracy_gain": selected_gain,
        "avg_tool_calls": [c.tool_calls for c in choices],
        "avg_visual_cost": [c.visual_cost for c in choices],
        "net_utility": [g-lambda_cost*c.visual_cost for g, c in zip(selected_gain, choices)],
        "top1_regret": [max(s.gains)-g for s, g in zip(samples, selected_gain)],
        "answer_rate": [float(c.index == 0) for c in choices],
        "unnecessary_tool_call_rate": [float(call and g <= 0) for call, g in zip(calls, selected_gain)],
    }
    source_metrics = {k: weighted(v) for k, v in quantities.items()}
    source_metrics.update({
        "useful_tool_precision": None if not call_rate else weighted(useful)/call_rate,
        "useful_tool_recall": None if not opportunity_rate else weighted(useful)/opportunity_rate,
        "unnecessary_fraction_among_calls": None if not call_rate else source_metrics["unnecessary_tool_call_rate"]/call_rate,
    })
    if abs(source_metrics["net_utility"]-reused["incremental_utility"]) > 1e-10:
        raise RuntimeError("utility disagrees with existing cost-accounting implementation")
    return {"source_balanced": source_metrics,
            "question_weighted": {k: mean(v) for k, v in quantities.items()},
            "definitions": {"useful": "selected raw gain > 0", "recall_denominator": "states with any positive raw gain", "unnecessary_tool_call_rate": "nonpositive selected gain and called / all states", "visual_cost": "incremental crop cost; selector overhead reported separately"}}


def paired_choice_interval(samples, candidate, baseline, *, lambda_cost, resamples=20000, seed=17):
    candidate_rows, candidate_calls = choice_ledgers(samples, candidate)
    baseline_rows, baseline_calls = choice_ledgers(samples, baseline)
    return paired_source_bootstrap_policy_difference(
        candidate_rows, candidate_calls, baseline_rows, baseline_calls,
        lambda_cost=lambda_cost, resamples=resamples, confidence_level=.95, seed=seed,
    )
