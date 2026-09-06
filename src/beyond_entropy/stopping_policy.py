"""Policies for one fixed additional visual acquisition."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class StoppingDecision:
    continue_acquisition: bool
    score: float
    policy: str


@dataclass(frozen=True)
class GainStoppingPolicy:
    lambda_cost: float
    threshold: float = 0.0
    name: str = "learned_gain"

    def __post_init__(self) -> None:
        if not math.isfinite(self.lambda_cost) or self.lambda_cost < 0:
            raise ValueError("lambda_cost must be finite and non-negative")
        if not math.isfinite(self.threshold):
            raise ValueError("threshold must be finite")

    def decide(self, predicted_gain: float, proposed_cost: float) -> StoppingDecision:
        score = float(predicted_gain) - self.lambda_cost * float(proposed_cost)
        return StoppingDecision(score > self.threshold, score, self.name)


@dataclass(frozen=True)
class RiskGainStoppingPolicy:
    lambda_cost: float
    gain_threshold: float = 0.0
    risk_threshold: float = 0.5
    name: str = "risk_plus_gain"

    def decide(
        self, predicted_gain: float, predicted_risk: float, proposed_cost: float
    ) -> StoppingDecision:
        net_gain = float(predicted_gain) - self.lambda_cost * float(proposed_cost)
        proceed = net_gain > self.gain_threshold and float(predicted_risk) > self.risk_threshold
        return StoppingDecision(proceed, min(net_gain, float(predicted_risk)), self.name)


def matched_rate_random_mask(
    decision_ids: Sequence[tuple[str, str]],
    *,
    rate: float,
    seed: int,
) -> list[bool]:
    """Deterministic random baseline with exactly the nearest feasible call rate."""

    if not 0 <= rate <= 1:
        raise ValueError("rate must be in [0, 1]")
    count = round(rate * len(decision_ids))
    ranked = sorted(
        range(len(decision_ids)),
        key=lambda index: hashlib.sha256(
            f"matched-random:{seed}:{decision_ids[index]}".encode()
        ).digest(),
    )
    selected = set(ranked[:count])
    return [index in selected for index in range(len(decision_ids))]
