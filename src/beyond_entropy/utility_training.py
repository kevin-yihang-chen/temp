"""Objective-consistent sanity gates shared by training and audit code."""
from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence, TypeVar


T = TypeVar("T")


BASE_SANITY_CHECKS = (
    "finite_loss_decreased",
    "backbone_gradients_nonzero",
    "vision_merger_updated",
    "language_block_updated",
    "head_updated",
    "single_original_image_no_candidate_execution",
)


def required_sanity_checks(method: str) -> tuple[str, ...]:
    if method == "format":
        return BASE_SANITY_CHECKS + ("support_memorized",)
    if method == "best_action":
        # CE only identifies the best class. It contains no signed magnitudes,
        # so positive/negative gain separation is not an identifiable gate.
        return BASE_SANITY_CHECKS + ("overfit_regret_zero",)
    if method == "utility":
        return BASE_SANITY_CHECKS + (
            "positive_negative_separated", "overfit_regret_zero"
        )
    raise ValueError("unknown Utility-SFT method")


def sanity_passed(method: str, checks: Mapping[str, object], *, engineering: bool) -> bool:
    required = required_sanity_checks(method)
    if set(required) - set(checks):
        raise ValueError("sanity report is missing objective-required checks")
    return not engineering and all(checks[name] is True for name in required)


def source_hash_subset(
    samples: Sequence[T], *, maximum_sources: int, seed: int, namespace: str
) -> list[T]:
    """Outcome-independent whole-source subset for a development pilot."""
    if maximum_sources <= 0 or not namespace:
        raise ValueError("positive source limit and namespace required")
    by_source: dict[str, list[T]] = {}
    for sample in samples:
        source = str(getattr(getattr(sample, "inputs"), "state").source_id)
        by_source.setdefault(source, []).append(sample)
    ordered = sorted(
        by_source,
        key=lambda source: hashlib.sha256(
            f"{namespace}:{seed}:{source}".encode()
        ).hexdigest(),
    )
    selected = set(ordered[:maximum_sources])
    return sorted(
        (sample for source in selected for sample in by_source[source]),
        key=lambda sample: getattr(getattr(sample, "inputs"), "state").state_id,
    )


def source_cycle_samples(
    grouped: Mapping[str, Sequence[T]], *, draws: int, seed: int, namespace: str
) -> list[T]:
    """Outcome-independent deterministic sampling with no source replacement per cycle.

    Each complete cycle visits every source exactly once in a hash-defined order.
    When a source owns multiple states, successive cycles rotate through a
    deterministic state order.  Labels are never inspected, so this schedule is
    shared unchanged by Format, Best-Action, and Utility SFT.
    """
    if type(draws) is not int or draws <= 0 or not grouped or not namespace:
        raise ValueError("positive draws, nonempty grouped samples, and namespace required")
    if any(not source or not samples for source, samples in grouped.items()):
        raise ValueError("every source must be nonempty")
    sources = sorted(str(source) for source in grouped)
    if len(sources) != len(grouped):
        raise ValueError("source identifiers must have unique string forms")
    result: list[T] = []
    cycle = -1
    ordered_sources: list[str] = []
    for index in range(draws):
        next_cycle, offset = divmod(index, len(sources))
        if next_cycle != cycle:
            cycle = next_cycle
            ordered_sources = sorted(
                sources,
                key=lambda source: hashlib.sha256(
                    f"{namespace}:{seed}:cycle:{cycle}:{source}".encode()
                ).hexdigest(),
            )
        source = ordered_sources[offset]
        states = sorted(
            grouped[source],
            key=lambda sample: getattr(getattr(sample, "inputs"), "state").state_id,
        )
        state_offset = int(
            hashlib.sha256(
                f"{namespace}:{seed}:state:{cycle}:{source}".encode()
            ).hexdigest(),
            16,
        ) % len(states)
        result.append(states[state_offset])
    return result


def supervision_kwargs(sample: Any, *, method: str, temperature: float, device: Any) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {"method": method, "temperature": temperature}
    if method == "format":
        result["support_labels"] = torch.tensor([sample.support_action], device=device)
    elif method in ("best_action", "utility", "pairwise"):
        result["gains"] = torch.tensor([sample.gains], device=device, dtype=torch.float32)
    else:
        raise ValueError("unknown Utility-SFT method")
    return result


def optimizer_to_device(optimizer: Any, device: Any) -> None:
    """Restore resumable optimizer tensors to the current logical GPU."""
    import torch

    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)
