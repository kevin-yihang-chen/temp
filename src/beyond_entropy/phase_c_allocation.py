"""Outcome-blind helpers for the factorized-method Phase-C data freeze."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence


def hash_rank(values: Iterable[str], *, seed: int, namespace: str) -> list[str]:
    """Return a stable unique ordering without consulting task outcomes."""

    if not namespace:
        raise ValueError("allocation namespace must be non-empty")
    return sorted(
        {str(value) for value in values},
        key=lambda value: (
            hashlib.sha256(f"{namespace}\0{seed}\0{value}".encode()).hexdigest(),
            value,
        ),
    )


def select_complete_groups(
    rows: Sequence[Mapping[str, Any]], *, group_key: str, group_count: int,
    seed: int, namespace: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Select all rows of a fixed number of hash-ranked source groups."""

    if group_count <= 0:
        raise ValueError("group_count must be positive")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group = str(row.get(group_key, "")).strip()
        if not group:
            raise ValueError(f"row is missing non-empty {group_key}")
        grouped[group].append(dict(row))
    ordered = hash_rank(grouped, seed=seed, namespace=namespace)
    if len(ordered) < group_count:
        raise ValueError("insufficient complete groups for allocation")
    selected = ordered[:group_count]
    selected_set = set(selected)
    return (
        [dict(row) for row in rows if str(row[group_key]) in selected_set],
        selected,
    )


def allocate_hrbench_phase_c(
    rows: Sequence[Mapping[str, Any]], *, historically_used_image_ids: Iterable[str],
    heldout_image_count: int, seed: int,
) -> dict[str, Any]:
    """Reserve unseen sequential image groups and leave all others for training.

    HRBench questions share images, so allocation is performed by decoded image
    identity.  An image touched by any earlier sequential development rollout
    is ineligible for held-out use, even when another question on it was unseen.
    """

    used = {str(value) for value in historically_used_image_ids}
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        image_id = str(row.get("image_id", "")).strip()
        source_id = str(row.get("source_id", "")).strip()
        if not image_id or not source_id:
            raise ValueError("HRBench rows require image_id and source_id")
        by_image[image_id].append(dict(row))
    eligible = [image for image in by_image if image not in used]
    ordered = hash_rank(
        eligible, seed=seed, namespace="factorized-phase-c-hrbench-heldout"
    )
    if heldout_image_count <= 0 or len(ordered) < heldout_image_count:
        raise ValueError("insufficient untouched HRBench images")
    heldout_images = set(ordered[:heldout_image_count])
    train = [dict(row) for row in rows if str(row["image_id"]) not in heldout_images]
    heldout = [dict(row) for row in rows if str(row["image_id"]) in heldout_images]
    if not heldout or not train:
        raise RuntimeError("HRBench allocation produced an empty role")
    return {
        "train_rows": train,
        "heldout_rows": heldout,
        "heldout_image_ids": sorted(heldout_images),
        "eligible_unseen_image_count": len(eligible),
        "historically_used_image_count": len(used),
    }


def role_overlap_audit(
    train_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Fail closed on state, source, or decoded-image overlap."""

    result = {}
    for key in ("state_id", "source_id", "image_id"):
        left = {str(row[key]) for row in train_rows}
        right = {str(row[key]) for row in heldout_rows}
        result[f"{key}_overlap"] = len(left & right)
    if any(result.values()):
        raise ValueError(f"Phase-C role leakage: {result}")
    return result
