from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Collection, Mapping, Sequence


@dataclass(frozen=True)
class SourceRoleSpec:
    """A fixed hash-rank interval and requested source count for one role."""

    name: str
    offset: int
    count: int


def _source_rank(group_id: str, *, seed: int, namespace: str) -> tuple[str, str]:
    payload = f"{namespace}\0{seed}\0{group_id}".encode()
    return hashlib.sha256(payload).hexdigest(), group_id


def allocate_source_roles(
    source_images: Mapping[str, str],
    *,
    roles: Sequence[SourceRoleSpec],
    excluded_image_ids: Collection[str] = (),
    excluded_source_group_ids: Collection[str] = (),
    seed: int,
    namespace: str,
) -> dict[str, Any]:
    """Allocate disjoint source roles with deterministic reserve backfilling.

    Source groups are ordered by a seeded SHA-256 rank. Each role first draws
    from its preregistered raw rank interval. Groups whose decoded-RGB image
    appeared in a prior bank are ineligible. If multiple groups decode to the
    same RGB image, only the earliest ranked group is eligible. Missing slots
    are filled, in role order, from the untouched suffix after all registered
    role intervals.
    """

    normalized_namespace = str(namespace).strip()
    if not normalized_namespace:
        raise ValueError("namespace must be non-empty")
    normalized_sources: dict[str, str] = {}
    for raw_group_id, raw_image_id in source_images.items():
        group_id = str(raw_group_id).strip()
        image_id = str(raw_image_id).strip()
        if not group_id or not image_id:
            raise ValueError("source group and image IDs must be non-empty")
        if group_id in normalized_sources and normalized_sources[group_id] != image_id:
            raise ValueError(f"source group {group_id!r} maps to multiple images")
        normalized_sources[group_id] = image_id
    if not normalized_sources:
        raise ValueError("source_images must not be empty")

    normalized_roles: list[SourceRoleSpec] = []
    role_names: set[str] = set()
    for role in roles:
        name = str(role.name).strip()
        if not name or name in role_names:
            raise ValueError("role names must be non-empty and unique")
        if role.offset < 0 or role.count <= 0:
            raise ValueError("role offsets must be non-negative and counts positive")
        normalized_roles.append(SourceRoleSpec(name, role.offset, role.count))
        role_names.add(name)
    if not normalized_roles:
        raise ValueError("roles must not be empty")
    normalized_roles.sort(key=lambda role: (role.offset, role.name))
    for left, right in zip(normalized_roles, normalized_roles[1:]):
        if left.offset + left.count > right.offset:
            raise ValueError("role hash-rank intervals must not overlap")

    excluded = {str(image_id).strip() for image_id in excluded_image_ids}
    if "" in excluded:
        raise ValueError("excluded image IDs must be non-empty")
    excluded_groups = {
        str(group_id).strip() for group_id in excluded_source_group_ids
    }
    if "" in excluded_groups:
        raise ValueError("excluded source group IDs must be non-empty")
    ordered_groups = sorted(
        normalized_sources,
        key=lambda group_id: _source_rank(
            group_id,
            seed=seed,
            namespace=normalized_namespace,
        ),
    )
    rank_by_group = {group_id: rank for rank, group_id in enumerate(ordered_groups)}

    canonical_group_by_image: dict[str, str] = {}
    duplicate_groups: list[dict[str, Any]] = []
    for group_id in ordered_groups:
        image_id = normalized_sources[group_id]
        canonical_group = canonical_group_by_image.setdefault(image_id, group_id)
        if canonical_group != group_id:
            duplicate_groups.append(
                {
                    "source_group_id": group_id,
                    "source_rank": rank_by_group[group_id],
                    "image_id": image_id,
                    "canonical_source_group_id": canonical_group,
                    "canonical_source_rank": rank_by_group[canonical_group],
                }
            )
    eligible_groups = {
        group_id
        for image_id, group_id in canonical_group_by_image.items()
        if image_id not in excluded and group_id not in excluded_groups
    }
    prior_collision_groups = [
        {
            "source_group_id": group_id,
            "source_rank": rank_by_group[group_id],
            "image_id": normalized_sources[group_id],
        }
        for group_id in ordered_groups
        if normalized_sources[group_id] in excluded
    ]
    prior_source_group_collisions = [
        {
            "source_group_id": group_id,
            "source_rank": rank_by_group[group_id],
            "image_id": normalized_sources[group_id],
        }
        for group_id in ordered_groups
        if group_id in excluded_groups
    ]

    reserve_start = max(role.offset + role.count for role in normalized_roles)
    reserve_cursor = reserve_start
    selected_groups: set[str] = set()
    role_payloads: dict[str, Any] = {}
    for role in normalized_roles:
        assignments: list[dict[str, Any]] = []
        for rank in range(role.offset, min(role.offset + role.count, len(ordered_groups))):
            group_id = ordered_groups[rank]
            if group_id not in eligible_groups or group_id in selected_groups:
                continue
            assignments.append(
                {
                    "source_group_id": group_id,
                    "source_rank": rank,
                    "image_id": normalized_sources[group_id],
                    "origin": "base_interval",
                }
            )
            selected_groups.add(group_id)
        base_selected_count = len(assignments)
        while len(assignments) < role.count:
            if reserve_cursor >= len(ordered_groups):
                raise ValueError(
                    f"insufficient eligible reserve sources to fill role {role.name!r}"
                )
            group_id = ordered_groups[reserve_cursor]
            source_rank = reserve_cursor
            reserve_cursor += 1
            if group_id not in eligible_groups or group_id in selected_groups:
                continue
            assignments.append(
                {
                    "source_group_id": group_id,
                    "source_rank": source_rank,
                    "image_id": normalized_sources[group_id],
                    "origin": "reserve_backfill",
                }
            )
            selected_groups.add(group_id)
        role_payloads[role.name] = {
            "offset": role.offset,
            "count": role.count,
            "base_interval_end_exclusive": role.offset + role.count,
            "base_selected_count": base_selected_count,
            "reserve_backfill_count": role.count - base_selected_count,
            "assignments": assignments,
        }

    selected_image_ids = [
        assignment["image_id"]
        for role in role_payloads.values()
        for assignment in role["assignments"]
    ]
    if len(selected_image_ids) != len(set(selected_image_ids)):
        raise RuntimeError("allocation produced cross-role RGB image overlap")
    if set(selected_image_ids) & excluded:
        raise RuntimeError("allocation produced prior-bank RGB image overlap")
    if selected_groups & excluded_groups:
        raise RuntimeError("allocation produced prior-bank source-group overlap")

    return {
        "schema_version": 1,
        "namespace": normalized_namespace,
        "seed": seed,
        "source_group_count": len(ordered_groups),
        "unique_image_count": len(canonical_group_by_image),
        "excluded_prior_image_count": len(excluded),
        "excluded_prior_source_group_count": len(excluded_groups),
        "prior_collision_source_group_count": len(prior_collision_groups),
        "prior_collision_source_groups": prior_collision_groups,
        "prior_source_group_collision_count": len(prior_source_group_collisions),
        "prior_source_group_collisions": prior_source_group_collisions,
        "duplicate_rgb_source_group_count": len(duplicate_groups),
        "duplicate_rgb_source_groups": duplicate_groups,
        "reserve_start": reserve_start,
        "reserve_consumed_end_exclusive": reserve_cursor,
        "roles": role_payloads,
    }
