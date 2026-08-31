from __future__ import annotations

import hashlib
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from .docvqa_train_allocation import (
    DATASET_ID,
    DATASET_NAME,
    DATASET_REVISION,
    DATASET_SPLIT,
    NAMESPACE,
    PROTOCOL_SHA256,
    ROLE_SPECS,
    SEED,
)
from .manifest_export import image_digest


RESERVE_ROLE = "reserve_toolgate_followup"
RESERVE_STATE_NAMESPACE = "docvqa-train-factorized-v2-reserve-toolgate-v1"
RESERVE_START = 9506
RESERVE_END_EXCLUSIVE = 10194
RESERVE_SOURCES = RESERVE_END_EXCLUSIVE - RESERVE_START


def _require(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"DocVQA reserve mismatch for {name}")


def _ranked_source_groups(
    source_images: Mapping[str, str],
    *,
    namespace: str,
    seed: int,
) -> list[str]:
    normalized: dict[str, str] = {}
    for raw_source, raw_image in source_images.items():
        source = str(raw_source).strip()
        image = str(raw_image).strip()
        if not source or not image:
            raise ValueError("DocVQA reserve source and image IDs must be non-empty")
        previous = normalized.setdefault(source, image)
        if previous != image:
            raise ValueError(f"DocVQA source {source!r} maps to multiple images")
    return sorted(
        normalized,
        key=lambda source: (
            hashlib.sha256(f"{namespace}\0{seed}\0{source}".encode()).digest(),
            source,
        ),
    )


def _main_role_identities(
    allocation_document: Mapping[str, Any],
) -> tuple[set[str], set[str]]:
    body = allocation_document.get("allocation")
    if not isinstance(body, Mapping):
        raise ValueError("DocVQA reserve allocation body is missing")
    roles = body.get("roles")
    if not isinstance(roles, Mapping):
        raise ValueError("DocVQA reserve allocation roles are missing")
    sources: set[str] = set()
    images: set[str] = set()
    for spec in ROLE_SPECS:
        payload = roles.get(spec.name)
        if not isinstance(payload, Mapping):
            raise ValueError(f"DocVQA reserve allocation lacks role {spec.name!r}")
        assignments = payload.get("assignments")
        if not isinstance(assignments, list) or len(assignments) != spec.count:
            raise ValueError(f"DocVQA reserve role {spec.name!r} count changed")
        for item in assignments:
            if not isinstance(item, Mapping):
                raise ValueError("DocVQA reserve allocation assignment is invalid")
            source = str(item.get("source_group_id", "")).strip()
            image = str(item.get("image_id", "")).strip()
            if not source or not image or source in sources or image in images:
                raise ValueError("DocVQA reserve main roles are not identity-disjoint")
            sources.add(source)
            images.add(image)
    if len(sources) != 9500 or len(images) != 9500:
        raise ValueError("DocVQA reserve main-role population changed")
    return sources, images


def select_reserve_identities(
    allocation_document: Mapping[str, Any],
    source_images: Mapping[str, str],
    *,
    excluded_image_ids: Collection[str],
    excluded_source_group_ids: Collection[str],
    reserve_start: int = RESERVE_START,
    reserve_end_exclusive: int = RESERVE_END_EXCLUSIVE,
) -> list[dict[str, Any]]:
    """Recompute the outcome-sealed suffix and fail on any eligibility drift.

    Selection reads only source-group and decoded-RGB identities.  Unlike the
    main allocator, this follow-up does not backfill: every immutable raw rank
    in the registered suffix must itself still be eligible.
    """

    if reserve_start < 0 or reserve_end_exclusive <= reserve_start:
        raise ValueError("DocVQA reserve rank interval is invalid")
    _require(allocation_document.get("protocol_sha256"), PROTOCOL_SHA256, "protocol")
    dataset = allocation_document.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("DocVQA reserve allocation lacks dataset provenance")
    expected_dataset = {
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "split": DATASET_SPLIT,
    }
    for name, expected in expected_dataset.items():
        _require(dataset.get(name), expected, f"dataset {name}")
    body = allocation_document.get("allocation")
    if not isinstance(body, Mapping):
        raise ValueError("DocVQA reserve allocation body is missing")
    _require(body.get("namespace"), NAMESPACE, "namespace")
    _require(body.get("seed"), SEED, "seed")
    _require(body.get("source_group_count"), len(source_images), "source count")
    if reserve_start == RESERVE_START:
        _require(
            body.get("reserve_consumed_end_exclusive"),
            RESERVE_START,
            "main-role reserve cursor",
        )
    if reserve_end_exclusive == RESERVE_END_EXCLUSIVE:
        _require(len(source_images), RESERVE_END_EXCLUSIVE, "immutable suffix end")

    contract = allocation_document.get("selection_contract")
    if not isinstance(contract, Mapping) or (
        contract.get("selection_target_fields_accessed") is not False
        or contract.get("selection_allowed_fields") != ["docId", "image"]
    ):
        raise ValueError("DocVQA reserve allocation selection contract changed")

    ranked = _ranked_source_groups(source_images, namespace=NAMESPACE, seed=SEED)
    if reserve_end_exclusive > len(ranked):
        raise ValueError("DocVQA reserve rank interval exceeds the source population")
    normalized_images = {
        str(source).strip(): str(image).strip()
        for source, image in source_images.items()
    }
    rank_by_source = {source: rank for rank, source in enumerate(ranked)}
    canonical_by_image: dict[str, str] = {}
    for source in ranked:
        canonical_by_image.setdefault(normalized_images[source], source)

    excluded_images = {str(value).strip() for value in excluded_image_ids}
    excluded_sources = {
        str(value).strip() for value in excluded_source_group_ids
    }
    if "" in excluded_images or "" in excluded_sources:
        raise ValueError("DocVQA reserve exclusions must be non-empty identities")
    main_sources, main_images = _main_role_identities(allocation_document)

    selected: list[dict[str, Any]] = []
    for rank in range(reserve_start, reserve_end_exclusive):
        source = ranked[rank]
        image = normalized_images[source]
        if canonical_by_image[image] != source:
            raise ValueError(f"DocVQA reserve rank {rank} is a duplicate-RGB source")
        if image in excluded_images or source in excluded_sources:
            raise ValueError(f"DocVQA reserve rank {rank} collides with a prior bank")
        if source in main_sources or image in main_images:
            raise ValueError(f"DocVQA reserve rank {rank} overlaps a main role")
        selected.append(
            {
                "source_group_id": source,
                "source_rank": rank,
                "image_id": image,
            }
        )
    if len(selected) != reserve_end_exclusive - reserve_start:
        raise RuntimeError("DocVQA reserve did not preserve the complete rank suffix")
    if len({item["source_group_id"] for item in selected}) != len(selected):
        raise RuntimeError("DocVQA reserve contains duplicate sources")
    if len({item["image_id"] for item in selected}) != len(selected):
        raise RuntimeError("DocVQA reserve contains duplicate RGB identities")
    if [rank_by_source[item["source_group_id"]] for item in selected] != list(
        range(reserve_start, reserve_end_exclusive)
    ):
        raise RuntimeError("DocVQA reserve rank order changed")
    return selected


def validate_reserve_rows(
    rows: Sequence[Mapping[str, Any]],
    identities: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute docId/RGB identity for every selected row before export."""

    expected = {
        str(item["source_group_id"]): str(item["image_id"]) for item in identities
    }
    if len(expected) != len(identities):
        raise ValueError("DocVQA reserve identities are not unique")
    observed: dict[str, str] = {}
    for index, row in enumerate(rows):
        source = str(row.get("docId", "")).strip()
        if source not in expected:
            raise ValueError(f"DocVQA reserve row {index} is outside the suffix")
        raw_image = row.get("image")
        convert = getattr(raw_image, "convert", None)
        if not callable(convert):
            raise ValueError(f"DocVQA reserve row {index} image is not decodable")
        digest = image_digest(convert("RGB"))
        if digest != expected[source]:
            raise ValueError(f"DocVQA reserve row {index} RGB differs from allocation")
        previous = observed.setdefault(source, digest)
        if previous != digest:
            raise ValueError(f"DocVQA reserve source {source!r} maps to multiple images")
    if set(observed) != set(expected):
        raise ValueError("DocVQA reserve rows do not cover the suffix exactly")
    return {
        "role": RESERVE_ROLE,
        "row_count": len(rows),
        "source_group_count": len(observed),
        "unique_image_count": len(set(observed.values())),
        "source_identity_recomputed": True,
        "selection_target_fields_accessed": False,
    }
