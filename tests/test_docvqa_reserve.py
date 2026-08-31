from __future__ import annotations

import hashlib
from typing import Any

import pytest
from PIL import Image

from beyond_entropy.docvqa_reserve import (
    select_reserve_identities,
    validate_reserve_rows,
)
from beyond_entropy.docvqa_train_allocation import (
    DATASET_ID,
    DATASET_NAME,
    DATASET_REVISION,
    DATASET_SPLIT,
    NAMESPACE,
    PROTOCOL_SHA256,
    ROLE_SPECS,
    SEED,
)
from beyond_entropy.manifest_export import image_digest


def _allocation_fixture(count: int = 9503) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    source_images = {
        f"source-{index:05d}": hashlib.sha256(f"image-{index}".encode()).hexdigest()
        for index in range(count)
    }
    ranked = sorted(
        source_images,
        key=lambda source: (
            hashlib.sha256(f"{NAMESPACE}\0{SEED}\0{source}".encode()).digest(),
            source,
        ),
    )
    roles: dict[str, Any] = {}
    cursor = 0
    for spec in ROLE_SPECS:
        selected = ranked[cursor : cursor + spec.count]
        roles[spec.name] = {
            "assignments": [
                {
                    "source_group_id": source,
                    "source_rank": rank,
                    "image_id": source_images[source],
                    "origin": "base_interval",
                }
                for rank, source in enumerate(selected, start=cursor)
            ]
        }
        cursor += spec.count
    allocation = {
        "protocol_sha256": PROTOCOL_SHA256,
        "dataset": {
            "dataset_id": DATASET_ID,
            "dataset_name": DATASET_NAME,
            "dataset_revision": DATASET_REVISION,
            "split": DATASET_SPLIT,
        },
        "selection_contract": {
            "selection_target_fields_accessed": False,
            "selection_allowed_fields": ["docId", "image"],
        },
        "allocation": {
            "namespace": NAMESPACE,
            "seed": SEED,
            "source_group_count": count,
            "reserve_consumed_end_exclusive": 9506,
            "roles": roles,
        },
    }
    return allocation, source_images, ranked


def test_select_reserve_identities_preserves_complete_raw_suffix():
    allocation, source_images, ranked = _allocation_fixture()
    selected = select_reserve_identities(
        allocation,
        source_images,
        excluded_image_ids=set(),
        excluded_source_group_ids=set(),
        reserve_start=9500,
        reserve_end_exclusive=9503,
    )
    assert [item["source_group_id"] for item in selected] == ranked[9500:9503]
    assert [item["source_rank"] for item in selected] == [9500, 9501, 9502]


def test_select_reserve_identities_fails_closed_on_prior_collision():
    allocation, source_images, ranked = _allocation_fixture()
    with pytest.raises(ValueError, match="prior bank"):
        select_reserve_identities(
            allocation,
            source_images,
            excluded_image_ids={source_images[ranked[9501]]},
            excluded_source_group_ids=set(),
            reserve_start=9500,
            reserve_end_exclusive=9503,
        )


def test_select_reserve_identities_fails_closed_on_duplicate_rgb():
    allocation, source_images, ranked = _allocation_fixture()
    source_images[ranked[9501]] = source_images[ranked[0]]
    with pytest.raises(ValueError, match="duplicate-RGB"):
        select_reserve_identities(
            allocation,
            source_images,
            excluded_image_ids=set(),
            excluded_source_group_ids=set(),
            reserve_start=9500,
            reserve_end_exclusive=9503,
        )


def test_validate_reserve_rows_recomputes_rgb_identity():
    image = Image.new("RGB", (2, 2), color=(10, 20, 30))
    identities = [
        {
            "source_group_id": "doc-1",
            "source_rank": 9506,
            "image_id": image_digest(image),
        }
    ]
    audit = validate_reserve_rows([{"docId": "doc-1", "image": image}], identities)
    assert audit["source_group_count"] == 1
    assert audit["selection_target_fields_accessed"] is False
