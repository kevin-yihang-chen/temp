import pytest

from beyond_entropy.source_allocation import SourceRoleSpec, allocate_source_roles


def _allocation(**overrides):
    source_images = {f"source-{index}": f"image-{index}" for index in range(14)}
    kwargs = {
        "source_images": source_images,
        "roles": [
            SourceRoleSpec("ranker", 0, 3),
            SourceRoleSpec("calibration", 3, 2),
            SourceRoleSpec("formal", 5, 3),
        ],
        "seed": 17,
        "namespace": "allocation-v1",
    }
    kwargs.update(overrides)
    return allocate_source_roles(**kwargs)


def test_source_role_allocation_is_deterministic_and_disjoint():
    first = _allocation()
    second = _allocation()
    assert first == second
    assignments = [
        assignment
        for role in first["roles"].values()
        for assignment in role["assignments"]
    ]
    assert len(assignments) == 8
    assert len({item["source_group_id"] for item in assignments}) == 8
    assert len({item["image_id"] for item in assignments}) == 8
    assert all(role["reserve_backfill_count"] == 0 for role in first["roles"].values())


def test_source_role_allocation_excludes_prior_and_duplicate_rgb_with_backfill():
    sources = {f"source-{index}": f"image-{index}" for index in range(14)}
    baseline = _allocation(source_images=sources)
    ranker_assignments = baseline["roles"]["ranker"]["assignments"]
    excluded_image = ranker_assignments[0]["image_id"]
    duplicate_target_group = ranker_assignments[1]["source_group_id"]
    sources[duplicate_target_group] = ranker_assignments[2]["image_id"]

    result = _allocation(
        source_images=sources,
        excluded_image_ids={excluded_image},
        excluded_source_group_ids={ranker_assignments[1]["source_group_id"]},
    )
    assignments = [
        assignment
        for role in result["roles"].values()
        for assignment in role["assignments"]
    ]
    assert excluded_image not in {item["image_id"] for item in assignments}
    assert len({item["image_id"] for item in assignments}) == 8
    assert result["prior_collision_source_group_count"] == 1
    assert result["prior_source_group_collision_count"] == 1
    assert result["duplicate_rgb_source_group_count"] == 1
    assert sum(
        role["reserve_backfill_count"] for role in result["roles"].values()
    ) >= 3


def test_source_role_allocation_keeps_first_hash_rank_for_duplicate_rgb():
    sources = {f"source-{index}": f"image-{index}" for index in range(14)}
    first = _allocation(source_images=sources)
    ordered_assignments = sorted(
        (
            assignment
            for role in first["roles"].values()
            for assignment in role["assignments"]
        ),
        key=lambda item: item["source_rank"],
    )
    earlier = ordered_assignments[0]
    later = ordered_assignments[-1]
    sources[later["source_group_id"]] = earlier["image_id"]
    result = _allocation(source_images=sources)
    duplicate = result["duplicate_rgb_source_groups"][0]
    assert duplicate["canonical_source_group_id"] == earlier["source_group_id"]
    assert duplicate["source_group_id"] == later["source_group_id"]


def test_source_role_allocation_rejects_overlapping_intervals():
    with pytest.raises(ValueError, match="must not overlap"):
        _allocation(
            roles=[
                SourceRoleSpec("first", 0, 4),
                SourceRoleSpec("second", 3, 2),
            ]
        )


def test_source_role_allocation_rejects_insufficient_reserve():
    with pytest.raises(ValueError, match="insufficient eligible reserve"):
        _allocation(
            source_images={"a": "shared", "b": "shared", "c": "shared"},
            roles=[SourceRoleSpec("ranker", 0, 2)],
        )
