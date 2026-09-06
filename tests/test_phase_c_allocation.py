import pytest

from beyond_entropy.phase_c_allocation import (
    allocate_hrbench_phase_c,
    hash_rank,
    role_overlap_audit,
    select_complete_groups,
)


def test_hash_rank_and_complete_group_selection_are_deterministic():
    rows = [
        {"state_id": f"s{i}", "source_id": f"g{i//2}", "image_id": f"i{i//2}"}
        for i in range(8)
    ]
    selected_a, groups_a = select_complete_groups(
        rows, group_key="source_id", group_count=2, seed=17, namespace="test"
    )
    selected_b, groups_b = select_complete_groups(
        list(reversed(rows)), group_key="source_id", group_count=2,
        seed=17, namespace="test",
    )
    assert groups_a == groups_b == hash_rank(
        [f"g{i}" for i in range(4)], seed=17, namespace="test"
    )[:2]
    assert {row["state_id"] for row in selected_a} == {
        row["state_id"] for row in selected_b
    }
    assert len(selected_a) == 4


def test_hrbench_heldout_excludes_every_historically_used_image():
    rows = [
        {"state_id": f"s{i}-{j}", "source_id": f"q{i}-{j}", "image_id": f"image-{i}"}
        for i in range(5) for j in range(4)
    ]
    result = allocate_hrbench_phase_c(
        rows, historically_used_image_ids={"image-0", "image-1"},
        heldout_image_count=2, seed=9,
    )
    heldout_images = {row["image_id"] for row in result["heldout_rows"]}
    assert len(heldout_images) == 2
    assert not heldout_images & {"image-0", "image-1"}
    assert role_overlap_audit(result["train_rows"], result["heldout_rows"]) == {
        "state_id_overlap": 0, "source_id_overlap": 0, "image_id_overlap": 0,
    }


def test_role_overlap_audit_fails_closed():
    row = {"state_id": "s", "source_id": "source", "image_id": "image"}
    with pytest.raises(ValueError, match="role leakage"):
        role_overlap_audit([row], [row])
