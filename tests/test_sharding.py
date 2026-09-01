from __future__ import annotations

import pytest

from beyond_entropy.sharding import stable_shard_index


def test_stable_sharding_is_deterministic_exhaustive_and_order_independent():
    states = [f"state-{index:04d}" for index in range(1000)]
    forward = {state: stable_shard_index(state, 7) for state in states}
    reverse = {state: stable_shard_index(state, 7) for state in reversed(states)}
    assert forward == reverse
    assert set(forward.values()) == set(range(7))
    assert sum(list(forward.values()).count(index) for index in range(7)) == len(states)


def test_stable_sharding_rejects_invalid_contract():
    with pytest.raises(ValueError, match="state_id"):
        stable_shard_index("", 2)
    with pytest.raises(ValueError, match="shard_count"):
        stable_shard_index("state", 0)


def test_shard_namespace_is_deterministic_and_domain_separated():
    states = [f"state-{index}" for index in range(32)]
    first = [stable_shard_index(state, 4, namespace="balanced-v1") for state in states]
    second = [stable_shard_index(state, 4, namespace="balanced-v1") for state in states]
    unnamespaced = [stable_shard_index(state, 4) for state in states]
    assert first == second
    assert first != unnamespaced
    with pytest.raises(ValueError, match="NUL"):
        stable_shard_index("state", 4, namespace="invalid\0namespace")
