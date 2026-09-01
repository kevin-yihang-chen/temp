from __future__ import annotations

import hashlib


SHARD_ALGORITHM = "sha256-state-id-v1"
_SHARD_DOMAIN = b"beyond-entropy-qwen-shard-v1\0"


def stable_shard_index(
    state_id: str,
    shard_count: int,
    *,
    namespace: str = "",
) -> int:
    """Assign a state to one deterministic shard independent of row order."""

    if not state_id:
        raise ValueError("state_id must be non-empty")
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if "\0" in namespace:
        raise ValueError("shard namespace must not contain NUL")
    namespaced_id = f"{namespace}\0{state_id}" if namespace else state_id
    digest = hashlib.sha256(_SHARD_DOMAIN + namespaced_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % shard_count
