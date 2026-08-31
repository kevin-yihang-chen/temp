from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .docvqa_reserve import RESERVE_END_EXCLUSIVE, RESERVE_SOURCES, RESERVE_START


MANDATORY_COMPONENTS = frozenset(
    {
        "allocation",
        "allocation_audit",
        "allocation_protocol",
        "comparator_protocol",
        "implementation_specification",
        "development_rollouts",
        "development_features",
        "policy_a_model",
        "policy_a_report",
        "policy_b_model",
        "policy_b_report",
        "comparator_verification",
        "reserve_identity_audit",
        "reserve_identity_auditor",
        "reserve_identity_module",
        "reserve_comparator_module",
        "reserve_freeze_module",
        "reserve_exporter",
        "reserve_scorer",
        "reserve_evaluator",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def component_path(
    freeze: Mapping[str, Any], name: str, *, verify: bool = True
) -> Path:
    components = freeze.get("components")
    if not isinstance(components, Mapping):
        raise ValueError("reserve freeze lacks components")
    item = components.get(name)
    if not isinstance(item, Mapping):
        raise ValueError(f"reserve freeze lacks component {name}")
    path = Path(str(item.get("path", ""))).resolve()
    expected = str(item.get("sha256", ""))
    if not path.is_file() or len(expected) != 64:
        raise ValueError(f"reserve freeze component {name} is invalid")
    if verify and sha256_file(path) != expected:
        raise ValueError(f"reserve freeze component {name} SHA-256 mismatch")
    return path


def validate_reserve_freeze(
    freeze: Mapping[str, Any],
    *,
    expected_code_revision: str | None = None,
    verify_components: bool = True,
) -> None:
    if freeze.get("schema_version") != 1:
        raise ValueError("reserve freeze schema mismatch")
    revision = str(freeze.get("code_revision", "")).strip()
    if len(revision) != 40:
        raise ValueError("reserve freeze code revision is invalid")
    if expected_code_revision is not None and revision != expected_code_revision:
        raise ValueError("reserve freeze code revision differs from execution")
    if freeze.get("reserve_outcomes_used") is not False:
        raise ValueError("reserve freeze used reserve outcomes")
    if freeze.get("formal_outcomes_used") is not False:
        raise ValueError("reserve freeze used formal outcomes")
    population = freeze.get("population")
    if not isinstance(population, Mapping) or population != {
        "rank_start": RESERVE_START,
        "rank_end_exclusive": RESERVE_END_EXCLUSIVE,
        "expected_source_groups": RESERVE_SOURCES,
        "manifest_materialized": False,
        "rollouts_collected": False,
        "outcomes_used": False,
    }:
        raise ValueError("reserve freeze population contract changed")
    components = freeze.get("components")
    if not isinstance(components, Mapping) or not MANDATORY_COMPONENTS.issubset(
        components
    ):
        raise ValueError("reserve freeze is missing mandatory components")
    for name in sorted(components):
        component_path(freeze, name, verify=verify_components)
