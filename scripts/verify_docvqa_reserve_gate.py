#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from beyond_entropy.docvqa_reserve import RESERVE_SOURCES
from beyond_entropy.reserve_freeze import sha256_file, validate_reserve_freeze


def _load(path: Path, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"reserve {name} must be a JSON object")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the materialized reserve gate")
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--expected-audit-sha256", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    freeze_path = args.freeze.resolve()
    manifest_path = args.manifest.resolve()
    audit_path = args.audit.resolve()
    for name, path, expected in (
        ("freeze", freeze_path, args.expected_freeze_sha256),
        ("manifest", manifest_path, args.expected_manifest_sha256),
        ("audit", audit_path, args.expected_audit_sha256),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"reserve {name} SHA-256 mismatch")
    freeze = _load(freeze_path, "freeze")
    validate_reserve_freeze(
        freeze, expected_code_revision=revision, verify_components=True
    )
    audit = _load(audit_path, "audit")
    manifest_audit = audit.get("manifest")
    if not isinstance(manifest_audit, dict) or (
        audit.get("passed") is not True
        or audit.get("freeze_sha256") != args.expected_freeze_sha256
        or audit.get("reserve_outcomes_collected") is not False
        or manifest_audit.get("manifest_sha256") != args.expected_manifest_sha256
        or manifest_audit.get("unique_sources") != RESERVE_SOURCES
        or manifest_audit.get("unique_images") != RESERVE_SOURCES
    ):
        raise ValueError("reserve manifest gate is not bound to the frozen population")
    print(
        json.dumps(
            {
                "passed": True,
                "code_revision": revision,
                "freeze_sha256": args.expected_freeze_sha256,
                "manifest_sha256": args.expected_manifest_sha256,
                "audit_sha256": args.expected_audit_sha256,
                "states": manifest_audit["count"],
                "sources": RESERVE_SOURCES,
                "reserve_outcomes_collected": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
