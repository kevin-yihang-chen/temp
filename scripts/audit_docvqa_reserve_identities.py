#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from beyond_entropy.docvqa_reserve import RESERVE_SOURCES, select_reserve_identities
from beyond_entropy.docvqa_train_allocation import (
    load_docvqa_source_images,
    load_prior_identities,
    sha256_file,
    verify_recomputed_allocation_bundle,
)


def _load(path: Path, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"reserve {name} must be a JSON object")
    return payload


def _identity_digest(items: list[dict[str, Any]]) -> str:
    payload = "\n".join(
        f"{item['source_rank']}\t{item['source_group_id']}\t{item['image_id']}"
        for item in items
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the sealed DocVQA reserve suffix using identities only"
    )
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--expected-allocation-sha256", required=True)
    parser.add_argument("--allocation-audit", type=Path, required=True)
    parser.add_argument("--expected-allocation-audit-sha256", required=True)
    parser.add_argument("--allocation-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    allocation_path = args.allocation.resolve()
    allocation_audit_path = args.allocation_audit.resolve()
    protocol_path = args.allocation_protocol.resolve()
    if sha256_file(allocation_path) != args.expected_allocation_sha256:
        raise ValueError("reserve allocation SHA-256 mismatch")
    if sha256_file(allocation_audit_path) != args.expected_allocation_audit_sha256:
        raise ValueError("reserve allocation audit SHA-256 mismatch")
    allocation = _load(allocation_path, "allocation")
    allocation_audit = _load(allocation_audit_path, "allocation audit")
    dataset = allocation.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("reserve allocation lacks dataset provenance")
    parquet_paths = [
        Path(str(value)).resolve() for value in dataset.get("parquet_files", [])
    ]
    parquet_hashes = [str(value) for value in dataset.get("parquet_sha256", [])]
    if not parquet_paths or len(parquet_paths) != len(parquet_hashes):
        raise ValueError("reserve Parquet provenance is incomplete")
    for path, expected in zip(parquet_paths, parquet_hashes):
        if sha256_file(path) != expected:
            raise ValueError("reserve source Parquet SHA-256 mismatch")
    prior_banks = allocation.get("prior_banks")
    if not isinstance(prior_banks, list) or any(
        not isinstance(item, Mapping) for item in prior_banks
    ):
        raise ValueError("reserve prior-bank provenance is invalid")
    prior_paths = [
        Path(str(item.get("manifest", ""))).resolve() for item in prior_banks
    ]
    prior_images, prior_sources, recomputed_prior = load_prior_identities(
        prior_paths, verify_images=True
    )
    source_images, row_count = load_docvqa_source_images(parquet_paths)
    verify_recomputed_allocation_bundle(
        allocation,
        allocation_audit,
        source_images=source_images,
        excluded_image_ids=prior_images,
        excluded_source_group_ids=prior_sources,
        prior_banks=recomputed_prior,
        parquet_files=parquet_paths,
        parquet_sha256=parquet_hashes,
        row_count=row_count,
        protocol_path=protocol_path,
        allocation_path=allocation_path,
        allocation_sha256=args.expected_allocation_sha256,
    )
    selected = select_reserve_identities(
        allocation,
        source_images,
        excluded_image_ids=prior_images,
        excluded_source_group_ids=prior_sources,
    )
    if len(selected) != RESERVE_SOURCES:
        raise RuntimeError("reserve identity audit did not produce 688 sources")
    repo = Path(__file__).resolve().parents[1]
    result = {
        "passed": True,
        "scientific_status": "identity-only audit of outcome-sealed reserve suffix",
        "code_revision": subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "allocation_sha256": args.expected_allocation_sha256,
        "allocation_audit_sha256": args.expected_allocation_audit_sha256,
        "source_group_count": len(selected),
        "unique_image_count": len({item["image_id"] for item in selected}),
        "rank_start": selected[0]["source_rank"],
        "rank_end_exclusive": selected[-1]["source_rank"] + 1,
        "identity_sequence_sha256": _identity_digest(selected),
        "selection_allowed_fields": ["docId", "image"],
        "selection_target_fields_accessed": False,
        "reserve_manifest_materialized": False,
        "reserve_outcomes_used": False,
        "formal_outcomes_used": False,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
