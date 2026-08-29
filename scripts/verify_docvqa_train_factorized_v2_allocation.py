from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from beyond_entropy.docvqa_train_allocation import (
    DATASET_ID,
    DATASET_NAME,
    DATASET_REVISION,
    DATASET_SPLIT,
    PROTOCOL_SHA256,
    load_docvqa_source_images,
    load_prior_identities,
    sha256_file,
    verify_recomputed_allocation_bundle,
)


def _load_mapping(path: Path, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independently recompute the sealed DocVQA-train allocation"
    )
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()

    allocation_path = args.allocation.resolve()
    audit_path = args.audit.resolve()
    protocol_path = args.protocol.resolve()
    for path in (allocation_path, audit_path, protocol_path):
        if not path.is_file():
            raise FileNotFoundError(f"verification input does not exist: {path}")
    if sha256_file(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("DocVQA preregistration SHA-256 mismatch")
    document = _load_mapping(allocation_path, "allocation")
    audit = _load_mapping(audit_path, "allocation audit")
    allocation_sha256 = sha256_file(allocation_path)
    if audit.get("allocation_sha256") != allocation_sha256:
        raise ValueError("allocation audit binds a different allocation SHA-256")

    dataset = document.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("allocation dataset provenance must be a JSON object")
    expected_dataset = {
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "split": DATASET_SPLIT,
    }
    for name, expected in expected_dataset.items():
        if dataset.get(name) != expected:
            raise ValueError(f"DocVQA dataset provenance changed for {name}")
    parquet_paths = [Path(value).resolve() for value in dataset.get("parquet_files", [])]
    parquet_sha256 = [str(value) for value in dataset.get("parquet_sha256", [])]
    if not parquet_paths or len(parquet_paths) != len(parquet_sha256):
        raise ValueError("allocation Parquet provenance is incomplete")
    for path, expected in zip(parquet_paths, parquet_sha256):
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"source Parquet SHA-256 mismatch: {path}")

    prior_banks = document.get("prior_banks")
    if not isinstance(prior_banks, list) or any(
        not isinstance(record, dict) for record in prior_banks
    ):
        raise ValueError("allocation prior-bank provenance is invalid")
    prior_manifest_paths = [
        Path(record["manifest"]).resolve() for record in prior_banks
    ]
    for path, record in zip(prior_manifest_paths, prior_banks):
        if not path.is_file() or sha256_file(path) != record.get("manifest_sha256"):
            raise ValueError(f"prior manifest SHA-256 mismatch: {path}")
    prior_image_ids, prior_docvqa_groups, recomputed_prior = load_prior_identities(
        prior_manifest_paths,
        verify_images=True,
    )
    if recomputed_prior != prior_banks:
        raise ValueError("prior-bank identity inventory differs from allocation")

    source_images, row_count = load_docvqa_source_images(parquet_paths)
    if row_count != dataset.get("row_count"):
        raise ValueError("DocVQA train row count differs from allocation")
    if len(source_images) != dataset.get("source_group_count"):
        raise ValueError("DocVQA train source count differs from allocation")
    report = verify_recomputed_allocation_bundle(
        document,
        audit,
        source_images=source_images,
        excluded_image_ids=prior_image_ids,
        excluded_source_group_ids=prior_docvqa_groups,
        prior_banks=recomputed_prior,
        parquet_files=parquet_paths,
        parquet_sha256=parquet_sha256,
        row_count=row_count,
        protocol_path=protocol_path,
        allocation_path=allocation_path,
        allocation_sha256=allocation_sha256,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
