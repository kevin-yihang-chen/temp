#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from beyond_entropy.docvqa_reserve import (
    RESERVE_ROLE,
    RESERVE_SOURCES,
    RESERVE_STATE_NAMESPACE,
    select_reserve_identities,
    validate_reserve_rows,
)
from beyond_entropy.docvqa_train_allocation import (
    NAMESPACE,
    SEED,
    load_docvqa_source_images,
    load_prior_identities,
    verify_recomputed_allocation_bundle,
)
from beyond_entropy.manifest_audit import audit_manifest
from beyond_entropy.manifest_export import export_benchmark_manifest
from beyond_entropy.reserve_freeze import (
    component_path,
    sha256_file,
    validate_reserve_freeze,
)


def _load(path: Path, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"reserve {name} must be a JSON object")
    return payload


def _revision(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean(repo: Path) -> None:
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("tracked worktree must be clean before reserve export")


def _publish_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize the outcome-sealed 688-source DocVQA reserve suffix"
    )
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    freeze_path = args.freeze.resolve()
    if sha256_file(freeze_path) != args.expected_freeze_sha256:
        raise ValueError("reserve freeze SHA-256 mismatch")
    freeze = _load(freeze_path, "freeze")
    revision = _revision(repo)
    validate_reserve_freeze(
        freeze, expected_code_revision=revision, verify_components=True
    )
    _require_clean(repo)
    output_dir = args.output_dir.resolve()
    audit_output = args.audit_output.resolve()
    staging_dir = output_dir.with_name(output_dir.name + ".tmp")
    staging_audit = audit_output.with_suffix(audit_output.suffix + ".tmp")
    for path in (output_dir, audit_output, staging_dir, staging_audit):
        if path.exists():
            raise FileExistsError(f"reserve output already exists: {path}")

    allocation_path = component_path(freeze, "allocation")
    allocation_audit_path = component_path(freeze, "allocation_audit")
    allocation_protocol = component_path(freeze, "allocation_protocol")
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
        raise ValueError("reserve source Parquet provenance is incomplete")
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
    for path, item in zip(prior_paths, prior_banks):
        if sha256_file(path) != str(item.get("manifest_sha256", "")):
            raise ValueError("reserve prior manifest SHA-256 mismatch")
    prior_images, prior_sources, recomputed_prior = load_prior_identities(
        prior_paths, verify_images=True
    )
    if recomputed_prior != prior_banks:
        raise ValueError("reserve prior-bank inventory changed")
    source_images, row_count = load_docvqa_source_images(parquet_paths)
    if row_count != dataset.get("row_count"):
        raise ValueError("reserve dataset row count changed")
    if len(source_images) != dataset.get("source_group_count"):
        raise ValueError("reserve dataset source count changed")
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
        protocol_path=allocation_protocol,
        allocation_path=allocation_path,
        allocation_sha256=sha256_file(allocation_path),
    )
    identities = select_reserve_identities(
        allocation,
        source_images,
        excluded_image_ids=prior_images,
        excluded_source_group_ids=prior_sources,
    )
    if len(identities) != RESERVE_SOURCES:
        raise RuntimeError("reserve suffix does not contain exactly 688 sources")

    try:
        from datasets import Dataset  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SystemExit("Install benchmark dependencies before reserve export") from exc
    reserve_sources = {str(item["source_group_id"]) for item in identities}
    identity_dataset = Dataset.from_parquet(
        [str(path) for path in parquet_paths], columns=["docId"]
    )
    all_group_ids = [str(value).strip() for value in identity_dataset["docId"]]
    source_indices = [
        index for index, source in enumerate(all_group_ids) if source in reserve_sources
    ]
    if {all_group_ids[index] for index in source_indices} != reserve_sources:
        raise ValueError("reserve sources do not match source Parquet")
    selected_dataset = Dataset.from_parquet(
        [str(path) for path in parquet_paths],
        filters=[("docId", "in", sorted(reserve_sources))],
    )
    selected_ids = [str(value).strip() for value in selected_dataset["docId"]]
    if selected_ids != [all_group_ids[index] for index in source_indices]:
        raise ValueError("reserve filtered row order differs from source Parquet")
    row_audit = validate_reserve_rows(selected_dataset, identities)  # type: ignore[arg-type]

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir()
    selection_metadata = {
        "freeze": str(freeze_path),
        "freeze_sha256": args.expected_freeze_sha256,
        "allocation": str(allocation_path),
        "allocation_sha256": sha256_file(allocation_path),
        "allocation_audit_sha256": sha256_file(allocation_audit_path),
        "namespace": NAMESPACE,
        "role": RESERVE_ROLE,
        "rank_start": identities[0]["source_rank"],
        "rank_end_exclusive": identities[-1]["source_rank"] + 1,
        "selected_source_group_count": RESERVE_SOURCES,
        "selection_uses_targets": False,
    }
    export_benchmark_manifest(
        selected_dataset,  # type: ignore[arg-type]
        source_indices=source_indices,
        task="docvqa",
        dataset_id=str(dataset["dataset_id"]),
        dataset_revision=str(dataset["dataset_revision"]),
        dataset_split=str(dataset["split"]),
        output_dir=staging_dir,
        seed=SEED,
        state_namespace=RESERVE_STATE_NAMESPACE,
        selection="outcome-sealed DocVQA ToolGate comparator reserve suffix",
        selection_metadata=selection_metadata,
    )
    provenance_path = staging_dir / "manifest.provenance.json"
    provenance = _load(provenance_path, "manifest provenance")
    provenance.update(
        {
            "code_revision": revision,
            "freeze_sha256": args.expected_freeze_sha256,
            "source_parquet_files": [str(path) for path in parquet_paths],
            "source_parquet_sha256": parquet_hashes,
            "row_identity_audit": row_audit,
            "reserve_outcomes_collected": False,
        }
    )
    provenance_path.write_text(
        json.dumps(provenance, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_audit = audit_manifest(staging_dir)
    expected_sources = {f"docvqa:{source}" for source in reserve_sources}
    expected_images = {str(item["image_id"]) for item in identities}
    if (
        manifest_audit.get("unique_sources") != RESERVE_SOURCES
        or manifest_audit.get("unique_images") != RESERVE_SOURCES
        or manifest_audit.get("_sources") != expected_sources
        or manifest_audit.get("_images") != expected_images
    ):
        raise ValueError("reserve manifest identity audit failed")
    clean_manifest_audit = {
        key: value for key, value in manifest_audit.items() if not key.startswith("_")
    }
    clean_manifest_audit["root"] = str(output_dir)
    audit_document = {
        "passed": True,
        "scientific_status": (
            "outcome-sealed DocVQA reserve manifest bound to frozen comparator"
        ),
        "code_revision": revision,
        "freeze": str(freeze_path),
        "freeze_sha256": args.expected_freeze_sha256,
        "manifest": clean_manifest_audit,
        "selection": selection_metadata,
        "row_identity_audit": row_audit,
        "reserve_outcomes_collected": False,
    }
    _publish_json(staging_audit, audit_document)
    os.replace(staging_dir, output_dir)
    os.replace(staging_audit, audit_output)
    print(
        json.dumps(
            {
                "manifest": str(output_dir / "manifest.jsonl"),
                "manifest_sha256": sha256_file(output_dir / "manifest.jsonl"),
                "manifest_provenance_sha256": sha256_file(
                    output_dir / "manifest.provenance.json"
                ),
                "audit": str(audit_output),
                "audit_sha256": sha256_file(audit_output),
                "sources": RESERVE_SOURCES,
                "states": clean_manifest_audit["count"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
