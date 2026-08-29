from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

from beyond_entropy.docvqa_formal import (
    check_hash,
    validate_policy_freeze,
)
from beyond_entropy.docvqa_formal_export import (
    FORMAL_ROLE,
    FORMAL_STATE_NAMESPACE,
    formal_role_identity_map,
    validate_formal_manifest_audit,
    validate_formal_rows,
    validate_sealed_formal_allocation,
)
from beyond_entropy.docvqa_train_allocation import (
    NAMESPACE,
    PROTOCOL_SHA256,
    SEED,
    load_docvqa_source_images,
    load_prior_identities,
    sha256_file,
    verify_recomputed_allocation_bundle,
)
from beyond_entropy.manifest_audit import audit_manifest
from beyond_entropy.manifest_export import export_benchmark_manifest


def _load_mapping(path: Path, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"DocVQA formal {name} must be a JSON object")
    return payload


def _require(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"DocVQA formal export mismatch for {name}")


def _frozen_component(
    freeze: Mapping[str, Any],
    name: str,
    supplied_path: Path,
) -> str:
    artifacts = freeze.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("DocVQA policy freeze lacks artifacts")
    component = artifacts.get(name)
    if not isinstance(component, Mapping):
        raise ValueError(f"DocVQA policy freeze lacks artifact {name}")
    frozen_path = Path(str(component.get("path", ""))).resolve()
    expected_hash = str(component.get("sha256", ""))
    _require(frozen_path, supplied_path.resolve(), f"{name} path")
    check_hash(supplied_path.resolve(), expected_hash, name)
    return expected_hash


def _git_revision(repo_dir: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean(repo_dir: Path) -> None:
    status = subprocess.run(
        ["git", "-C", str(repo_dir), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("tracked worktree must be clean before DocVQA formal export")


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the sealed 3,500-source DocVQA-train formal role only "
            "after successful policy freeze"
        )
    )
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--allocation-audit", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--policy-freeze", type=Path, required=True)
    parser.add_argument("--expected-policy-freeze-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parents[1]
    allocation_path = args.allocation.resolve()
    allocation_audit_path = args.allocation_audit.resolve()
    protocol_path = args.protocol.resolve()
    policy_freeze_path = args.policy_freeze.resolve()
    output_dir = args.output_dir.resolve()
    audit_output = args.audit_output.resolve()
    staging_dir = output_dir.with_name(output_dir.name + ".tmp")
    staging_audit = audit_output.with_suffix(audit_output.suffix + ".tmp")
    for destination in (output_dir, audit_output, staging_dir, staging_audit):
        if destination.exists():
            raise FileExistsError(
                f"DocVQA formal output or staging destination exists: {destination}"
            )

    check_hash(
        policy_freeze_path,
        args.expected_policy_freeze_sha256,
        "policy freeze",
    )
    freeze = _load_mapping(policy_freeze_path, "policy freeze")
    validate_policy_freeze(freeze)
    code_revision = _git_revision(repo_dir)
    _require(code_revision, freeze.get("code_revision"), "code revision")
    _require_clean(repo_dir)
    allocation_sha256 = _frozen_component(freeze, "allocation", allocation_path)
    allocation_audit_sha256 = _frozen_component(
        freeze,
        "allocation_audit",
        allocation_audit_path,
    )
    _require(
        _frozen_component(freeze, "protocol", protocol_path),
        PROTOCOL_SHA256,
        "protocol SHA-256",
    )

    allocation = _load_mapping(allocation_path, "allocation")
    allocation_audit = _load_mapping(allocation_audit_path, "allocation audit")
    formal_identities = validate_sealed_formal_allocation(
        allocation,
        allocation_audit,
        allocation_sha256=allocation_sha256,
    )
    dataset_metadata = allocation.get("dataset")
    if not isinstance(dataset_metadata, Mapping):
        raise ValueError("DocVQA formal allocation lacks dataset metadata")
    parquet_paths = [
        Path(str(value)).resolve()
        for value in dataset_metadata.get("parquet_files", [])
    ]
    parquet_sha256 = [
        str(value) for value in dataset_metadata.get("parquet_sha256", [])
    ]
    if not parquet_paths or len(parquet_paths) != len(parquet_sha256):
        raise ValueError("DocVQA formal Parquet provenance is incomplete")
    for path, expected in zip(parquet_paths, parquet_sha256):
        check_hash(path, expected, "source Parquet")

    prior_banks = allocation.get("prior_banks")
    if not isinstance(prior_banks, list) or any(
        not isinstance(record, Mapping) for record in prior_banks
    ):
        raise ValueError("DocVQA formal prior-bank provenance is invalid")
    prior_manifest_paths = [
        Path(str(record.get("manifest", ""))).resolve() for record in prior_banks
    ]
    for path, record in zip(prior_manifest_paths, prior_banks):
        check_hash(path, str(record.get("manifest_sha256", "")), "prior manifest")
    prior_images, prior_sources, recomputed_prior = load_prior_identities(
        prior_manifest_paths,
        verify_images=True,
    )
    if recomputed_prior != prior_banks:
        raise ValueError("DocVQA formal prior-bank inventory changed")
    source_images, row_count = load_docvqa_source_images(parquet_paths)
    _require(row_count, dataset_metadata.get("row_count"), "dataset row count")
    _require(
        len(source_images),
        dataset_metadata.get("source_group_count"),
        "dataset source count",
    )
    verify_recomputed_allocation_bundle(
        allocation,
        allocation_audit,
        source_images=source_images,
        excluded_image_ids=prior_images,
        excluded_source_group_ids=prior_sources,
        prior_banks=recomputed_prior,
        parquet_files=parquet_paths,
        parquet_sha256=parquet_sha256,
        row_count=row_count,
        protocol_path=protocol_path,
        allocation_path=allocation_path,
        allocation_sha256=allocation_sha256,
    )

    try:
        from datasets import Dataset  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SystemExit("Install benchmark dependencies before formal export") from exc
    formal_sources = set(formal_identities)
    identity_dataset = Dataset.from_parquet(
        [str(path) for path in parquet_paths],
        columns=["docId"],
    )
    group_ids = [str(value).strip() for value in identity_dataset["docId"]]
    source_indices = [
        index for index, group_id in enumerate(group_ids) if group_id in formal_sources
    ]
    if {group_ids[index] for index in source_indices} != formal_sources:
        raise ValueError("DocVQA formal source groups do not match source Parquet")
    selected_dataset = Dataset.from_parquet(
        [str(path) for path in parquet_paths],
        filters=[("docId", "in", sorted(formal_sources))],
    )
    selected_group_ids = [str(value).strip() for value in selected_dataset["docId"]]
    if selected_group_ids != [group_ids[index] for index in source_indices]:
        raise ValueError("DocVQA filtered formal row order differs from source Parquet")
    row_identity_audit = validate_formal_rows(
        selected_dataset,  # type: ignore[arg-type]
        allocation,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir()
    selection_metadata = {
        "allocation": str(allocation_path),
        "allocation_sha256": allocation_sha256,
        "allocation_audit_sha256": allocation_audit_sha256,
        "policy_freeze": str(policy_freeze_path),
        "policy_freeze_sha256": args.expected_policy_freeze_sha256,
        "protocol_sha256": PROTOCOL_SHA256,
        "namespace": NAMESPACE,
        "role": FORMAL_ROLE,
        "selected_source_group_count": len(formal_sources),
        "selection_uses_targets": False,
    }
    export_benchmark_manifest(
        selected_dataset,  # type: ignore[arg-type]
        source_indices=source_indices,
        task="docvqa",
        dataset_id=str(dataset_metadata["dataset_id"]),
        dataset_revision=str(dataset_metadata["dataset_revision"]),
        dataset_split=str(dataset_metadata["split"]),
        output_dir=staging_dir,
        seed=SEED,
        state_namespace=FORMAL_STATE_NAMESPACE,
        selection=(
            "sealed DocVQA formal role opened after successful fixed-sequence calibration"
        ),
        selection_metadata=selection_metadata,
    )
    provenance_path = staging_dir / "manifest.provenance.json"
    provenance = _load_mapping(provenance_path, "manifest provenance")
    provenance.update(
        {
            "source_parquet_files": [str(path) for path in parquet_paths],
            "source_parquet_sha256": parquet_sha256,
            "allocation_audit_sha256": allocation_audit_sha256,
            "policy_freeze_sha256": args.expected_policy_freeze_sha256,
            "code_revision": code_revision,
            "formal_outcomes_collected": False,
            "row_identity_audit": row_identity_audit,
        }
    )
    provenance_path.write_text(
        json.dumps(provenance, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    raw_manifest_audit = audit_manifest(staging_dir)
    clean_manifest_audit = validate_formal_manifest_audit(
        raw_manifest_audit,
        allocation,
        allocation_sha256=allocation_sha256,
        allocation_audit_sha256=allocation_audit_sha256,
        policy_freeze_sha256=args.expected_policy_freeze_sha256,
    )
    clean_manifest_audit["root"] = str(output_dir)
    allocation_overlap = allocation_audit.get("overlap")
    assert isinstance(allocation_overlap, Mapping)
    audit_document = {
        "passed": True,
        "scientific_status": (
            "outcome-unseen DocVQA-train formal manifest bound to the frozen policy"
        ),
        "code_revision": code_revision,
        "allocation": str(allocation_path),
        "allocation_sha256": allocation_sha256,
        "allocation_audit_sha256": allocation_audit_sha256,
        "policy_freeze": str(policy_freeze_path),
        "policy_freeze_sha256": args.expected_policy_freeze_sha256,
        "formal": clean_manifest_audit,
        "overlap": dict(allocation_overlap),
        "row_identity_audit": row_identity_audit,
        "formal_outcomes_collected": False,
    }
    _write_json_exclusive(staging_audit, audit_document)
    os.replace(staging_dir, output_dir)
    os.replace(staging_audit, audit_output)

    published = validate_formal_manifest_audit(
        audit_manifest(output_dir),
        allocation,
        allocation_sha256=allocation_sha256,
        allocation_audit_sha256=allocation_audit_sha256,
        policy_freeze_sha256=args.expected_policy_freeze_sha256,
    )
    published["root"] = str(output_dir)
    if published != clean_manifest_audit:
        raise RuntimeError("DocVQA formal manifest changed during publication")
    print(
        json.dumps(
            {
                "manifest": str((output_dir / "manifest.jsonl").resolve()),
                "manifest_sha256": clean_manifest_audit["manifest_sha256"],
                "manifest_provenance_sha256": sha256_file(
                    output_dir / "manifest.provenance.json"
                ),
                "audit": str(audit_output),
                "audit_sha256": sha256_file(audit_output),
                "states": clean_manifest_audit["count"],
                "sources": clean_manifest_audit["unique_sources"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
