from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

from beyond_entropy.factorized_formal import (
    ALLOCATION_AUDIT_SHA256,
    ALLOCATION_SHA256,
    FORMAL_SOURCES,
    check_hash,
    load_mapping,
    sha256_file,
    validate_policy_freeze,
)
from beyond_entropy.manifest_audit import audit_manifest
from beyond_entropy.manifest_export import export_benchmark_manifest


REVISION = "9c0699cd19768ac5ab97568f6b3cbac4c0062884"
NAMESPACE = "beyond-entropy-textvqa-train-scale-v1"
SEED = 20260828


def _clean_audit(report: Mapping[str, Any], final_root: Path) -> dict[str, Any]:
    clean = {name: value for name, value in report.items() if not name.startswith("_")}
    clean["root"] = str(final_root.resolve())
    return clean


def _assignment_identities(role: Mapping[str, Any], expected: int) -> tuple[set[str], set[str]]:
    assignments = role.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != expected:
        raise ValueError(f"allocation role must contain exactly {expected} assignments")
    groups = {str(item["source_group_id"]) for item in assignments}
    images = {str(item["image_id"]) for item in assignments}
    if len(groups) != expected or len(images) != expected:
        raise ValueError("allocation role contains duplicate source or RGB identities")
    return groups, images


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize the sealed 5,953-source factorized-v2 formal role"
    )
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--allocation-audit", type=Path, required=True)
    parser.add_argument("--policy-freeze", type=Path, required=True)
    parser.add_argument("--expected-policy-freeze-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parents[1]
    check_hash(args.allocation, ALLOCATION_SHA256, "allocation")
    check_hash(args.allocation_audit, ALLOCATION_AUDIT_SHA256, "allocation audit")
    check_hash(
        args.policy_freeze,
        args.expected_policy_freeze_sha256,
        "policy freeze",
    )
    freeze = load_mapping(args.policy_freeze, "policy freeze")
    validate_policy_freeze(freeze)
    code_revision = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if code_revision != freeze.get("code_revision"):
        raise ValueError("repository revision differs from factorized policy freeze")
    tracked_status = subprocess.run(
        ["git", "-C", str(repo_dir), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if tracked_status.strip():
        raise ValueError("tracked worktree must be clean before formal export")
    temporary_dir = args.output_dir.with_name(args.output_dir.name + ".tmp")
    temporary_audit = args.audit_output.with_suffix(args.audit_output.suffix + ".tmp")
    if (
        args.output_dir.exists()
        or args.audit_output.exists()
        or temporary_dir.exists()
        or temporary_audit.exists()
    ):
        raise FileExistsError("formal export or temporary destination already exists")

    allocation = load_mapping(args.allocation, "allocation")
    allocation_audit = load_mapping(args.allocation_audit, "allocation audit")
    if allocation_audit.get("passed") is not True or (
        allocation_audit.get("formal_outcomes_collected") is not False
    ):
        raise ValueError("allocation audit does not preserve the sealed formal role")
    contract = allocation.get("selection_contract")
    dataset_metadata = allocation.get("dataset")
    allocation_body = allocation.get("allocation")
    if not isinstance(contract, Mapping) or not isinstance(dataset_metadata, Mapping):
        raise ValueError("allocation is missing selection or dataset metadata")
    if not isinstance(allocation_body, Mapping):
        raise ValueError("allocation body must be a mapping")
    if (
        contract.get("selection_target_fields_accessed") is not False
        or contract.get("formal_manifest_exported") is not False
        or contract.get("formal_rollouts_collected") is not False
    ):
        raise ValueError("allocation no longer describes an untouched formal role")
    if (
        dataset_metadata.get("dataset_id") != "lmms-lab/textvqa"
        or dataset_metadata.get("dataset_revision") != REVISION
        or dataset_metadata.get("split") != "train"
        or allocation_body.get("namespace") != NAMESPACE
        or allocation_body.get("seed") != SEED
    ):
        raise ValueError("allocation dataset or namespace contract mismatch")

    roles = allocation_body.get("roles")
    if not isinstance(roles, Mapping):
        raise ValueError("allocation roles must be a mapping")
    calibration_role = roles.get("risk_calibration")
    formal_role = roles.get("formal_test")
    if not isinstance(calibration_role, Mapping) or not isinstance(
        formal_role, Mapping
    ):
        raise ValueError("allocation is missing calibration or formal role")
    calibration_groups, calibration_images = _assignment_identities(
        calibration_role, 3000
    )
    formal_groups, formal_images = _assignment_identities(
        formal_role, FORMAL_SOURCES
    )
    if formal_groups & calibration_groups or formal_images & calibration_images:
        raise ValueError("formal role overlaps fresh calibration identities")

    parent_path = Path(str(allocation.get("parent_allocation", ""))).resolve()
    parent_expected = str(allocation.get("parent_allocation_sha256", ""))
    check_hash(parent_path, parent_expected, "parent allocation")
    parent = load_mapping(parent_path, "parent allocation")
    parent_roles = parent.get("allocation", {}).get("roles")
    if not isinstance(parent_roles, Mapping):
        raise ValueError("parent allocation roles are invalid")
    parent_groups: set[str] = set()
    parent_images: set[str] = set()
    for role in parent_roles.values():
        if not isinstance(role, Mapping):
            raise ValueError("parent allocation role is invalid")
        groups, images = _assignment_identities(role, int(role.get("count", -1)))
        parent_groups.update(groups)
        parent_images.update(images)
    if formal_groups & parent_groups or formal_images & parent_images:
        raise ValueError("formal role overlaps an earlier allocation identity")

    parquet_paths = [
        Path(path).resolve() for path in dataset_metadata.get("parquet_files", [])
    ]
    parquet_hashes = [
        str(value) for value in dataset_metadata.get("parquet_sha256", [])
    ]
    if not parquet_paths or len(parquet_paths) != len(parquet_hashes):
        raise ValueError("allocation Parquet provenance is incomplete")
    for path, expected in zip(parquet_paths, parquet_hashes):
        check_hash(path, expected, "source Parquet")

    try:
        from datasets import Dataset  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SystemExit("Install benchmark dependencies before formal export") from exc
    dataset = Dataset.from_parquet([str(path) for path in parquet_paths])
    if len(dataset) != int(dataset_metadata.get("row_count", -1)):
        raise ValueError("source Parquet row count differs from allocation")
    group_ids = [str(value).strip() for value in dataset["image_id"]]
    source_indices = [
        index for index, group_id in enumerate(group_ids) if group_id in formal_groups
    ]
    if {group_ids[index] for index in source_indices} != formal_groups:
        raise ValueError("formal source groups do not match the source dataset")
    selected_dataset = dataset.select(source_indices)
    result = export_benchmark_manifest(
        selected_dataset,
        source_indices=source_indices,
        task="textvqa",
        dataset_id="lmms-lab/textvqa",
        dataset_revision=REVISION,
        dataset_split="train",
        output_dir=temporary_dir,
        seed=SEED,
        state_namespace="textvqa-train-factorized-v2-formal",
        selection=(
            "sealed fixed-sequence factorized reserve opened after successful calibration"
        ),
        selection_metadata={
            "allocation": str(args.allocation.resolve()),
            "allocation_sha256": ALLOCATION_SHA256,
            "policy_freeze": str(args.policy_freeze.resolve()),
            "policy_freeze_sha256": args.expected_policy_freeze_sha256,
            "namespace": NAMESPACE,
            "role": "formal_test",
            "source_rank_start": 16000,
            "source_rank_end_exclusive": 21953,
            "selected_source_group_count": FORMAL_SOURCES,
            "selection_uses_targets": False,
        },
    )
    result["manifest"] = str((args.output_dir / "manifest.jsonl").resolve())
    result["source_parquet_files"] = [str(path) for path in parquet_paths]
    result["source_parquet_sha256"] = parquet_hashes
    provenance_path = temporary_dir / "manifest.provenance.json"
    provenance_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit = audit_manifest(temporary_dir)
    if audit["_images"] != formal_images:
        raise RuntimeError("formal export images differ from allocation")
    expected_sources = {f"textvqa:{group_id}" for group_id in formal_groups}
    if audit["_sources"] != expected_sources:
        raise RuntimeError("formal export sources differ from allocation")
    clean_audit = _clean_audit(audit, args.output_dir)
    audit_document = {
        "passed": True,
        "scientific_status": (
            "outcome-unseen factorized-v2 formal manifest bound to the frozen policy"
        ),
        "allocation": str(args.allocation.resolve()),
        "allocation_sha256": ALLOCATION_SHA256,
        "allocation_audit_sha256": ALLOCATION_AUDIT_SHA256,
        "policy_freeze": str(args.policy_freeze.resolve()),
        "policy_freeze_sha256": args.expected_policy_freeze_sha256,
        "formal": clean_audit,
        "overlap": {
            "formal_fresh_calibration_images": 0,
            "formal_fresh_calibration_sources": 0,
            "formal_parent_images": 0,
            "formal_parent_sources": 0,
        },
        "formal_outcomes_collected": False,
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    with temporary_audit.open("x", encoding="utf-8") as handle:
        json.dump(audit_document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_dir, args.output_dir)
    os.replace(temporary_audit, args.audit_output)
    print(
        json.dumps(
            {
                "manifest": str((args.output_dir / "manifest.jsonl").resolve()),
                "manifest_sha256": clean_audit["manifest_sha256"],
                "manifest_provenance_sha256": sha256_file(
                    args.output_dir / "manifest.provenance.json"
                ),
                "audit": str(args.audit_output.resolve()),
                "audit_sha256": sha256_file(args.audit_output),
                "states": clean_audit["count"],
                "sources": clean_audit["unique_sources"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
