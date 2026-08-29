from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

from beyond_entropy.docvqa_calibration import validate_calibration_preoutcome_gate
from beyond_entropy.docvqa_candidate_freeze import validate_candidate_freeze_gate
from beyond_entropy.docvqa_manifest_export import (
    EXPORTABLE_ROLES,
    ROLE_STATE_NAMESPACES,
    role_identity_map,
    validate_docvqa_role_rows,
    validate_exported_docvqa_manifest,
)
from beyond_entropy.docvqa_train_allocation import (
    DATASET_ID,
    DATASET_REVISION,
    DATASET_SPLIT,
    NAMESPACE,
    PROTOCOL_SHA256,
    SEED,
    discover_prior_manifests,
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
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _require(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"DocVQA manifest export mismatch for {name}")


def _require_hash(path: Path, expected: str, name: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"DocVQA manifest export input is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"DocVQA manifest export {name} SHA-256 mismatch")
    return actual


def _tracked_revision(repo_dir: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("tracked worktree must be clean before DocVQA manifest export")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_unmaterialized(path: Path, name: str) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"{name} must remain unmaterialized")


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export one outcome-ordered DocVQA-train development manifest while "
            "keeping formal identities sealed"
        )
    )
    parser.add_argument("--role", choices=sorted(EXPORTABLE_ROLES), required=True)
    parser.add_argument("--parquet-file", type=Path, action="append", required=True)
    parser.add_argument("--prior-manifest-root", type=Path, action="append", default=[])
    parser.add_argument("--prior-manifest", type=Path, action="append", default=[])
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--allocation-audit", type=Path, required=True)
    parser.add_argument("--expected-allocation-sha256", required=True)
    parser.add_argument("--expected-allocation-audit-sha256", required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--candidate-audit", type=Path)
    parser.add_argument("--expected-candidate-sha256")
    parser.add_argument("--expected-candidate-audit-sha256")
    parser.add_argument("--ranker-manifest-dir", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-code-revision", required=True)
    parser.add_argument("--formal-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    candidate_values = (
        args.candidate,
        args.candidate_audit,
        args.expected_candidate_sha256,
        args.expected_candidate_audit_sha256,
    )
    if args.role == "risk_calibration":
        if any(value is None for value in candidate_values):
            parser.error("risk_calibration requires candidate paths and hashes")
        if args.ranker_manifest_dir is None:
            parser.error("risk_calibration requires --ranker-manifest-dir")
    elif any(value is not None for value in candidate_values):
        parser.error("ranker_training must be exported before any candidate exists")
    elif args.ranker_manifest_dir is not None:
        parser.error("ranker_training cannot consume itself as a prior role")

    allocation_path = args.allocation.resolve()
    allocation_audit_path = args.allocation_audit.resolve()
    protocol_path = args.protocol.resolve()
    output_dir = args.output_dir.resolve()
    staging_dir = output_dir.with_name(output_dir.name + ".tmp")
    formal_output_dir = args.formal_output_dir.resolve()
    if output_dir.exists() or staging_dir.exists():
        raise FileExistsError("DocVQA manifest output or staging path already exists")
    _require_unmaterialized(formal_output_dir, "DocVQA formal output")

    allocation_sha256 = _require_hash(
        allocation_path,
        args.expected_allocation_sha256,
        "allocation",
    )
    allocation_audit_sha256 = _require_hash(
        allocation_audit_path,
        args.expected_allocation_audit_sha256,
        "allocation audit",
    )
    _require_hash(protocol_path, PROTOCOL_SHA256, "protocol")
    allocation = _load_mapping(allocation_path, "allocation")
    allocation_audit = _load_mapping(allocation_audit_path, "allocation audit")
    repo_dir = Path(__file__).resolve().parents[1]
    code_revision = _tracked_revision(repo_dir)
    _require(code_revision, args.expected_code_revision, "code revision")
    _require(allocation.get("code_revision"), code_revision, "allocation code revision")

    candidate: dict[str, Any] | None = None
    candidate_audit: dict[str, Any] | None = None
    candidate_sha256: str | None = None
    candidate_audit_sha256: str | None = None
    if args.role == "risk_calibration":
        assert args.candidate is not None
        assert args.candidate_audit is not None
        assert args.expected_candidate_sha256 is not None
        assert args.expected_candidate_audit_sha256 is not None
        candidate_path = args.candidate.resolve()
        candidate_audit_path = args.candidate_audit.resolve()
        candidate_sha256 = _require_hash(
            candidate_path,
            args.expected_candidate_sha256,
            "candidate",
        )
        candidate_audit_sha256 = _require_hash(
            candidate_audit_path,
            args.expected_candidate_audit_sha256,
            "candidate audit",
        )
        candidate = _load_mapping(candidate_path, "candidate")
        candidate_audit = _load_mapping(candidate_audit_path, "candidate audit")
        validate_calibration_preoutcome_gate(
            candidate,
            candidate_audit,
            allocation,
            allocation_audit,
            candidate_sha256=candidate_sha256,
            allocation_sha256=allocation_sha256,
            code_revision=code_revision,
        )
    else:
        validate_candidate_freeze_gate(
            allocation,
            allocation_audit,
            allocation_sha256=allocation_sha256,
        )

    parquet_paths = [path.resolve() for path in args.parquet_file]
    if any(not path.is_file() for path in parquet_paths):
        raise FileNotFoundError("one or more DocVQA train Parquet shards do not exist")
    parquet_sha256 = [sha256_file(path) for path in parquet_paths]
    excluded_roots = [output_dir, staging_dir, formal_output_dir]
    if args.ranker_manifest_dir is not None:
        excluded_roots.append(args.ranker_manifest_dir.resolve())
    prior_manifests = discover_prior_manifests(
        args.prior_manifest_root,
        args.prior_manifest,
        excluded_roots=excluded_roots,
    )
    prior_images, prior_sources, prior_records = load_prior_identities(
        prior_manifests,
        verify_images=True,
    )
    source_images, row_count = load_docvqa_source_images(parquet_paths)
    verify_recomputed_allocation_bundle(
        allocation,
        allocation_audit,
        source_images=source_images,
        excluded_image_ids=prior_images,
        excluded_source_group_ids=prior_sources,
        prior_banks=prior_records,
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
        raise SystemExit("Install benchmark dependencies before export") from exc
    dataset = Dataset.from_parquet([str(path) for path in parquet_paths])
    group_ids = [str(value).strip() for value in dataset["docId"]]
    selected_sources = set(role_identity_map(allocation, args.role))
    source_indices = [
        index for index, group_id in enumerate(group_ids) if group_id in selected_sources
    ]
    selected_dataset = dataset.select(source_indices)
    row_identity_audit = validate_docvqa_role_rows(
        selected_dataset,  # type: ignore[arg-type]
        allocation,
        args.role,
    )

    ranker_manifest_sha256: str | None = None
    if args.role == "risk_calibration":
        assert args.ranker_manifest_dir is not None
        ranker_dir = args.ranker_manifest_dir.resolve()
        ranker_audit = audit_manifest(ranker_dir)
        validate_exported_docvqa_manifest(
            ranker_audit,
            allocation,
            "ranker_training",
            allocation_sha256=allocation_sha256,
            allocation_audit_sha256=allocation_audit_sha256,
        )
        ranker_manifest_sha256 = str(ranker_audit["manifest_sha256"])

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir()
    selection_metadata: dict[str, Any] = {
        "allocation": str(allocation_path),
        "allocation_sha256": allocation_sha256,
        "allocation_audit_sha256": allocation_audit_sha256,
        "protocol_sha256": PROTOCOL_SHA256,
        "namespace": NAMESPACE,
        "role": args.role,
        "selected_source_group_count": len(selected_sources),
        "selection_uses_targets": False,
    }
    if candidate_sha256 is not None:
        selection_metadata["candidate_sha256"] = candidate_sha256
    export_benchmark_manifest(
        selected_dataset,  # type: ignore[arg-type]
        source_indices=source_indices,
        task="docvqa",
        dataset_id=DATASET_ID,
        dataset_revision=DATASET_REVISION,
        dataset_split=DATASET_SPLIT,
        output_dir=staging_dir,
        seed=SEED,
        state_namespace=ROLE_STATE_NAMESPACES[args.role],
        selection="frozen DocVQA source-role membership; target-independent",
        selection_metadata=selection_metadata,
    )
    provenance_path = staging_dir / "manifest.provenance.json"
    provenance = _load_mapping(provenance_path, "manifest provenance")
    provenance.update(
        {
            "source_parquet_files": [str(path) for path in parquet_paths],
            "source_parquet_sha256": parquet_sha256,
            "allocation_audit_sha256": allocation_audit_sha256,
            "candidate_audit_sha256": candidate_audit_sha256,
            "code_revision": code_revision,
            "formal_manifest_exported": False,
        }
    )
    provenance_path.write_text(
        json.dumps(provenance, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    raw_manifest_audit = audit_manifest(staging_dir)
    manifest_audit = validate_exported_docvqa_manifest(
        raw_manifest_audit,
        allocation,
        args.role,
        allocation_sha256=allocation_sha256,
        allocation_audit_sha256=allocation_audit_sha256,
        candidate_sha256=candidate_sha256,
    )
    manifest_audit["manifest"]["root"] = str(output_dir)
    audit_document = {
        **manifest_audit,
        "code_revision": code_revision,
        "allocation": str(allocation_path),
        "allocation_audit": str(allocation_audit_path),
        "candidate_audit_sha256": candidate_audit_sha256,
        "manifest_provenance_sha256": sha256_file(provenance_path),
        "ranker_manifest_sha256": ranker_manifest_sha256,
        "source_parquet_sha256": parquet_sha256,
        "row_identity_audit": row_identity_audit,
        "ranker_targets_materialized": args.role == "ranker_training",
        "calibration_targets_materialized": args.role == "risk_calibration",
        "formal_targets_materialized": False,
    }
    _write_json_exclusive(staging_dir / "manifest.audit.json", audit_document)
    os.replace(staging_dir, output_dir)

    post_audit = validate_exported_docvqa_manifest(
        audit_manifest(output_dir),
        allocation,
        args.role,
        allocation_sha256=allocation_sha256,
        allocation_audit_sha256=allocation_audit_sha256,
        candidate_sha256=candidate_sha256,
    )
    if post_audit["manifest"] != audit_document["manifest"]:
        raise RuntimeError("DocVQA manifest changed during atomic publication")
    _require_unmaterialized(formal_output_dir, "DocVQA formal output")
    print(json.dumps(audit_document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
