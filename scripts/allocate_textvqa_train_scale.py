from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from beyond_entropy.manifest_audit import audit_manifest
from beyond_entropy.manifest_export import export_benchmark_manifest, image_digest
from beyond_entropy.source_allocation import SourceRoleSpec, allocate_source_roles


REVISION = "9c0699cd19768ac5ab97568f6b3cbac4c0062884"
NAMESPACE = "beyond-entropy-textvqa-train-scale-v1"
SEED = 20260828
ROLE_SPECS = (
    SourceRoleSpec("ranker_training", 0, 5000),
    SourceRoleSpec("risk_calibration", 5000, 3000),
    SourceRoleSpec("formal_test", 8000, 5000),
)
HEX_DIGEST = re.compile(r"[0-9a-f]{64}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_beneath(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _discover_prior_manifests(
    roots: Sequence[Path],
    explicit: Sequence[Path],
    *,
    excluded_roots: Sequence[Path],
) -> list[Path]:
    manifests = {path.resolve() for path in explicit}
    for root in roots:
        manifests.update(path.resolve() for path in root.resolve().rglob("manifest.jsonl"))
    selected = sorted(
        path
        for path in manifests
        if not any(_is_beneath(path, root.resolve()) for root in excluded_roots)
    )
    if not selected:
        raise ValueError("at least one prior manifest is required for isolation audit")
    missing = [path for path in selected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"prior manifest does not exist: {missing[0]}")
    return selected


def _load_prior_identities(
    manifest_paths: Sequence[Path],
    *,
    verify_images: bool,
) -> tuple[set[str], set[str], list[dict[str, Any]]]:
    image_paths: dict[str, Path] = {}
    image_ids: set[str] = set()
    textvqa_source_groups: set[str] = set()
    manifest_records: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        row_count = 0
        manifest_image_ids: set[str] = set()
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise ValueError(
                        f"prior manifest row must be a mapping: {manifest_path}:{line_number}"
                    )
                image_id = str(payload.get("image_id", "")).strip()
                relative_image = str(payload.get("image_path", "")).strip()
                source_id = str(payload.get("source_id", "")).strip()
                if not HEX_DIGEST.fullmatch(image_id) or not relative_image or not source_id:
                    raise ValueError(
                        f"invalid prior identity fields: {manifest_path}:{line_number}"
                    )
                resolved_image = (manifest_path.parent / relative_image).resolve()
                if not resolved_image.is_file():
                    raise FileNotFoundError(
                        f"prior manifest image does not exist: {resolved_image}"
                    )
                image_paths.setdefault(image_id, resolved_image)
                image_ids.add(image_id)
                manifest_image_ids.add(image_id)
                if source_id.startswith("textvqa:"):
                    textvqa_source_groups.add(source_id.removeprefix("textvqa:"))
                row_count += 1
        manifest_records.append(
            {
                "manifest": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "row_count": row_count,
                "unique_image_count": len(manifest_image_ids),
            }
        )
    if verify_images:
        from PIL import Image

        total = len(image_paths)
        for index, (declared_id, image_path) in enumerate(sorted(image_paths.items()), 1):
            with Image.open(image_path) as raw_image:
                actual_id = image_digest(raw_image.convert("RGB"))
            if actual_id != declared_id:
                raise ValueError(
                    f"decoded-RGB digest mismatch for prior image {image_path}"
                )
            if index % 500 == 0 or index == total:
                print(f"verified prior RGB identities: {index}/{total}", flush=True)
    return image_ids, textvqa_source_groups, manifest_records


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _role_assignments(allocation: Mapping[str, Any], role: str) -> list[Mapping[str, Any]]:
    roles = allocation["roles"]
    assert isinstance(roles, Mapping)
    role_payload = roles[role]
    assert isinstance(role_payload, Mapping)
    assignments = role_payload["assignments"]
    assert isinstance(assignments, list)
    return assignments


def _prepare_export_destination(path: Path, *, resume: bool) -> None:
    if path.exists() and any(path.iterdir()) and not resume:
        raise FileExistsError(
            f"export destination is non-empty; pass --resume to reuse it: {path}"
        )


def _export_role(
    dataset: Any,
    group_ids: Sequence[str],
    allocation: Mapping[str, Any],
    *,
    role: str,
    state_namespace: str,
    output_dir: Path,
    allocation_path: Path,
    allocation_sha256: str,
    parquet_paths: Sequence[Path],
    parquet_sha256: Sequence[str],
) -> dict[str, Any]:
    assignments = _role_assignments(allocation, role)
    selected_groups = {str(item["source_group_id"]) for item in assignments}
    source_indices = [
        index for index, group_id in enumerate(group_ids) if group_id in selected_groups
    ]
    selected_dataset = dataset.select(source_indices)
    result = export_benchmark_manifest(
        selected_dataset,
        source_indices=source_indices,
        task="textvqa",
        dataset_id="lmms-lab/textvqa",
        dataset_revision=REVISION,
        dataset_split="train",
        output_dir=output_dir,
        seed=SEED,
        state_namespace=state_namespace,
        selection=(
            "preregistered SHA-256 source-rank role with prior/source/RGB "
            "exclusion and reserve backfill"
        ),
        selection_metadata={
            "allocation": str(allocation_path),
            "allocation_sha256": allocation_sha256,
            "namespace": NAMESPACE,
            "role": role,
            "selected_source_group_count": len(selected_groups),
            "selection_uses_targets": False,
        },
    )
    result["source_parquet_files"] = [str(path) for path in parquet_paths]
    result["source_parquet_sha256"] = list(parquet_sha256)
    _write_json(output_dir / "manifest.provenance.json", result)
    return result


def _clean_audit(report: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if not key.startswith("_")}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze outcome-independent TextVQA train source roles"
    )
    parser.add_argument("--parquet-file", type=Path, action="append", required=True)
    parser.add_argument("--prior-manifest-root", type=Path, action="append", default=[])
    parser.add_argument("--prior-manifest", type=Path, action="append", default=[])
    parser.add_argument("--allocation-output", type=Path, required=True)
    parser.add_argument("--ranker-output-dir", type=Path)
    parser.add_argument("--calibration-output-dir", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--skip-prior-image-verification", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    export_paths = (args.ranker_output_dir, args.calibration_output_dir, args.audit_output)
    if any(path is not None for path in export_paths) and not all(
        path is not None for path in export_paths
    ):
        parser.error(
            "--ranker-output-dir, --calibration-output-dir, and --audit-output "
            "must be provided together"
        )
    parquet_paths = [path.resolve() for path in args.parquet_file]
    if any(not path.is_file() for path in parquet_paths):
        raise FileNotFoundError("one or more TextVQA Parquet shards do not exist")
    excluded_roots = [
        path.resolve()
        for path in (args.ranker_output_dir, args.calibration_output_dir)
        if path is not None
    ]
    prior_manifests = _discover_prior_manifests(
        args.prior_manifest_root,
        args.prior_manifest,
        excluded_roots=excluded_roots,
    )
    prior_image_ids, prior_textvqa_groups, prior_records = _load_prior_identities(
        prior_manifests,
        verify_images=not args.skip_prior_image_verification,
    )

    parquet_sha256 = []
    for index, path in enumerate(parquet_paths, 1):
        parquet_sha256.append(_sha256(path))
        print(f"hashed source Parquet shards: {index}/{len(parquet_paths)}", flush=True)

    try:
        from datasets import Dataset
    except ImportError as exc:
        raise SystemExit("Install benchmark dependencies before allocation") from exc
    dataset = Dataset.from_parquet([str(path) for path in parquet_paths])
    group_ids = [str(group_id).strip() for group_id in dataset["image_id"]]
    first_index_by_group: dict[str, int] = {}
    for index, group_id in enumerate(group_ids):
        if not group_id:
            raise ValueError(f"empty TextVQA image_id at row {index}")
        first_index_by_group.setdefault(group_id, index)
    source_images: dict[str, str] = {}
    total_sources = len(first_index_by_group)
    for position, (group_id, source_index) in enumerate(first_index_by_group.items(), 1):
        image = dataset[source_index]["image"].convert("RGB")
        source_images[group_id] = image_digest(image)
        if position % 500 == 0 or position == total_sources:
            print(f"hashed TextVQA train source images: {position}/{total_sources}", flush=True)

    allocation = allocate_source_roles(
        source_images,
        roles=ROLE_SPECS,
        excluded_image_ids=prior_image_ids,
        excluded_source_group_ids=prior_textvqa_groups,
        seed=SEED,
        namespace=NAMESPACE,
    )
    allocation_document = {
        "schema_version": 1,
        "dataset": {
            "dataset_id": "lmms-lab/textvqa",
            "dataset_revision": REVISION,
            "split": "train",
            "row_count": len(dataset),
            "parquet_files": [str(path) for path in parquet_paths],
            "parquet_sha256": parquet_sha256,
        },
        "selection_contract": {
            "target_fields_accessed": False,
            "allowed_fields": ["image_id", "image"],
            "formal_manifest_exported": False,
            "formal_rollouts_collected": False,
        },
        "prior_banks": prior_records,
        "allocation": allocation,
    }
    allocation_path = args.allocation_output.resolve()
    serialized = json.dumps(allocation_document, indent=2, sort_keys=True) + "\n"
    if allocation_path.exists() and allocation_path.read_text(encoding="utf-8") != serialized:
        if not args.resume:
            raise FileExistsError(
                "allocation output already exists with different bytes; pass --resume "
                "only after checking the discrepancy"
            )
    allocation_path.parent.mkdir(parents=True, exist_ok=True)
    allocation_path.write_text(serialized, encoding="utf-8")
    allocation_sha256 = _sha256(allocation_path)
    print(f"allocation SHA-256: {allocation_sha256}", flush=True)

    if args.ranker_output_dir is None:
        return
    ranker_dir = args.ranker_output_dir.resolve()
    calibration_dir = args.calibration_output_dir.resolve()
    _prepare_export_destination(ranker_dir, resume=args.resume)
    _prepare_export_destination(calibration_dir, resume=args.resume)
    ranker_result = _export_role(
        dataset,
        group_ids,
        allocation,
        role="ranker_training",
        state_namespace="textvqa-train-scale-ranker",
        output_dir=ranker_dir,
        allocation_path=allocation_path,
        allocation_sha256=allocation_sha256,
        parquet_paths=parquet_paths,
        parquet_sha256=parquet_sha256,
    )
    calibration_result = _export_role(
        dataset,
        group_ids,
        allocation,
        role="risk_calibration",
        state_namespace="textvqa-train-scale-calibration",
        output_dir=calibration_dir,
        allocation_path=allocation_path,
        allocation_sha256=allocation_sha256,
        parquet_paths=parquet_paths,
        parquet_sha256=parquet_sha256,
    )

    ranker_audit = audit_manifest(ranker_dir)
    calibration_audit = audit_manifest(calibration_dir)
    expected_ranker_images = {
        str(item["image_id"])
        for item in _role_assignments(allocation, "ranker_training")
    }
    expected_calibration_images = {
        str(item["image_id"])
        for item in _role_assignments(allocation, "risk_calibration")
    }
    if ranker_audit["_images"] != expected_ranker_images:
        raise RuntimeError("ranker manifest images do not match frozen allocation")
    if calibration_audit["_images"] != expected_calibration_images:
        raise RuntimeError("calibration manifest images do not match frozen allocation")
    if ranker_audit["_images"] & calibration_audit["_images"]:
        raise RuntimeError("ranker/calibration decoded-RGB overlap")
    if (ranker_audit["_images"] | calibration_audit["_images"]) & prior_image_ids:
        raise RuntimeError("development manifest has prior-bank decoded-RGB overlap")
    audit_document = {
        "passed": True,
        "allocation": str(allocation_path),
        "allocation_sha256": allocation_sha256,
        "prior_manifest_count": len(prior_manifests),
        "prior_unique_image_count": len(prior_image_ids),
        "prior_textvqa_source_group_count": len(prior_textvqa_groups),
        "ranker_training": _clean_audit(ranker_audit),
        "risk_calibration": _clean_audit(calibration_audit),
        "formal_test": {
            "allocated_source_count": len(
                _role_assignments(allocation, "formal_test")
            ),
            "manifest_exported": False,
            "rollouts_collected": False,
        },
        "overlap": {
            "ranker_calibration_images": 0,
            "development_prior_images": 0,
        },
        "manifest_sha256": {
            "ranker_training": ranker_result["manifest_sha256"],
            "risk_calibration": calibration_result["manifest_sha256"],
        },
    }
    _write_json(args.audit_output.resolve(), audit_document)
    print(json.dumps(audit_document, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
