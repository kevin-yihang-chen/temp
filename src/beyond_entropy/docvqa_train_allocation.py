from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence

from .manifest_export import image_digest
from .source_allocation import SourceRoleSpec, allocate_source_roles


DATASET_ID = "lmms-lab/DocVQA"
DATASET_NAME = "DocVQA"
DATASET_REVISION = "539088ef8a8ada01ac8e2e6d4e372586748a265e"
DATASET_SPLIT = "train"
NAMESPACE = "beyond-entropy-docvqa-train-factorized-v2"
SEED = 20260829
PROTOCOL_SHA256 = (
    "f2fc21218085d0b2bce1c92f3a4c30e1dac78b5e813d28a03258bf28fdb06124"
)
ROLE_SPECS = (
    SourceRoleSpec("ranker_training", 0, 3500),
    SourceRoleSpec("risk_calibration", 3500, 2500),
    SourceRoleSpec("formal_test", 6000, 3500),
)
SELECTED_SOURCE_COUNT = 9500
_HEX_DIGEST = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_source_image_identity(
    source_images: dict[str, str],
    *,
    source_group_id: str,
    image_id: str,
) -> bool:
    """Record one row identity and reject a source mapped to multiple RGB images."""

    group_id = str(source_group_id).strip()
    digest = str(image_id).strip()
    if not group_id:
        raise ValueError("source group ID must be non-empty")
    if _HEX_DIGEST.fullmatch(digest) is None:
        raise ValueError("source image ID must be a decoded-RGB SHA-256 digest")
    previous = source_images.get(group_id)
    if previous is None:
        source_images[group_id] = digest
        return True
    if previous != digest:
        raise ValueError(f"DocVQA docId {group_id!r} maps to multiple RGB images")
    return False


def load_docvqa_source_images(
    parquet_paths: Sequence[Path],
) -> tuple[dict[str, str], int]:
    """Read only DocVQA ``docId`` and image columns and validate every row."""

    if not parquet_paths or any(not path.is_file() for path in parquet_paths):
        raise FileNotFoundError("one or more DocVQA train Parquet shards do not exist")
    try:
        from datasets import Dataset  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SystemExit("Install benchmark dependencies before allocation") from exc
    dataset = Dataset.from_parquet([str(path) for path in parquet_paths])
    required_columns = {"docId", "image"}
    if not required_columns.issubset(dataset.column_names):
        raise ValueError("DocVQA train is missing docId or image")
    identity_dataset = dataset.select_columns(["docId", "image"])
    group_ids = [str(group_id).strip() for group_id in identity_dataset["docId"]]

    source_images: dict[str, str] = {}
    total_rows = len(identity_dataset)
    for source_index, group_id in enumerate(group_ids):
        raw_image = identity_dataset[source_index]["image"]
        convert = getattr(raw_image, "convert", None)
        if not callable(convert):
            raise ValueError(f"DocVQA image for docId {group_id!r} is not decodable")
        record_source_image_identity(
            source_images,
            source_group_id=group_id,
            image_id=image_digest(convert("RGB")),
        )
        position = source_index + 1
        if position % 1000 == 0 or position == total_rows:
            print(
                "validated DocVQA train row identities: "
                f"{position}/{total_rows}; unique sources={len(source_images)}",
                flush=True,
            )
    return source_images, total_rows


def _is_beneath(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def discover_prior_manifests(
    roots: Sequence[Path],
    explicit: Sequence[Path],
    *,
    excluded_roots: Sequence[Path] = (),
) -> list[Path]:
    manifests = {path.resolve() for path in explicit}
    for root in roots:
        resolved_root = root.resolve()
        if not resolved_root.is_dir():
            raise FileNotFoundError(f"prior manifest root does not exist: {root}")
        manifests.update(path.resolve() for path in resolved_root.rglob("manifest.jsonl"))
    selected = sorted(
        path
        for path in manifests
        if not any(_is_beneath(path, root.resolve()) for root in excluded_roots)
    )
    if not selected:
        raise ValueError("at least one prior manifest is required for isolation audit")
    for path in selected:
        if not path.is_file():
            raise FileNotFoundError(f"prior manifest does not exist: {path}")
    return selected


def load_prior_identities(
    manifest_paths: Sequence[Path],
    *,
    verify_images: bool = True,
) -> tuple[set[str], set[str], list[dict[str, Any]]]:
    """Load only source and decoded-RGB identities from historical manifests."""

    image_files: set[tuple[str, Path]] = set()
    image_ids: set[str] = set()
    docvqa_source_groups: set[str] = set()
    manifest_records: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        resolved_manifest = manifest_path.resolve()
        row_count = 0
        manifest_image_ids: set[str] = set()
        manifest_docvqa_groups: set[str] = set()
        with resolved_manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise ValueError(
                        "prior manifest row must be a mapping: "
                        f"{resolved_manifest}:{line_number}"
                    )
                image_id = str(payload.get("image_id", "")).strip()
                image_path = str(payload.get("image_path", "")).strip()
                source_id = str(payload.get("source_id", "")).strip()
                if (
                    _HEX_DIGEST.fullmatch(image_id) is None
                    or not image_path
                    or not source_id
                ):
                    raise ValueError(
                        "invalid prior identity fields: "
                        f"{resolved_manifest}:{line_number}"
                    )
                resolved_image = (resolved_manifest.parent / image_path).resolve()
                if not resolved_image.is_file():
                    raise FileNotFoundError(
                        f"prior manifest image does not exist: {resolved_image}"
                    )
                image_files.add((image_id, resolved_image))
                image_ids.add(image_id)
                manifest_image_ids.add(image_id)
                if source_id.startswith("docvqa:"):
                    group_id = source_id.removeprefix("docvqa:").strip()
                    if not group_id:
                        raise ValueError(
                            "empty prior DocVQA source group: "
                            f"{resolved_manifest}:{line_number}"
                        )
                    docvqa_source_groups.add(group_id)
                    manifest_docvqa_groups.add(group_id)
                row_count += 1
        manifest_records.append(
            {
                "manifest": str(resolved_manifest),
                "manifest_sha256": sha256_file(resolved_manifest),
                "row_count": row_count,
                "unique_image_count": len(manifest_image_ids),
                "docvqa_source_group_count": len(manifest_docvqa_groups),
            }
        )
    if verify_images:
        from PIL import Image

        total = len(image_files)
        for index, (declared_id, resolved_image_path) in enumerate(
            sorted(image_files), start=1
        ):
            with Image.open(resolved_image_path) as raw_image:
                actual_id = image_digest(raw_image.convert("RGB"))
            if actual_id != declared_id:
                raise ValueError(
                    "decoded-RGB digest mismatch for prior image "
                    f"{resolved_image_path}"
                )
            if index % 500 == 0 or index == total:
                print(f"verified prior RGB identities: {index}/{total}", flush=True)
    return image_ids, docvqa_source_groups, manifest_records


def _role_assignments(
    allocation: Mapping[str, Any], role: str
) -> list[Mapping[str, Any]]:
    roles = allocation.get("roles")
    if not isinstance(roles, Mapping):
        raise ValueError("allocation roles must be a mapping")
    role_payload = roles.get(role)
    if not isinstance(role_payload, Mapping):
        raise ValueError(f"allocation is missing role {role!r}")
    assignments = role_payload.get("assignments")
    if not isinstance(assignments, list) or any(
        not isinstance(item, Mapping) for item in assignments
    ):
        raise ValueError(f"allocation role {role!r} has invalid assignments")
    return assignments


def build_allocation_document(
    source_images: Mapping[str, str],
    *,
    excluded_image_ids: Collection[str],
    excluded_source_group_ids: Collection[str],
    prior_banks: Sequence[Mapping[str, Any]],
    parquet_files: Sequence[Path],
    parquet_sha256: Sequence[str],
    row_count: int,
    protocol_path: Path,
    code_revision: str,
) -> dict[str, Any]:
    if len(parquet_files) == 0 or len(parquet_files) != len(parquet_sha256):
        raise ValueError("Parquet provenance is incomplete")
    if row_count <= 0:
        raise ValueError("DocVQA train row count must be positive")
    if any(
        _HEX_DIGEST.fullmatch(str(image_id).strip()) is None
        for image_id in source_images.values()
    ):
        raise ValueError("source image IDs must be decoded-RGB SHA-256 digests")
    if any(_HEX_DIGEST.fullmatch(str(value).strip()) is None for value in parquet_sha256):
        raise ValueError("Parquet SHA-256 provenance is invalid")
    revision = str(code_revision).strip()
    if not revision:
        raise ValueError("code revision must be non-empty")
    allocation = allocate_source_roles(
        source_images,
        roles=ROLE_SPECS,
        excluded_image_ids=excluded_image_ids,
        excluded_source_group_ids=excluded_source_group_ids,
        seed=SEED,
        namespace=NAMESPACE,
    )
    selected = [
        item
        for role in ROLE_SPECS
        for item in _role_assignments(allocation, role.name)
    ]
    if len(selected) != SELECTED_SOURCE_COUNT:
        raise RuntimeError("DocVQA allocation did not select exactly 9,500 sources")
    return {
        "schema_version": 1,
        "scientific_status": (
            "outcome-unseen DocVQA-train factorized-v2 identity allocation"
        ),
        "code_revision": revision,
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": PROTOCOL_SHA256,
        "dataset": {
            "dataset_id": DATASET_ID,
            "dataset_name": DATASET_NAME,
            "dataset_revision": DATASET_REVISION,
            "split": DATASET_SPLIT,
            "row_count": row_count,
            "source_group_count": len(source_images),
            "parquet_files": [str(path.resolve()) for path in parquet_files],
            "parquet_sha256": list(parquet_sha256),
        },
        "selection_contract": {
            "selection_target_fields_accessed": False,
            "selection_allowed_fields": ["docId", "image"],
            "ranker_manifest_exported": False,
            "calibration_manifest_exported": False,
            "formal_manifest_exported": False,
            "ranker_outcomes_collected": False,
            "calibration_outcomes_collected": False,
            "formal_outcomes_collected": False,
        },
        "prior_banks": [dict(record) for record in prior_banks],
        "allocation": allocation,
    }


def build_allocation_audit(
    document: Mapping[str, Any],
    *,
    allocation_path: Path,
    allocation_sha256: str,
    excluded_image_ids: Collection[str],
    excluded_source_group_ids: Collection[str],
) -> dict[str, Any]:
    if document.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("allocation protocol SHA-256 mismatch")
    contract = document.get("selection_contract")
    if not isinstance(contract, Mapping) or contract != {
        "selection_target_fields_accessed": False,
        "selection_allowed_fields": ["docId", "image"],
        "ranker_manifest_exported": False,
        "calibration_manifest_exported": False,
        "formal_manifest_exported": False,
        "ranker_outcomes_collected": False,
        "calibration_outcomes_collected": False,
        "formal_outcomes_collected": False,
    }:
        raise ValueError("allocation selection contract changed")
    allocation = document.get("allocation")
    if not isinstance(allocation, Mapping):
        raise ValueError("allocation body must be a mapping")
    if allocation.get("namespace") != NAMESPACE or allocation.get("seed") != SEED:
        raise ValueError("allocation namespace or seed changed")

    role_sources: dict[str, set[str]] = {}
    role_images: dict[str, set[str]] = {}
    role_reports: dict[str, dict[str, Any]] = {}
    for spec in ROLE_SPECS:
        assignments = _role_assignments(allocation, spec.name)
        if len(assignments) != spec.count:
            raise ValueError(f"allocation count changed for role {spec.name!r}")
        sources = {str(item.get("source_group_id", "")) for item in assignments}
        images = {str(item.get("image_id", "")) for item in assignments}
        if "" in sources or "" in images:
            raise ValueError(f"allocation role {spec.name!r} has empty identities")
        if len(sources) != spec.count or len(images) != spec.count:
            raise ValueError(f"allocation role {spec.name!r} is not internally unique")
        role_sources[spec.name] = sources
        role_images[spec.name] = images
        role_payload = allocation["roles"][spec.name]
        role_reports[spec.name] = {
            "offset": spec.offset,
            "allocated_source_count": len(sources),
            "allocated_unique_image_count": len(images),
            "base_selected_count": int(role_payload["base_selected_count"]),
            "reserve_backfill_count": int(role_payload["reserve_backfill_count"]),
            "manifest_exported": False,
            "outcomes_collected": False,
        }

    excluded_images = {str(value).strip() for value in excluded_image_ids}
    excluded_sources = {
        str(value).strip() for value in excluded_source_group_ids
    }
    selected_images = set().union(*role_images.values())
    selected_sources = set().union(*role_sources.values())
    if len(selected_images) != SELECTED_SOURCE_COUNT:
        raise ValueError("allocation has cross-role RGB overlap")
    if len(selected_sources) != SELECTED_SOURCE_COUNT:
        raise ValueError("allocation has cross-role source overlap")
    if selected_images & excluded_images:
        raise ValueError("allocation overlaps prior RGB identities")
    if selected_sources & excluded_sources:
        raise ValueError("allocation overlaps prior DocVQA source identities")

    role_pairs = (
        ("ranker_training", "risk_calibration"),
        ("ranker_training", "formal_test"),
        ("risk_calibration", "formal_test"),
    )
    overlaps: dict[str, int] = {}
    for left, right in role_pairs:
        overlaps[f"{left}_{right}_sources"] = len(
            role_sources[left] & role_sources[right]
        )
        overlaps[f"{left}_{right}_images"] = len(
            role_images[left] & role_images[right]
        )
    overlaps["selected_prior_sources"] = len(selected_sources & excluded_sources)
    overlaps["selected_prior_images"] = len(selected_images & excluded_images)
    if any(overlaps.values()):
        raise RuntimeError("DocVQA identity allocation audit found overlap")

    prior_banks = document.get("prior_banks")
    if not isinstance(prior_banks, list) or not prior_banks:
        raise ValueError("allocation must bind at least one prior manifest")
    return {
        "passed": True,
        "scientific_status": (
            "identity-only DocVQA allocation passed; every outcome role remains sealed"
        ),
        "allocation": str(allocation_path.resolve()),
        "allocation_sha256": allocation_sha256,
        "protocol_sha256": PROTOCOL_SHA256,
        "prior_manifest_count": len(prior_banks),
        "prior_unique_image_count": len(excluded_images),
        "prior_docvqa_source_group_count": len(excluded_sources),
        "source_group_count": int(allocation["source_group_count"]),
        "unique_image_count": int(allocation["unique_image_count"]),
        "roles": role_reports,
        "overlap": overlaps,
        "ranker_outcomes_collected": False,
        "calibration_outcomes_collected": False,
        "formal_outcomes_collected": False,
    }


def verify_recomputed_allocation_bundle(
    document: Mapping[str, Any],
    audit: Mapping[str, Any],
    *,
    source_images: Mapping[str, str],
    excluded_image_ids: Collection[str],
    excluded_source_group_ids: Collection[str],
    prior_banks: Sequence[Mapping[str, Any]],
    parquet_files: Sequence[Path],
    parquet_sha256: Sequence[str],
    row_count: int,
    protocol_path: Path,
    allocation_path: Path,
    allocation_sha256: str,
) -> dict[str, Any]:
    """Recompute an allocation and audit byte-for-byte from frozen inputs."""

    code_revision = str(document.get("code_revision", "")).strip()
    rebuilt_document = build_allocation_document(
        source_images,
        excluded_image_ids=excluded_image_ids,
        excluded_source_group_ids=excluded_source_group_ids,
        prior_banks=prior_banks,
        parquet_files=parquet_files,
        parquet_sha256=parquet_sha256,
        row_count=row_count,
        protocol_path=protocol_path,
        code_revision=code_revision,
    )
    if dict(document) != rebuilt_document:
        raise ValueError("DocVQA allocation differs from deterministic recomputation")
    rebuilt_audit = build_allocation_audit(
        rebuilt_document,
        allocation_path=allocation_path,
        allocation_sha256=allocation_sha256,
        excluded_image_ids=excluded_image_ids,
        excluded_source_group_ids=excluded_source_group_ids,
    )
    if dict(audit) != rebuilt_audit:
        raise ValueError("DocVQA allocation audit differs from recomputation")
    return {
        "passed": True,
        "allocation_sha256": allocation_sha256,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_group_count": rebuilt_audit["source_group_count"],
        "selected_source_count": SELECTED_SOURCE_COUNT,
        "prior_manifest_count": rebuilt_audit["prior_manifest_count"],
        "ranker_outcomes_collected": False,
        "calibration_outcomes_collected": False,
        "formal_outcomes_collected": False,
    }
