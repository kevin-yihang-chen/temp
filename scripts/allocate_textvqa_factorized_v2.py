from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from beyond_entropy.manifest_audit import audit_manifest
from beyond_entropy.manifest_export import export_benchmark_manifest, image_digest


REVISION = "9c0699cd19768ac5ab97568f6b3cbac4c0062884"
NAMESPACE = "beyond-entropy-textvqa-train-scale-v1"
SEED = 20260828
PARENT_ALLOCATION_SHA256 = (
    "da6d41584bf4f3bfb91426fa9fa3bcb61a659846147c279eaab2aedb776e1657"
)
CANDIDATE_SHA256 = (
    "9a6c9d032ebdbc271b7d3c829fbb3d6ff167cac01b54ce75adc8da86e3063342"
)
CANDIDATE_AUDIT_SHA256 = (
    "63d8040e25701a6dc4e2f2841d4e10c2b688ccb1a0f23e65f15ea6450eb5d294"
)
PROTOCOL_SHA256 = (
    "babf01d4090263d1cfcb28c42f86f7b13ae9de4bb6bab0ca10d6e4707f02e2ca"
)
CALIBRATION_OFFSET = 13000
CALIBRATION_SOURCES = 3000
FORMAL_OFFSET = 16000
FORMAL_SOURCES = 5953
TOTAL_SOURCES = 21953


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mapping(path: Path, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _source_rank(group_id: str) -> tuple[str, str]:
    payload = f"{NAMESPACE}\0{SEED}\0{group_id}".encode()
    return hashlib.sha256(payload).hexdigest(), group_id


def _write_frozen_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    resume: bool,
) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if not resume:
            raise FileExistsError(f"frozen output already exists: {path}")
        if path.read_text(encoding="utf-8") != serialized:
            raise ValueError(f"existing frozen output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def _clean_audit(report: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if not key.startswith("_")}


def _parent_role_identities(
    parent_body: Mapping[str, Any],
) -> tuple[set[str], set[str]]:
    roles = parent_body.get("roles")
    if not isinstance(roles, Mapping) or set(roles) != {
        "ranker_training",
        "risk_calibration",
        "formal_test",
    }:
        raise ValueError("parent allocation roles differ from the frozen contract")
    groups: set[str] = set()
    images: set[str] = set()
    expected_counts = {
        "ranker_training": 5000,
        "risk_calibration": 3000,
        "formal_test": 5000,
    }
    for name, count in expected_counts.items():
        role = roles[name]
        if not isinstance(role, Mapping):
            raise ValueError(f"parent role is invalid: {name}")
        assignments = role.get("assignments")
        if not isinstance(assignments, list) or len(assignments) != count:
            raise ValueError(f"parent role count is invalid: {name}")
        groups.update(str(item["source_group_id"]) for item in assignments)
        images.update(str(item["image_id"]) for item in assignments)
    if len(groups) != 13000 or len(images) != 13000:
        raise ValueError("parent allocation identities are not disjoint")
    return groups, images


def _validate_parent(parent: Mapping[str, Any]) -> Mapping[str, Any]:
    dataset = parent.get("dataset")
    body = parent.get("allocation")
    contract = parent.get("selection_contract")
    if not isinstance(dataset, Mapping) or not isinstance(body, Mapping):
        raise ValueError("parent allocation is missing dataset or body")
    if not isinstance(contract, Mapping):
        raise ValueError("parent allocation is missing the selection contract")
    expected_dataset = {
        "dataset_id": "lmms-lab/textvqa",
        "dataset_revision": REVISION,
        "split": "train",
        "row_count": 34602,
    }
    for name, value in expected_dataset.items():
        if dataset.get(name) != value:
            raise ValueError(f"parent dataset changed for {name}")
    if (
        body.get("namespace") != NAMESPACE
        or body.get("seed") != SEED
        or body.get("source_group_count") != TOTAL_SOURCES
        or body.get("unique_image_count") != TOTAL_SOURCES
        or body.get("reserve_start") != CALIBRATION_OFFSET
        or body.get("reserve_consumed_end_exclusive") != CALIBRATION_OFFSET
        or body.get("prior_collision_source_group_count") != 0
        or body.get("prior_source_group_collision_count") != 0
        or body.get("duplicate_rgb_source_group_count") != 0
    ):
        raise ValueError("parent reserve or collision contract changed")
    if (
        contract.get("target_fields_accessed") is not False
        or contract.get("formal_manifest_exported") is not False
        or contract.get("formal_rollouts_collected") is not False
    ):
        raise ValueError("parent allocation no longer has a sealed formal role")
    return dataset


def _validate_candidate(
    candidate: Mapping[str, Any],
    candidate_audit: Mapping[str, Any],
) -> None:
    expected = {
        "model_type": "multidomain_factorized_action_value",
        "training_protocol": "source_grouped_oof_domain_source_balanced_v2",
        "sample_weighting": "equal_domain_then_equal_source_then_equal_row",
        "feature_mode": "hybrid-context-semantic",
        "selected_alpha": 1.0,
        "seed": SEED,
        "n_folds": 5,
        "lambda_cost": 0.05,
    }
    for name, value in expected.items():
        if candidate.get(name) != value:
            raise ValueError(f"candidate changed for {name}")
    if candidate.get("threshold") is not None:
        raise ValueError("candidate is already calibrated")
    thresholds = candidate.get("threshold_grid")
    if (
        not isinstance(thresholds, list)
        or len(thresholds) != 11
        or any(
            float(left) <= float(right)
            for left, right in zip(thresholds, thresholds[1:])
        )
    ):
        raise ValueError("candidate threshold sequence is not frozen")
    freeze = candidate.get("candidate_freeze")
    if not isinstance(freeze, Mapping):
        raise ValueError("candidate is missing freeze provenance")
    if (
        freeze.get("protocol_sha256") != PROTOCOL_SHA256
        or freeze.get("calibration_outcomes_used") is not False
        or freeze.get("formal_outcomes_used") is not False
    ):
        raise ValueError("candidate freeze used held-out outcomes")
    if (
        candidate_audit.get("passed") is not True
        or candidate_audit.get("candidate_sha256") != CANDIDATE_SHA256
        or candidate_audit.get("calibration_outcomes_used") is not False
        or candidate_audit.get("formal_outcomes_used") is not False
    ):
        raise ValueError("candidate audit is invalid")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Allocate fresh factorized TextVQA calibration and formal roles"
    )
    parser.add_argument("--parent-allocation", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-audit", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--allocation-output", type=Path, required=True)
    parser.add_argument("--calibration-output-dir", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    frozen_inputs = (
        (args.parent_allocation, PARENT_ALLOCATION_SHA256, "parent allocation"),
        (args.candidate, CANDIDATE_SHA256, "candidate"),
        (args.candidate_audit, CANDIDATE_AUDIT_SHA256, "candidate audit"),
        (args.protocol, PROTOCOL_SHA256, "protocol"),
    )
    for path, expected, name in frozen_inputs:
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"{name} SHA-256 mismatch")
    parent = _load_mapping(args.parent_allocation, "parent allocation")
    dataset_metadata = _validate_parent(parent)
    parent_groups, parent_images = _parent_role_identities(parent["allocation"])
    candidate = _load_mapping(args.candidate, "candidate")
    candidate_audit = _load_mapping(args.candidate_audit, "candidate audit")
    _validate_candidate(candidate, candidate_audit)

    parquet_paths = [
        Path(path).resolve() for path in dataset_metadata.get("parquet_files", [])
    ]
    parquet_hashes = [
        str(value) for value in dataset_metadata.get("parquet_sha256", [])
    ]
    if not parquet_paths or len(parquet_paths) != len(parquet_hashes):
        raise ValueError("parent Parquet provenance is incomplete")
    for index, (path, expected) in enumerate(
        zip(parquet_paths, parquet_hashes), start=1
    ):
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"source Parquet changed: {path}")
        print(f"verified source Parquet shards: {index}/{len(parquet_paths)}", flush=True)

    try:
        from datasets import Dataset
    except ImportError as exc:
        raise SystemExit("Install benchmark dependencies before allocation") from exc
    dataset = Dataset.from_parquet([str(path) for path in parquet_paths])
    if len(dataset) != int(dataset_metadata["row_count"]):
        raise ValueError("source dataset row count changed")
    group_ids = [str(value).strip() for value in dataset["image_id"]]
    first_index_by_group: dict[str, int] = {}
    for index, group_id in enumerate(group_ids):
        if not group_id:
            raise ValueError(f"empty source group at row {index}")
        first_index_by_group.setdefault(group_id, index)
    ordered_groups = sorted(first_index_by_group, key=_source_rank)
    if len(ordered_groups) != TOTAL_SOURCES:
        raise ValueError("TextVQA source count changed")
    calibration_groups = ordered_groups[
        CALIBRATION_OFFSET : CALIBRATION_OFFSET + CALIBRATION_SOURCES
    ]
    formal_groups = ordered_groups[FORMAL_OFFSET:]
    if len(calibration_groups) != CALIBRATION_SOURCES or len(formal_groups) != FORMAL_SOURCES:
        raise RuntimeError("reserve role counts are inconsistent")
    selected_groups = calibration_groups + formal_groups
    if set(selected_groups) & parent_groups:
        raise RuntimeError("new reserve roles overlap parent source groups")

    image_by_group: dict[str, str] = {}
    for position, group_id in enumerate(selected_groups, start=1):
        raw_image = dataset[first_index_by_group[group_id]]["image"]
        image_by_group[group_id] = image_digest(raw_image.convert("RGB"))
        if position % 500 == 0 or position == len(selected_groups):
            print(
                f"hashed fresh reserve RGB identities: {position}/{len(selected_groups)}",
                flush=True,
            )
    new_images = set(image_by_group.values())
    if len(new_images) != len(selected_groups):
        raise RuntimeError("fresh reserve roles contain duplicate decoded RGB images")
    if new_images & parent_images:
        raise RuntimeError("fresh reserve roles overlap parent decoded RGB images")

    def role_payload(groups: list[str], offset: int) -> dict[str, Any]:
        return {
            "offset": offset,
            "count": len(groups),
            "base_interval_end_exclusive": offset + len(groups),
            "base_selected_count": len(groups),
            "reserve_backfill_count": 0,
            "assignments": [
                {
                    "source_group_id": group_id,
                    "source_rank": offset + index,
                    "image_id": image_by_group[group_id],
                    "origin": "base_interval",
                }
                for index, group_id in enumerate(groups)
            ],
        }

    allocation_document = {
        "schema_version": 1,
        "scientific_status": (
            "outcome-unseen reserve allocation for fixed-sequence factorized branch"
        ),
        "parent_allocation": str(args.parent_allocation.resolve()),
        "parent_allocation_sha256": PARENT_ALLOCATION_SHA256,
        "candidate": str(args.candidate.resolve()),
        "candidate_sha256": CANDIDATE_SHA256,
        "candidate_audit": str(args.candidate_audit.resolve()),
        "candidate_audit_sha256": CANDIDATE_AUDIT_SHA256,
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": PROTOCOL_SHA256,
        "dataset": dict(dataset_metadata),
        "selection_contract": {
            "selection_target_fields_accessed": False,
            "selection_allowed_fields": ["image_id", "image"],
            "calibration_manifest_exported": True,
            "calibration_targets_materialized_after_selection": True,
            "calibration_outcomes_collected": False,
            "formal_manifest_exported": False,
            "formal_rollouts_collected": False,
        },
        "allocation": {
            "schema_version": 1,
            "namespace": NAMESPACE,
            "seed": SEED,
            "source_group_count": TOTAL_SOURCES,
            "unique_image_count": TOTAL_SOURCES,
            "parent_role_end_exclusive": CALIBRATION_OFFSET,
            "reserve_start": TOTAL_SOURCES,
            "reserve_consumed_end_exclusive": TOTAL_SOURCES,
            "roles": {
                "risk_calibration": role_payload(
                    calibration_groups, CALIBRATION_OFFSET
                ),
                "formal_test": role_payload(formal_groups, FORMAL_OFFSET),
            },
        },
    }
    _write_frozen_json(
        args.allocation_output.resolve(), allocation_document, resume=args.resume
    )
    allocation_sha256 = _sha256(args.allocation_output)
    print(f"fresh allocation SHA-256: {allocation_sha256}", flush=True)

    calibration_group_set = set(calibration_groups)
    calibration_indices = [
        index for index, group_id in enumerate(group_ids) if group_id in calibration_group_set
    ]
    calibration_dataset = dataset.select(calibration_indices)
    export = export_benchmark_manifest(
        calibration_dataset,
        source_indices=calibration_indices,
        task="textvqa",
        dataset_id="lmms-lab/textvqa",
        dataset_revision=REVISION,
        dataset_split="train",
        output_dir=args.calibration_output_dir.resolve(),
        seed=SEED,
        state_namespace="textvqa-train-factorized-v2-calibration",
        selection="frozen untouched-reserve SHA-256 source-rank interval",
        selection_metadata={
            "allocation": str(args.allocation_output.resolve()),
            "allocation_sha256": allocation_sha256,
            "candidate_sha256": CANDIDATE_SHA256,
            "protocol_sha256": PROTOCOL_SHA256,
            "namespace": NAMESPACE,
            "role": "risk_calibration",
            "source_rank_start": CALIBRATION_OFFSET,
            "source_rank_end_exclusive": FORMAL_OFFSET,
            "selected_source_group_count": CALIBRATION_SOURCES,
            "selection_uses_targets": False,
        },
    )
    export["source_parquet_files"] = [str(path) for path in parquet_paths]
    export["source_parquet_sha256"] = parquet_hashes
    provenance_path = args.calibration_output_dir / "manifest.provenance.json"
    provenance_path.write_text(
        json.dumps(export, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    calibration_audit = audit_manifest(args.calibration_output_dir)
    expected_calibration_images = {
        image_by_group[group_id] for group_id in calibration_groups
    }
    expected_calibration_sources = {
        f"textvqa:{group_id}" for group_id in calibration_groups
    }
    if calibration_audit["_images"] != expected_calibration_images:
        raise RuntimeError("calibration manifest images differ from allocation")
    if calibration_audit["_sources"] != expected_calibration_sources:
        raise RuntimeError("calibration manifest sources differ from allocation")

    audit_document = {
        "passed": True,
        "scientific_status": (
            "fresh calibration manifest exported; formal role remains identity-only"
        ),
        "allocation": str(args.allocation_output.resolve()),
        "allocation_sha256": allocation_sha256,
        "candidate_sha256": CANDIDATE_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "calibration": _clean_audit(calibration_audit),
        "formal": {
            "allocated_source_count": FORMAL_SOURCES,
            "allocated_unique_image_count": len(
                {image_by_group[group_id] for group_id in formal_groups}
            ),
            "manifest_exported": False,
            "rollouts_collected": False,
        },
        "overlap": {
            "new_calibration_formal_sources": 0,
            "new_calibration_formal_images": 0,
            "new_parent_sources": 0,
            "new_parent_images": 0,
        },
        "calibration_outcomes_collected": False,
        "formal_outcomes_collected": False,
    }
    _write_frozen_json(
        args.audit_output.resolve(), audit_document, resume=args.resume
    )
    print(json.dumps(audit_document, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
