from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from beyond_entropy.manifest_audit import audit_manifest
from beyond_entropy.manifest_export import export_benchmark_manifest


REVISION = "9c0699cd19768ac5ab97568f6b3cbac4c0062884"
SEED = 20260828
ALLOCATION_NAMESPACE = "beyond-entropy-textvqa-train-scale-v1"


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


def _verify_freeze_components(freeze: Mapping[str, Any]) -> None:
    if freeze.get("formal_gate_status") != "ready_for_formal_manifest":
        raise ValueError("scaled policy freeze has not opened the formal gate")
    formal = freeze.get("formal_test")
    if not isinstance(formal, Mapping):
        raise ValueError("scaled policy freeze is missing formal-test metadata")
    if formal.get("allocated_sources") != 5000:
        raise ValueError("scaled policy freeze has the wrong formal source count")
    if bool(formal.get("manifest_materialized")) or bool(formal.get("rollouts_collected")):
        raise ValueError("scaled policy freeze does not describe a sealed formal role")
    for section_name in ("artifacts", "implementation"):
        section = freeze.get(section_name)
        if not isinstance(section, Mapping) or not section:
            raise ValueError(f"scaled policy freeze is missing {section_name}")
        for name, item in section.items():
            if not isinstance(item, Mapping):
                raise ValueError(f"invalid frozen component {section_name}.{name}")
            path = Path(str(item.get("path", ""))).resolve()
            expected = str(item.get("sha256", ""))
            if not path.is_file() or _sha256(path) != expected:
                raise ValueError(f"frozen component changed: {section_name}.{name}")


def _prior_identities(prior_banks: Any) -> tuple[set[str], set[str]]:
    if not isinstance(prior_banks, list) or not prior_banks:
        raise ValueError("allocation must record prior manifest banks")
    images: set[str] = set()
    textvqa_groups: set[str] = set()
    for item in prior_banks:
        if not isinstance(item, Mapping):
            raise ValueError("invalid prior-bank allocation record")
        path = Path(str(item.get("manifest", ""))).resolve()
        if not path.is_file() or _sha256(path) != item.get("manifest_sha256"):
            raise ValueError(f"prior manifest changed after allocation: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                image_id = str(row.get("image_id", "")).strip()
                source_id = str(row.get("source_id", "")).strip()
                if not image_id or not source_id:
                    raise ValueError(f"prior manifest contains an invalid identity: {path}")
                images.add(image_id)
                if source_id.startswith("textvqa:"):
                    textvqa_groups.add(source_id.removeprefix("textvqa:"))
    return images, textvqa_groups


def _clean_audit(report: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in report.items() if not name.startswith("_")}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize the sealed 5,000-source TextVQA formal manifest"
    )
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--expected-allocation-sha256", required=True)
    parser.add_argument("--policy-freeze", type=Path, required=True)
    parser.add_argument("--expected-policy-freeze-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parents[1]
    if _sha256(args.allocation) != args.expected_allocation_sha256:
        raise ValueError("allocation SHA-256 mismatch")
    if _sha256(args.policy_freeze) != args.expected_policy_freeze_sha256:
        raise ValueError("policy-freeze SHA-256 mismatch")
    allocation = _load_mapping(args.allocation, "allocation")
    freeze = _load_mapping(args.policy_freeze, "policy freeze")
    _verify_freeze_components(freeze)
    code_revision = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if code_revision != freeze.get("code_revision"):
        raise ValueError("repository revision differs from the policy freeze")
    tracked_status = subprocess.run(
        ["git", "-C", str(repo_dir), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if tracked_status.strip():
        raise ValueError("tracked worktree must be clean before formal export")
    if args.output_dir.exists() or args.audit_output.exists():
        raise FileExistsError("formal export or audit destination already exists")

    contract = allocation.get("selection_contract")
    dataset_metadata = allocation.get("dataset")
    allocation_body = allocation.get("allocation")
    if not isinstance(contract, Mapping) or not isinstance(dataset_metadata, Mapping):
        raise ValueError("allocation is missing its selection or dataset contract")
    if not isinstance(allocation_body, Mapping):
        raise ValueError("allocation body must be a mapping")
    if contract != {
        "allowed_fields": ["image_id", "image"],
        "formal_manifest_exported": False,
        "formal_rollouts_collected": False,
        "target_fields_accessed": False,
    }:
        raise ValueError("allocation no longer describes an untouched formal role")
    if (
        dataset_metadata.get("dataset_id") != "lmms-lab/textvqa"
        or dataset_metadata.get("dataset_revision") != REVISION
        or dataset_metadata.get("split") != "train"
        or allocation_body.get("namespace") != ALLOCATION_NAMESPACE
        or allocation_body.get("seed") != SEED
    ):
        raise ValueError("allocation dataset or namespace contract mismatch")

    parquet_paths = [Path(path).resolve() for path in dataset_metadata["parquet_files"]]
    parquet_hashes = [str(value) for value in dataset_metadata["parquet_sha256"]]
    if len(parquet_paths) != len(parquet_hashes) or not parquet_paths:
        raise ValueError("allocation Parquet provenance is incomplete")
    for path, expected in zip(parquet_paths, parquet_hashes):
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"source Parquet changed after allocation: {path}")

    roles = allocation_body.get("roles")
    if not isinstance(roles, Mapping):
        raise ValueError("allocation roles must be a mapping")
    formal_role = roles.get("formal_test")
    if not isinstance(formal_role, Mapping):
        raise ValueError("allocation has no formal role")
    assignments = formal_role.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != 5000:
        raise ValueError("formal allocation must contain exactly 5,000 sources")
    selected_groups = {str(item["source_group_id"]) for item in assignments}
    expected_images = {str(item["image_id"]) for item in assignments}
    if len(selected_groups) != 5000 or len(expected_images) != 5000:
        raise ValueError("formal allocation source or RGB identities are not unique")
    development_images: set[str] = set()
    development_groups: set[str] = set()
    for role_name in ("ranker_training", "risk_calibration"):
        role = roles.get(role_name)
        if not isinstance(role, Mapping) or not isinstance(role.get("assignments"), list):
            raise ValueError(f"allocation role is incomplete: {role_name}")
        development_images.update(str(item["image_id"]) for item in role["assignments"])
        development_groups.update(
            str(item["source_group_id"]) for item in role["assignments"]
        )
    prior_images, prior_groups = _prior_identities(allocation.get("prior_banks"))
    if expected_images & (development_images | prior_images):
        raise ValueError("formal allocation has development/prior RGB overlap")
    if selected_groups & (development_groups | prior_groups):
        raise ValueError("formal allocation has development/prior source overlap")

    try:
        from datasets import Dataset
    except ImportError as exc:
        raise SystemExit("Install benchmark dependencies before formal export") from exc
    dataset = Dataset.from_parquet([str(path) for path in parquet_paths])
    if len(dataset) != int(dataset_metadata.get("row_count", -1)):
        raise ValueError("source Parquet row count differs from the allocation")
    group_ids = [str(group_id).strip() for group_id in dataset["image_id"]]
    source_indices = [
        index for index, group_id in enumerate(group_ids) if group_id in selected_groups
    ]
    if {group_ids[index] for index in source_indices} != selected_groups:
        raise ValueError("formal source groups do not exactly match the source dataset")
    selected_dataset = dataset.select(source_indices)
    result = export_benchmark_manifest(
        selected_dataset,
        source_indices=source_indices,
        task="textvqa",
        dataset_id="lmms-lab/textvqa",
        dataset_revision=REVISION,
        dataset_split="train",
        output_dir=args.output_dir,
        seed=SEED,
        state_namespace="textvqa-train-scale-formal",
        selection="sealed SHA-256 source-role allocation opened after successful calibration",
        selection_metadata={
            "allocation": str(args.allocation.resolve()),
            "allocation_sha256": args.expected_allocation_sha256,
            "policy_freeze": str(args.policy_freeze.resolve()),
            "policy_freeze_sha256": args.expected_policy_freeze_sha256,
            "namespace": ALLOCATION_NAMESPACE,
            "role": "formal_test",
            "selected_source_group_count": 5000,
            "selection_uses_targets": False,
        },
    )
    result["source_parquet_files"] = [str(path) for path in parquet_paths]
    result["source_parquet_sha256"] = parquet_hashes
    provenance_path = args.output_dir / "manifest.provenance.json"
    provenance_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit = audit_manifest(args.output_dir)
    if audit["_images"] != expected_images:
        raise RuntimeError("formal export images differ from the frozen allocation")
    expected_sources = {f"textvqa:{group_id}" for group_id in selected_groups}
    if audit["_sources"] != expected_sources:
        raise RuntimeError("formal export sources differ from the frozen allocation")
    audit_document = {
        "passed": True,
        "scientific_status": "outcome-unseen formal manifest; policy frozen before export",
        "allocation": str(args.allocation.resolve()),
        "allocation_sha256": args.expected_allocation_sha256,
        "policy_freeze": str(args.policy_freeze.resolve()),
        "policy_freeze_sha256": args.expected_policy_freeze_sha256,
        "formal": _clean_audit(audit),
        "overlap": {
            "formal_development_images": 0,
            "formal_prior_images": 0,
            "formal_development_sources": 0,
            "formal_prior_textvqa_sources": 0,
        },
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    with args.audit_output.open("x", encoding="utf-8") as handle:
        json.dump(audit_document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(audit_document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
