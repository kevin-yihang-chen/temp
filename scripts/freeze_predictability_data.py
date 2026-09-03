from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from beyond_entropy.manifest_export import (
    export_benchmark_manifest,
    image_digest,
    stratified_unique_group_sample_indices,
)
from beyond_entropy.predictability_audit import (
    SplitIdentity,
    assign_disjoint_split_roles,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_path(root: Path, spec: Mapping[str, Any]) -> Path:
    raw = Path(str(spec["path"]))
    path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != spec["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual}")
    return path


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid manifest row {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"manifest row is not a mapping: {path}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()


def scan_blocked_manifests(root: Path, output_root: Path) -> dict[str, Any]:
    paths = sorted(root.glob("data/**/manifest*.jsonl"))
    blocked_images: set[str] = set()
    blocked_sources: set[str] = set()
    inventory = []
    for path in paths:
        if (
            output_root in path.parents
            or output_root.with_name(output_root.name + ".staging") in path.parents
        ):
            continue
        rows = read_manifest(path)
        blocked_images.update(
            str(row["image_id"]) for row in rows if row.get("image_id")
        )
        blocked_sources.update(
            str(row["source_id"]) for row in rows if row.get("source_id")
        )
        inventory.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": sha256_file(path),
                "rows": len(rows),
            }
        )
    return {
        "manifests": inventory,
        "blocked_image_ids": blocked_images,
        "blocked_source_ids": blocked_sources,
    }


def adjusted_subset_rows(
    rows: Sequence[Mapping[str, Any]], source_manifest: Path, destination: Path
) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        value = dict(row)
        raw_image = Path(str(value["image_path"]))
        absolute = (
            raw_image if raw_image.is_absolute() else source_manifest.parent / raw_image
        )
        value["image_path"] = os.path.relpath(absolute.resolve(), destination)
        result.append(value)
    return result


def hash_rank(values: Iterable[str], *, seed: int, namespace: str) -> list[str]:
    return sorted(
        set(values),
        key=lambda value: (
            hashlib.sha256(f"{namespace}\0{seed}\0{value}".encode()).hexdigest(),
            value,
        ),
    )


def load_parquet_dataset(paths: Sequence[Path]) -> Any:
    from datasets import Dataset, concatenate_datasets  # type: ignore[import-untyped]

    return concatenate_datasets([Dataset.from_parquet(str(path)) for path in paths])


def freeze_chartqa(
    *,
    root: Path,
    stage: Path,
    config: Mapping[str, Any],
    blocked_images: set[str],
    seed: int,
) -> dict[str, Any]:
    opened = checked_path(root, config["opened_development_manifest"])
    opened_rows = read_manifest(opened)
    if len(opened_rows) != int(config["train_states"]) + int(
        config["validation_states"]
    ):
        raise ValueError("ChartQA opened development count changed")
    ordered = hash_rank(
        (str(row["state_id"]) for row in opened_rows),
        seed=seed,
        namespace="predictability-chartqa-development",
    )
    train_ids = set(ordered[: int(config["train_states"])])
    development_roles = {
        "train": [row for row in opened_rows if str(row["state_id"]) in train_ids],
        "validation": [
            row for row in opened_rows if str(row["state_id"]) not in train_ids
        ],
    }
    reports: dict[str, Any] = {}
    for role, rows in development_roles.items():
        destination = stage / "chartqa" / role
        manifest = destination / "manifest.jsonl"
        adjusted = adjusted_subset_rows(rows, opened, destination)
        reports[role] = {
            "manifest": str(manifest.relative_to(stage)),
            "manifest_sha256": write_jsonl(adjusted, manifest),
            "states": len(adjusted),
            "sources": len({row["source_id"] for row in adjusted}),
            "images": len({row["image_id"] for row in adjusted}),
            "historically_opened": True,
        }

    parquet_paths = [checked_path(root, item) for item in config["parquets"]]
    dataset = load_parquet_dataset(parquet_paths)
    candidate_indices = []
    candidate_labels = []
    candidate_images = []
    for index, row in enumerate(dataset):
        image_id = image_digest(row["image"].convert("RGB"))
        if image_id in blocked_images:
            continue
        group = int(row["human_or_machine"])
        if group not in (0, 1):
            raise ValueError("unexpected ChartQA human_or_machine value")
        candidate_indices.append(index)
        candidate_labels.append("human_train" if group == 0 else "augmented_train")
        candidate_images.append(image_id)
    selected_positions = stratified_unique_group_sample_indices(
        candidate_labels,
        candidate_images,
        count=int(config["test_states"]),
        seed=seed,
    )
    source_indices = [candidate_indices[position] for position in selected_positions]
    export_rows = []
    for index in source_indices:
        row = dataset[index]
        labels = list(row["label"])
        if len(labels) != 1:
            raise ValueError("ChartQA row must have one label")
        export_rows.append(
            {
                "image": row["image"],
                "type": (
                    "human_train"
                    if int(row["human_or_machine"]) == 0
                    else "augmented_train"
                ),
                "question": str(row["query"]),
                "answer": str(labels[0]),
            }
        )
    exported = export_benchmark_manifest(
        export_rows,
        source_indices=source_indices,
        task="chartqa",
        dataset_id=str(config["dataset_id"]),
        dataset_revision=str(config["revision"]),
        output_dir=stage / "chartqa" / "test",
        seed=seed,
        state_namespace="chartqa-predictability-test",
        dataset_split="train",
        selection=str(config["test_selection"]),
    )
    reports["test"] = {
        "manifest": str(Path(exported["manifest"]).relative_to(stage)),
        "manifest_sha256": exported["manifest_sha256"],
        "states": exported["count"],
        "sources": exported["unique_sources"],
        "images": exported["unique_images"],
        "historically_opened": False,
        "source_indices": source_indices,
    }
    return reports


def freeze_docvqa(
    *,
    root: Path,
    stage: Path,
    config: Mapping[str, Any],
    blocked_images: set[str],
    blocked_sources: set[str],
    seed: int,
) -> dict[str, Any]:
    opened = checked_path(root, config["opened_development_manifest"])
    opened_rows = read_manifest(opened)
    source_ids = {str(row["source_id"]) for row in opened_rows}
    if len(source_ids) != int(config["train_sources"]) + int(
        config["validation_sources"]
    ):
        raise ValueError("DocVQA opened development source count changed")
    ordered_sources = hash_rank(
        source_ids, seed=seed, namespace="predictability-docvqa-development"
    )
    train_sources = set(ordered_sources[: int(config["train_sources"])])
    reports: dict[str, Any] = {}
    for role, selected in (
        ("train", train_sources),
        ("validation", source_ids - train_sources),
    ):
        rows = [row for row in opened_rows if str(row["source_id"]) in selected]
        destination = stage / "docvqa" / role
        manifest = destination / "manifest.jsonl"
        adjusted = adjusted_subset_rows(rows, opened, destination)
        reports[role] = {
            "manifest": str(manifest.relative_to(stage)),
            "manifest_sha256": write_jsonl(adjusted, manifest),
            "states": len(adjusted),
            "sources": len(selected),
            "images": len({row["image_id"] for row in adjusted}),
            "historically_opened": True,
        }

    parquet_paths = [checked_path(root, item) for item in config["validation_parquets"]]
    dataset = load_parquet_dataset(parquet_paths)
    group_rows: dict[str, list[int]] = defaultdict(list)
    group_images: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(dataset):
        source_id = f"docvqa:{str(row['docId']).strip()}"
        if source_id in blocked_sources:
            continue
        image_id = image_digest(row["image"].convert("RGB"))
        group_rows[source_id].append(index)
        group_images[source_id].add(image_id)
    eligible_sources = [
        source_id
        for source_id, images in group_images.items()
        if not (images & blocked_images)
    ]
    selected_sources = set(
        hash_rank(
            eligible_sources,
            seed=seed,
            namespace="predictability-docvqa-test",
        )[: int(config["test_sources"])]
    )
    if len(selected_sources) != int(config["test_sources"]):
        raise ValueError("not enough untouched DocVQA validation sources")
    source_indices = sorted(
        index for source in selected_sources for index in group_rows[source]
    )
    export_rows = [dataset[index] for index in source_indices]
    exported = export_benchmark_manifest(
        export_rows,
        source_indices=source_indices,
        task="docvqa",
        dataset_id=str(config["dataset_id"]),
        dataset_revision=str(config["revision"]),
        output_dir=stage / "docvqa" / "test",
        seed=seed,
        state_namespace="docvqa-predictability-test",
        dataset_split="validation",
        selection=str(config["test_selection"]),
    )
    reports["test"] = {
        "manifest": str(Path(exported["manifest"]).relative_to(stage)),
        "manifest_sha256": exported["manifest_sha256"],
        "states": exported["count"],
        "sources": exported["unique_sources"],
        "images": exported["unique_images"],
        "historically_opened": False,
        "source_indices": source_indices,
    }
    return reports


def hrbench_image_digest(encoded: str) -> str:
    from PIL import Image

    with Image.open(io.BytesIO(base64.b64decode(encoded))) as loaded:
        return image_digest(loaded.convert("RGB"))


def freeze_hrbench(
    *, root: Path, stage: Path, config: Mapping[str, Any], seed: int
) -> dict[str, Any]:
    import pyarrow.parquet as parquet  # type: ignore[import-untyped]

    parquet_path = checked_path(root, config["parquet"])
    rows = parquet.read_table(parquet_path, memory_map=True).to_pylist()
    if len(rows) != 800:
        raise ValueError(f"expected 800 HRBench rows, found {len(rows)}")
    identities = [
        SplitIdentity(
            item_id=str(row["index"]),
            source_id=f"hrbench:{row['index']}",
            image_rgb_sha256=hrbench_image_digest(str(row["image"])),
        )
        for row in rows
    ]
    assignments, split_audit = assign_disjoint_split_roles(
        identities,
        seed=seed,
        fractions=(
            float(config["train_fraction"]),
            float(config["validation_fraction"]),
            float(config["test_fraction"]),
        ),
    )
    reports: dict[str, Any] = {"split_audit": split_audit}
    for role in ("train", "validation", "test"):
        source_indices = [
            index
            for index, row in enumerate(rows)
            if assignments[str(row["index"])] == role
        ]
        selected_rows = [rows[index] for index in source_indices]
        exported = export_benchmark_manifest(
            selected_rows,
            source_indices=source_indices,
            task="hrbench8k",
            dataset_id=str(config["dataset_id"]),
            dataset_revision=str(config["revision"]),
            output_dir=stage / "hrbench" / role,
            seed=seed,
            state_namespace=f"hrbench8k-predictability-{role}",
            dataset_split=str(config["split"]),
            selection="source/RGB connected-component deterministic allocation",
        )
        reports[role] = {
            "manifest": str(Path(exported["manifest"]).relative_to(stage)),
            "manifest_sha256": exported["manifest_sha256"],
            "states": exported["count"],
            "sources": exported["unique_sources"],
            "images": exported["unique_images"],
            "historically_opened": False,
            "source_indices": source_indices,
        }
    return reports


def audit_outputs(stage: Path, reports: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for benchmark in ("chartqa", "docvqa", "hrbench"):
        role_rows = {
            role: read_manifest(stage / reports[benchmark][role]["manifest"])
            for role in ("train", "validation", "test")
        }
        pairwise = {}
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            left_sources = {str(row["source_id"]) for row in role_rows[left]}
            right_sources = {str(row["source_id"]) for row in role_rows[right]}
            left_images = {str(row["image_id"]) for row in role_rows[left]}
            right_images = {str(row["image_id"]) for row in role_rows[right]}
            pairwise[f"{left}_vs_{right}"] = {
                "source_overlap": len(left_sources & right_sources),
                "decoded_rgb_overlap": len(left_images & right_images),
            }
        if any(value for item in pairwise.values() for value in item.values()):
            raise ValueError(f"{benchmark} split leakage: {pairwise}")
        result[benchmark] = {"passed": True, "pairwise": pairwise}
    return result


def summarize_existing_benchmark(stage: Path, benchmark: str) -> dict[str, Any] | None:
    role_manifests = {
        role: stage / benchmark / role / "manifest.jsonl"
        for role in ("train", "validation", "test")
    }
    existing = {role: path.is_file() for role, path in role_manifests.items()}
    if not any(existing.values()):
        return None
    if not all(existing.values()):
        raise ValueError(
            f"partial staged benchmark cannot be resumed: {benchmark} {existing}"
        )
    reports: dict[str, Any] = {}
    for role, path in role_manifests.items():
        rows = read_manifest(path)
        reports[role] = {
            "manifest": str(path.relative_to(stage)),
            "manifest_sha256": sha256_file(path),
            "states": len(rows),
            "sources": len({str(row["source_id"]) for row in rows}),
            "images": len({str(row["image_id"]) for row in rows}),
            "historically_opened": benchmark != "hrbench" and role != "test",
            "resumed_from_complete_staging": True,
        }
        provenance = path.with_name("manifest.provenance.json")
        if provenance.is_file():
            payload = json.loads(provenance.read_text(encoding="utf-8"))
            if payload.get("manifest_sha256") != reports[role]["manifest_sha256"]:
                raise ValueError(f"staged provenance hash mismatch: {provenance}")
            if isinstance(payload.get("source_indices"), list):
                reports[role]["source_indices"] = payload["source_indices"]
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the three predictability datasets"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--resume-staging", action="store_true")
    args = parser.parse_args()
    root = args.repository_root.resolve()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "predictability_data_allocation_config_v1":
        raise ValueError("unexpected allocation config schema")
    destination = (root / str(config["output_root"])).resolve()
    stage = destination.with_name(destination.name + ".staging")
    if destination.exists():
        raise FileExistsError("allocation output already exists")
    if stage.exists() and not args.resume_staging:
        raise FileExistsError(
            "allocation staging path exists; explicit resume is required"
        )
    stage.mkdir(parents=True, exist_ok=args.resume_staging)
    blockers = scan_blocked_manifests(root, destination)
    seed = int(config["seed"])
    chartqa = summarize_existing_benchmark(stage, "chartqa")
    if chartqa is None:
        chartqa = freeze_chartqa(
            root=root,
            stage=stage,
            config=config["chartqa"],
            blocked_images=blockers["blocked_image_ids"],
            seed=seed,
        )
    docvqa = summarize_existing_benchmark(stage, "docvqa")
    if docvqa is None:
        docvqa = freeze_docvqa(
            root=root,
            stage=stage,
            config=config["docvqa"],
            blocked_images=blockers["blocked_image_ids"],
            blocked_sources=blockers["blocked_source_ids"],
            seed=seed,
        )
    hrbench = summarize_existing_benchmark(stage, "hrbench")
    if hrbench is None:
        hrbench = freeze_hrbench(
            root=root,
            stage=stage,
            config=config["hrbench"],
            seed=seed,
        )
    reports = {
        "chartqa": chartqa,
        "docvqa": docvqa,
        "hrbench": hrbench,
    }
    output_audit = audit_outputs(stage, reports)
    report = {
        "schema": "predictability_data_allocation_report_v1",
        "config": {
            "path": str(config_path.relative_to(root)),
            "sha256": sha256_file(config_path),
        },
        "seed": seed,
        "blocked_manifest_inventory": blockers["manifests"],
        "blocked_unique_image_ids": len(blockers["blocked_image_ids"]),
        "blocked_unique_source_ids": len(blockers["blocked_source_ids"]),
        "benchmarks": reports,
        "cross_role_audit": output_audit,
        "selection_used_model_outcomes": False,
        "new_test_rollouts_opened": False,
        "code_revision": os.environ.get("BE_CODE_REVISION"),
    }
    report_path = stage / "allocation.report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stage.replace(destination)
    print(
        json.dumps(
            {
                "output": str(destination),
                "report_sha256": sha256_file(destination / "allocation.report.json"),
                "cross_role_audit": output_audit,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
