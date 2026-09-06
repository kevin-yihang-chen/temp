#!/usr/bin/env python3
"""Freeze outcome-blind Phase-C train and fresh sequential held-out manifests."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from beyond_entropy.manifest_export import (
    export_benchmark_manifest,
    image_digest,
    stratified_unique_group_sample_indices,
)
from beyond_entropy.phase_c_allocation import (
    allocate_hrbench_phase_c,
    hash_rank,
    role_overlap_audit,
    select_complete_groups,
)

# Reuse the already-audited manifest/path/parquet primitives instead of
# reimplementing image serialization or historical-bank inventory logic.
from freeze_predictability_data import (  # type: ignore[import-not-found]
    adjusted_subset_rows,
    checked_path,
    load_parquet_dataset,
    read_manifest,
    scan_blocked_manifests,
    sha256_file,
    write_jsonl,
)


def _role_summary(path: Path, rows: list[dict[str, Any]], *, fresh: bool) -> dict[str, Any]:
    return {
        "manifest": str(path),
        "manifest_sha256": sha256_file(path),
        "states": len(rows),
        "sources": len({str(row["source_id"]) for row in rows}),
        "images": len({str(row["image_id"]) for row in rows}),
        "fresh_sequential_outcomes": fresh,
    }


def _write_adjusted(
    rows: list[dict[str, Any]], *, source_manifest: Path, destination: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    destination.mkdir(parents=True, exist_ok=False)
    manifest = destination / "manifest.jsonl"
    adjusted = adjusted_subset_rows(rows, source_manifest, destination)
    write_jsonl(adjusted, manifest)
    return manifest, adjusted


def freeze_chartqa(
    root: Path, stage: Path, spec: Mapping[str, Any], blockers: Mapping[str, Any], seed: int,
) -> dict[str, Any]:
    source_manifest = checked_path(root, spec["train_manifest"])
    source_rows = read_manifest(source_manifest)
    train_rows, train_groups = select_complete_groups(
        source_rows, group_key="source_id", group_count=int(spec["train_states"]),
        seed=seed, namespace="factorized-phase-c-chartqa-train",
    )
    train_path, adjusted_train = _write_adjusted(
        train_rows, source_manifest=source_manifest, destination=stage / "chartqa" / "train"
    )

    parquet_paths = [checked_path(root, item) for item in spec["parquets"]]
    dataset = load_parquet_dataset(parquet_paths)
    candidate_indices, candidate_labels, candidate_images = [], [], []
    blocked_images = set(blockers["blocked_image_ids"])
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
        candidate_labels, candidate_images, count=int(spec["heldout_states"]), seed=seed,
    )
    source_indices = [candidate_indices[position] for position in selected_positions]
    export_rows = []
    for index in source_indices:
        row = dataset[index]
        labels = list(row["label"])
        if len(labels) != 1:
            raise ValueError("ChartQA row must contain exactly one label")
        export_rows.append({
            "image": row["image"],
            "type": "human_train" if int(row["human_or_machine"]) == 0 else "augmented_train",
            "question": str(row["query"]),
            "answer": str(labels[0]),
        })
    exported = export_benchmark_manifest(
        export_rows, source_indices=source_indices, task="chartqa",
        dataset_id=str(spec["dataset_id"]), dataset_revision=str(spec["revision"]),
        output_dir=stage / "chartqa" / "heldout", seed=seed,
        state_namespace="chartqa-factorized-phase-c-heldout", dataset_split="train",
        selection="outcome-blind balanced hash sample excluding every historical manifest",
    )
    heldout_path = Path(exported["manifest"])
    heldout_rows = read_manifest(heldout_path)
    overlap = role_overlap_audit(adjusted_train, heldout_rows)
    if {row["image_id"] for row in heldout_rows} & blocked_images:
        raise RuntimeError("ChartQA held-out collided with historical image")
    return {
        "train": _role_summary(train_path, adjusted_train, fresh=False),
        "heldout": _role_summary(heldout_path, heldout_rows, fresh=True),
        "train_selected_source_ids": train_groups,
        "heldout_source_indices": source_indices,
        "historical_image_overlap": 0,
        "role_overlap": overlap,
    }


def freeze_docvqa(
    root: Path, stage: Path, spec: Mapping[str, Any], blockers: Mapping[str, Any], seed: int,
) -> dict[str, Any]:
    source_manifest = checked_path(root, spec["train_manifest"])
    source_rows = read_manifest(source_manifest)
    train_rows, train_groups = select_complete_groups(
        source_rows, group_key="source_id", group_count=int(spec["train_sources"]),
        seed=seed, namespace="factorized-phase-c-docvqa-train",
    )
    train_path, adjusted_train = _write_adjusted(
        train_rows, source_manifest=source_manifest, destination=stage / "docvqa" / "train"
    )

    dataset = load_parquet_dataset([
        checked_path(root, item) for item in spec["validation_parquets"]
    ])
    blocked_images = set(blockers["blocked_image_ids"])
    blocked_sources = set(blockers["blocked_source_ids"])
    group_rows: dict[str, list[int]] = defaultdict(list)
    group_images: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(dataset):
        source_id = f"docvqa:{str(row['docId']).strip()}"
        if source_id in blocked_sources:
            continue
        image_id = image_digest(row["image"].convert("RGB"))
        group_rows[source_id].append(index)
        group_images[source_id].add(image_id)
    eligible = [
        source for source, images in group_images.items() if not (images & blocked_images)
    ]
    selected_sources = hash_rank(
        eligible, seed=seed, namespace="factorized-phase-c-docvqa-heldout"
    )[: int(spec["heldout_sources"])]
    if len(selected_sources) != int(spec["heldout_sources"]):
        raise ValueError("insufficient new DocVQA held-out sources")
    source_indices = sorted(
        index for source in selected_sources for index in group_rows[source]
    )
    exported = export_benchmark_manifest(
        [dataset[index] for index in source_indices], source_indices=source_indices,
        task="docvqa", dataset_id=str(spec["dataset_id"]),
        dataset_revision=str(spec["revision"]), output_dir=stage / "docvqa" / "heldout",
        seed=seed, state_namespace="docvqa-factorized-phase-c-heldout",
        dataset_split="validation",
        selection="whole-doc outcome-blind hash sample excluding every historical manifest",
    )
    heldout_path = Path(exported["manifest"])
    heldout_rows = read_manifest(heldout_path)
    overlap = role_overlap_audit(adjusted_train, heldout_rows)
    if ({row["image_id"] for row in heldout_rows} & blocked_images
            or {row["source_id"] for row in heldout_rows} & blocked_sources):
        raise RuntimeError("DocVQA held-out collided with historical source or image")
    return {
        "train": _role_summary(train_path, adjusted_train, fresh=False),
        "heldout": _role_summary(heldout_path, heldout_rows, fresh=True),
        "train_selected_source_ids": train_groups,
        "heldout_selected_source_ids": selected_sources,
        "heldout_source_indices": source_indices,
        "historical_source_overlap": 0,
        "historical_image_overlap": 0,
        "role_overlap": overlap,
    }


def freeze_hrbench(
    root: Path, stage: Path, spec: Mapping[str, Any], seed: int,
) -> dict[str, Any]:
    source_manifest = checked_path(root, spec["train_manifest"])
    source_rows = read_manifest(source_manifest)
    used_images = set()
    for rollout_spec in spec["historical_sequential_rollouts"]:
        path = checked_path(root, rollout_spec)
        for row in read_manifest(path):
            used_images.add(str(row["image_id"]))
    allocation = allocate_hrbench_phase_c(
        source_rows, historically_used_image_ids=used_images,
        heldout_image_count=int(spec["heldout_images"]), seed=seed,
    )
    train_path, adjusted_train = _write_adjusted(
        allocation["train_rows"], source_manifest=source_manifest,
        destination=stage / "hrbench" / "train",
    )
    heldout_path, adjusted_heldout = _write_adjusted(
        allocation["heldout_rows"], source_manifest=source_manifest,
        destination=stage / "hrbench" / "heldout",
    )
    overlap = role_overlap_audit(adjusted_train, adjusted_heldout)
    if {row["image_id"] for row in adjusted_heldout} & used_images:
        raise RuntimeError("HRBench held-out contains a prior sequential image")
    return {
        "train": _role_summary(train_path, adjusted_train, fresh=False),
        "heldout": _role_summary(heldout_path, adjusted_heldout, fresh=True),
        "heldout_image_ids": allocation["heldout_image_ids"],
        "eligible_unseen_image_count": allocation["eligible_unseen_image_count"],
        "historically_used_image_count": allocation["historically_used_image_count"],
        "historical_sequential_image_overlap": 0,
        "role_overlap": overlap,
        "limitation": (
            "same official HRBench-8K pool as prior static work, but held-out image "
            "groups have no prior sequential STOP/CONTINUE outcomes"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.repository_root.resolve()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    if config.get("schema") != "factorized_phase_c_allocation_v1":
        raise ValueError("unexpected Phase-C allocation schema")
    destination = (root / str(config["output_root"])).resolve()
    stage = destination.with_name(destination.name + ".staging")
    if destination.exists() or stage.exists():
        raise FileExistsError("Phase-C allocation destination or staging already exists")
    stage.mkdir(parents=True)
    blockers = scan_blocked_manifests(root, destination)
    seed = int(config["seed"])
    try:
        benchmarks = {
            "chartqa": freeze_chartqa(root, stage, config["chartqa"], blockers, seed),
            "docvqa": freeze_docvqa(root, stage, config["docvqa"], blockers, seed),
            "hrbench": freeze_hrbench(root, stage, config["hrbench"], seed),
        }
        report = {
            "schema": "factorized_phase_c_allocation_report_v1",
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "code_revision": os.environ.get("BE_CODE_REVISION"),
            "seed": seed,
            "selection_used_model_outcomes": False,
            "heldout_sequential_outcomes_opened": False,
            "blocked_manifest_inventory": blockers["manifests"],
            "blocked_unique_image_ids": len(blockers["blocked_image_ids"]),
            "blocked_unique_source_ids": len(blockers["blocked_source_ids"]),
            "benchmarks": benchmarks,
        }
        report_path = stage / "allocation.report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        stage.replace(destination)
    except Exception:
        # Keep staging for a forensic diagnosis; never silently overwrite it.
        raise
    final_report = destination / "allocation.report.json"
    print(json.dumps({
        "output": str(destination), "report_sha256": sha256_file(final_report),
        "counts": {
            benchmark: {
                role: benchmarks[benchmark][role]["states"]
                for role in ("train", "heldout")
            }
            for benchmark in benchmarks
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
