from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from beyond_entropy.manifest_export import (
    export_benchmark_manifest,
    image_digest,
    stratified_unique_group_sample_indices,
)


DATASET_ID = "HuggingFaceM4/ChartQA"
DATASET_REVISION = "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5"
EXPECTED_PARQUET_HASHES = {
    "train-00000-of-00003-49492f364babfa44.parquet": (
        "169979f8a8c64ba93ac1be22916bf9b49c9c2e4c57f5c0104e6f671f2aacd83a"
    ),
    "train-00001-of-00003-7302bae5e425bbc7.parquet": (
        "a8819291729c807f163b93dcb2dc1fe60044d56412a0c0626cb7312df7392140"
    ),
    "train-00002-of-00003-194c9400785577a2.parquet": (
        "528133aec54c7e948f40b8af56b083b98f9b7d1fdcdd1cd9e375f3df984f7ae4"
    ),
}
EXPECTED_MANIFEST_HASHES = {
    "development": "3c485aa5c09cc9491f866ba5737a78c2b79c3539c6de2663c964b2cff90d814a",
    "validation": "d3178218853b10447228963e839716f0eac768b51bdc0f5b4a83268d3819b58b",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_manifest_image_ids(path: Path) -> set[str]:
    image_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict) or "image_id" not in value:
                raise ValueError(f"invalid manifest row {line_number}: {path}")
            image_ids.add(str(value["image_id"]))
    return image_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze an image-disjoint ChartQA train replication slice"
    )
    parser.add_argument("--parquet", type=Path, nargs="+", required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4500)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()

    parquet_paths = sorted(path.resolve() for path in args.parquet)
    actual_parquet_hashes = {path.name: sha256(path) for path in parquet_paths}
    if actual_parquet_hashes != EXPECTED_PARQUET_HASHES:
        raise ValueError(f"ChartQA train parquet hash mismatch: {actual_parquet_hashes}")
    actual_manifest_hashes = {
        "development": sha256(args.development_manifest),
        "validation": sha256(args.validation_manifest),
    }
    if actual_manifest_hashes != EXPECTED_MANIFEST_HASHES:
        raise ValueError(f"blocked manifest hash mismatch: {actual_manifest_hashes}")
    blocked_image_ids = read_manifest_image_ids(args.development_manifest)
    blocked_image_ids.update(read_manifest_image_ids(args.validation_manifest))

    from datasets import Dataset, concatenate_datasets  # type: ignore[import-untyped]

    dataset = concatenate_datasets(
        [Dataset.from_parquet(str(path)) for path in parquet_paths]
    )
    if len(dataset) != 28299:
        raise ValueError(f"expected 28299 ChartQA train rows, found {len(dataset)}")

    candidate_source_indices: list[int] = []
    candidate_labels: list[str] = []
    candidate_image_ids: list[str] = []
    excluded_overlap_rows = 0
    for source_index, row in enumerate(dataset):
        group = int(row["human_or_machine"])
        if group not in (0, 1):
            raise ValueError(f"unexpected human_or_machine at train index {source_index}")
        image_id = image_digest(row["image"].convert("RGB"))
        if image_id in blocked_image_ids:
            excluded_overlap_rows += 1
            continue
        candidate_source_indices.append(source_index)
        candidate_labels.append("human_train" if group == 0 else "augmented_train")
        candidate_image_ids.append(image_id)

    selected_candidate_indices = stratified_unique_group_sample_indices(
        candidate_labels,
        candidate_image_ids,
        count=args.count,
        seed=args.seed,
    )
    source_indices = [
        candidate_source_indices[index] for index in selected_candidate_indices
    ]
    rows = []
    for source_index in source_indices:
        row = dataset[source_index]
        labels = list(row["label"])
        if len(labels) != 1:
            raise ValueError(f"expected one target label at train index {source_index}")
        group = int(row["human_or_machine"])
        rows.append(
            {
                "image": row["image"],
                "type": "human_train" if group == 0 else "augmented_train",
                "question": str(row["query"]),
                "answer": str(labels[0]),
            }
        )

    result = export_benchmark_manifest(
        rows,
        source_indices=source_indices,
        task="chartqa",
        dataset_id=DATASET_ID,
        dataset_revision=DATASET_REVISION,
        output_dir=args.output_dir,
        seed=args.seed,
        state_namespace="chartqa-train-replication",
    )
    expected_strata = {"augmented_train": args.count // 2, "human_train": args.count // 2}
    if args.count % 2 or result["stratum_counts"] != expected_strata:
        raise ValueError(f"replication sample is not balanced: {result['stratum_counts']}")
    if result["unique_images"] != args.count:
        raise ValueError("replication sample must contain exactly one state per image")
    result.update(
        {
            "split": "train",
            "selection": (
                "seeded balanced sample with one state per image after excluding "
                "all development and validation images"
            ),
            "source_parquets": [str(path) for path in parquet_paths],
            "source_parquet_sha256": actual_parquet_hashes,
            "blocked_manifests": {
                "development": str(args.development_manifest.resolve()),
                "validation": str(args.validation_manifest.resolve()),
            },
            "blocked_manifest_sha256": actual_manifest_hashes,
            "blocked_unique_images": len(blocked_image_ids),
            "excluded_overlap_rows": excluded_overlap_rows,
            "eligible_rows": len(candidate_source_indices),
        }
    )
    provenance_path = args.output_dir / "manifest.provenance.json"
    provenance_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
