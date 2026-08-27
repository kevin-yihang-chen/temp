from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from beyond_entropy.manifest_export import export_benchmark_manifest


DATASET_ID = "HuggingFaceM4/ChartQA"
DATASET_REVISION = "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5"
DEVELOPMENT_IMAGE_OVERLAPS = {
    "93424b05310af662a5c0c2d046b69d0eb84b2d9df6d8dd3479797fa71739f066"
}


def image_digest(image: object) -> str:
    converted = image.convert("RGB")  # type: ignore[attr-defined]
    width, height = converted.size
    digest = hashlib.sha256()
    digest.update(f"RGB:{width}x{height}:".encode())
    digest.update(converted.tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the official ChartQA val split")
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    from datasets import Dataset  # type: ignore[import-untyped]

    parquet_path = args.parquet.resolve()
    dataset = Dataset.from_parquet(str(parquet_path))
    if len(dataset) != 1920:
        raise ValueError(f"expected 1920 ChartQA val examples, found {len(dataset)}")
    rows = []
    source_indices = []
    excluded_source_indices = []
    for index, row in enumerate(dataset):
        labels = list(row["label"])
        if len(labels) != 1:
            raise ValueError(f"expected one target label at val index {index}")
        group = int(row["human_or_machine"])
        if group not in (0, 1):
            raise ValueError(f"unexpected human_or_machine value at val index {index}")
        if image_digest(row["image"]) in DEVELOPMENT_IMAGE_OVERLAPS:
            excluded_source_indices.append(index)
            continue
        rows.append(
            {
                "image": row["image"],
                "type": "human_val" if group == 0 else "augmented_val",
                "question": str(row["query"]),
                "answer": str(labels[0]),
            }
        )
        source_indices.append(index)
    if len(rows) != 1918 or excluded_source_indices != [300, 301]:
        raise ValueError("unexpected ChartQA val/test image-overlap exclusion result")
    result = export_benchmark_manifest(
        rows,
        source_indices=source_indices,
        task="chartqa",
        dataset_id=DATASET_ID,
        dataset_revision=DATASET_REVISION,
        output_dir=args.output_dir,
        seed=0,
        state_namespace="chartqa-val",
    )
    result["split"] = "val"
    result["selection"] = "entire official val split excluding development-image overlap"
    result["excluded_development_image_ids"] = sorted(DEVELOPMENT_IMAGE_OVERLAPS)
    result["excluded_source_indices"] = excluded_source_indices
    result["source_parquet"] = str(parquet_path)
    result["source_parquet_sha256"] = hashlib.sha256(
        parquet_path.read_bytes()
    ).hexdigest()
    provenance_path = args.output_dir / "manifest.provenance.json"
    provenance_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
