from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from beyond_entropy.manifest_audit import audit_manifest
from beyond_entropy.manifest_export import export_benchmark_manifest


DATASET_REVISION = "1dfdbccaf56948821b5fa8ffe5d186fe4751e46d"
SHORT_TRAIN_SHA256 = "660611371a9b8342000ca69af85063c6404061de6aad0251b9f4910fdaceb800"
ALLOCATION_SHA256 = "ccfc2c0f18d36f6b31a6200c31a991d75ba6bb6ed3160b72ed5cfcca25473c49"
EXPECTED_IMAGES = 6007
EXPECTED_UNIQUE_RGB_IMAGES = 5993
EXPECTED_QA_ROWS = 14511
TOP_LEVEL_ROW = re.compile(r'\{"image_id"\s*:\s*(\d+)\s*,\s*"question"\s*:')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sums(directory: Path) -> None:
    for line in (directory / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, filename = line.split(maxsplit=1)
        path = directory / filename.strip()
        if sha256_file(path) != digest:
            raise RuntimeError(f"allocation checksum mismatch for {path}")


def extract_selected_rows(
    path: Path, selected_image_ids: set[str]
) -> tuple[list[dict[str, Any]], list[int], int]:
    """Deserialize only selected top-level objects from the one-line JSON array.

    The raw file must be read to locate object boundaries, but target-bearing
    fields in unselected objects are never passed to ``json.loads``.
    """
    text = path.read_text(encoding="utf-8")
    matches = list(TOP_LEVEL_ROW.finditer(text))
    if not matches:
        raise ValueError("no ScreenQA top-level rows found")
    closing_bracket = text.rfind("]")
    if closing_bracket <= matches[-1].start():
        raise ValueError("ScreenQA JSON array closing bracket is missing")
    rows: list[dict[str, Any]] = []
    source_indices: list[int] = []
    seen_images: set[str] = set()
    for source_index, match in enumerate(matches):
        image_id = match.group(1)
        if image_id not in selected_image_ids:
            continue
        end = matches[source_index + 1].start() if source_index + 1 < len(matches) else closing_bracket
        serialized = text[match.start():end].strip()
        if serialized.endswith(","):
            serialized = serialized[:-1].rstrip()
        row = json.loads(serialized)
        if str(row.get("image_id", "")) != image_id:
            raise ValueError(f"ScreenQA row identity drift at source index {source_index}")
        rows.append(row)
        source_indices.append(source_index)
        seen_images.add(image_id)
    if seen_images != selected_image_ids:
        missing = sorted(selected_image_ids - seen_images, key=int)
        raise ValueError(f"selected ScreenQA identities missing from annotations: {missing[:10]}")
    return rows, source_indices, len(matches)


def load_allocation(
    allocation_dir: Path,
) -> tuple[dict[str, Any], set[str], dict[str, str]]:
    verify_sums(allocation_dir)
    allocation_path = allocation_dir / "allocation.json"
    if sha256_file(allocation_path) != ALLOCATION_SHA256:
        raise RuntimeError("ScreenQA allocation hash mismatch")
    allocation = json.loads(allocation_path.read_text(encoding="utf-8"))
    if allocation["selection_contract"] != {
        "formal_outcomes_opened": False,
        "official_validation_test_untouched": True,
        "outcomes_accessed": False,
        "question_text_accessed": False,
        "reserve_outcomes_opened": False,
        "target_fields_accessed": False,
    }:
        raise RuntimeError("ScreenQA sealed allocation contract changed")
    role = allocation["roles"]["ranker_training"]
    selected_ids = {str(image_id) for image_id in role["image_ids"]}
    if len(selected_ids) != EXPECTED_IMAGES or int(role["allocated_qa_rows_identity_only"]) != EXPECTED_QA_ROWS:
        raise RuntimeError("ScreenQA ranker allocation counts changed")
    component_by_image: dict[str, str] = {}
    with (allocation_dir / "component_roles.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["role"] != "ranker_training":
                continue
            component_id = str(row["component_id"])
            for image_id in row["image_ids"]:
                image_id = str(image_id)
                if image_id in component_by_image:
                    raise RuntimeError("ranker image occurs in multiple components")
                component_by_image[image_id] = component_id
    if set(component_by_image) != selected_ids:
        raise RuntimeError("ranker component mapping does not cover allocation")
    return allocation, selected_ids, component_by_image


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open and export only the frozen ScreenQA ranker-training role"
    )
    parser.add_argument("--allocation-dir", type=Path, required=True)
    parser.add_argument("--short-train", type=Path, required=True)
    parser.add_argument("--rico-images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help=(
            "audit and finalize an already exported manifest after a recoverable "
            "post-export failure"
        ),
    )
    args = parser.parse_args()

    if args.output_dir.exists() and not args.finalize_existing:
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.finalize_existing and not args.output_dir.is_dir():
        raise FileNotFoundError(
            f"existing export directory does not exist: {args.output_dir}"
        )
    if sha256_file(args.short_train) != SHORT_TRAIN_SHA256:
        raise RuntimeError("ScreenQA Short train annotation hash mismatch")
    allocation, selected_ids, component_by_image = load_allocation(args.allocation_dir)
    rows, source_indices, source_row_count = extract_selected_rows(args.short_train, selected_ids)
    if len(rows) != EXPECTED_QA_ROWS or source_row_count != 68951:
        raise RuntimeError("ScreenQA selected/source QA count mismatch")

    export_rows: list[dict[str, Any]] = []
    for row in rows:
        image_id = str(row["image_id"])
        image_path = args.rico_images_dir / f"{image_id}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(f"selected RICO image is missing: {image_path}")
        export_rows.append(
            {
                **row,
                "image": str(image_path.resolve()),
                "source_group_id": component_by_image[image_id],
            }
        )

    if args.finalize_existing:
        result = json.loads(
            (args.output_dir / "manifest.provenance.json").read_text(encoding="utf-8")
        )
    else:
        result = export_benchmark_manifest(
            export_rows,
            source_indices=source_indices,
            task="screenqa",
            dataset_id="google-research-datasets/screen_qa",
            dataset_revision=DATASET_REVISION,
            dataset_split="train",
            output_dir=args.output_dir,
            seed=20260831,
            state_namespace="screenqa-train-factorized-v1-ranker",
            selection="frozen SHA-256 whole app-plus-duplicate component allocation",
            selection_metadata={
                "allocation": str((args.allocation_dir / "allocation.json").resolve()),
                "allocation_sha256": ALLOCATION_SHA256,
                "role": "ranker_training",
                "selected_image_count": EXPECTED_IMAGES,
                "selected_qa_count": EXPECTED_QA_ROWS,
                "short_train_sha256": SHORT_TRAIN_SHA256,
                "only_selected_annotation_objects_deserialized": True,
                "unselected_question_or_target_fields_accessed": False,
                "formal_outcomes_opened": False,
                "calibration_outcomes_opened": False,
                "reserve_outcomes_opened": False,
            },
        )
    audit = audit_manifest(args.output_dir)
    manifest_rows = [
        json.loads(line)
        for line in (args.output_dir / "manifest.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    exported_rico_ids = {str(row["rico_ui_id"]) for row in manifest_rows}
    expected_sources = {f"screenqa:{component_id}" for component_id in set(component_by_image.values())}
    if exported_rico_ids != selected_ids:
        raise RuntimeError("exported ScreenQA RICO identities differ from allocation")
    if audit["_sources"] != expected_sources:
        raise RuntimeError("exported ScreenQA source components differ from allocation")
    if (
        audit["count"] != EXPECTED_QA_ROWS
        or audit["unique_images"] != EXPECTED_UNIQUE_RGB_IMAGES
    ):
        raise RuntimeError("exported ScreenQA manifest counts differ from allocation")
    clean_audit = {key: value for key, value in audit.items() if not key.startswith("_")}
    access_audit = {
        "passed": True,
        "scientific_status": "only frozen ranker-training labels opened",
        "allocation_sha256": ALLOCATION_SHA256,
        "short_train_sha256": SHORT_TRAIN_SHA256,
        "source_annotation_rows_scanned_for_boundaries": source_row_count,
        "annotation_objects_deserialized": len(rows),
        "selected_rico_images": len(selected_ids),
        "unique_decoded_rgb_images": audit["unique_images"],
        "selected_rico_ids_collapsed_by_exact_rgb_identity": (
            len(selected_ids) - audit["unique_images"]
        ),
        "selected_source_components": len(expected_sources),
        "unselected_annotation_objects_deserialized": 0,
        "risk_calibration_opened": False,
        "formal_test_opened": False,
        "reserve_opened": False,
        "untouched_opened": False,
        "official_validation_test_opened": False,
        "manifest": clean_audit,
        "export_provenance": result,
    }
    write_json(args.output_dir / "manifest.audit.json", access_audit)
    with (args.output_dir / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in sorted(args.output_dir.iterdir()):
            if path.is_file() and path.name != "SHA256SUMS":
                handle.write(f"{sha256_file(path)}  {path.name}\n")
    print(json.dumps(access_audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
