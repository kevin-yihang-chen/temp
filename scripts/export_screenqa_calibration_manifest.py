#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

from beyond_entropy.manifest_audit import audit_manifest
from beyond_entropy.manifest_export import export_benchmark_manifest


DATASET_REVISION = "1dfdbccaf56948821b5fa8ffe5d186fe4751e46d"
SHORT_TRAIN_SHA256 = "660611371a9b8342000ca69af85063c6404061de6aad0251b9f4910fdaceb800"
ALLOCATION_SHA256 = "ccfc2c0f18d36f6b31a6200c31a991d75ba6bb6ed3160b72ed5cfcca25473c49"
PROTOCOL_SHA256 = "c6118d8a013a171c3eecad374a3271e3bf00dfd199864d3efaab27c7b44e36b7"
EXPECTED_IMAGES = 4001
EXPECTED_QA_ROWS = 9951
EXPECTED_SOURCES = 1016
EXPECTED_SOURCE_ROWS = 68951
TOP_LEVEL_ROW = re.compile(r'\{"image_id"\s*:\s*(\d+)\s*,\s*"question"\s*:')
HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
GIT_REVISION = re.compile(r"[0-9a-f]{40,64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def verify_sums(directory: Path) -> None:
    checksum_path = directory / "SHA256SUMS"
    if not checksum_path.is_file():
        raise FileNotFoundError(f"checksum bundle is missing: {checksum_path}")
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, filename = line.split(maxsplit=1)
        path = directory / filename.strip()
        if sha256_file(path) != digest:
            raise ValueError(f"checksum mismatch for {path}")


def _require_empty(path: Path, name: str) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"{name} must remain unmaterialized")


def verify_candidate(candidate_dir: Path) -> dict[str, Any]:
    verify_sums(candidate_dir)
    model_path = candidate_dir / "model.json"
    report_path = candidate_dir / "selected-ranker-report.json"
    audit_path = candidate_dir / "candidate.audit.json"
    model = _load_json(model_path)
    audit = _load_json(audit_path)
    if (
        audit.get("candidate_frozen") is not True
        or audit.get("semantic_escalation_required") is not False
        or audit.get("calibration_outcomes_opened") is not False
        or audit.get("formal_outcomes_opened") is not False
        or audit.get("reserve_outcomes_opened") is not False
    ):
        raise ValueError("ScreenQA low-capacity candidate is not safely frozen")
    if audit.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("ScreenQA candidate protocol hash mismatch")
    model_sha256 = sha256_file(model_path)
    report_sha256 = sha256_file(report_path)
    if audit.get("selected_model_sha256") != model_sha256:
        raise ValueError("ScreenQA frozen candidate model hash mismatch")
    if audit.get("selected_ranker_report_sha256") != report_sha256:
        raise ValueError("ScreenQA selected ranker report hash mismatch")
    if model.get("feature_mode") not in {
        "context-geometry",
        "spatial-context-geometry",
    } or model.get("feature_mode") != audit.get("selected_feature_mode"):
        raise ValueError("ScreenQA frozen candidate feature mode mismatch")
    if model.get("threshold") is not None:
        raise ValueError("ScreenQA execution threshold must be unset before calibration")
    development_threshold = model.get("development_oof_threshold")
    if not isinstance(development_threshold, (int, float)) or not math.isfinite(
        float(development_threshold)
    ):
        raise ValueError("ScreenQA development OOF threshold is missing")
    raw_grid = model.get("threshold_grid")
    if not isinstance(raw_grid, list) or not raw_grid:
        raise ValueError("ScreenQA frozen threshold grid is missing")
    threshold_grid = [float(value) for value in raw_grid]
    if any(not math.isfinite(value) for value in threshold_grid) or any(
        strict <= permissive
        for strict, permissive in zip(threshold_grid, threshold_grid[1:])
    ):
        raise ValueError("ScreenQA frozen threshold grid is not strictly descending")
    if audit.get("threshold_grid") != raw_grid:
        raise ValueError("ScreenQA candidate audit threshold grid mismatch")
    contract = model.get("calibration_contract")
    expected_contract = {
        "method": "fixed_sequence_bounded_mean_kl_ltt_v1",
        "threshold_order": "strict_to_permissive_descending",
        "threshold_rate_weighting": "pooled_development_decisions",
        "target_pooled_development_call_rates": [
            0.005,
            0.01,
            0.015,
            0.02,
            0.03,
            0.05,
        ],
        "constraints": [
            {"kind": "induced_harm", "limit": 0.005},
            {"kind": "net_negative_call_mass", "limit": 0.02},
        ],
        "family_error": 0.05,
        "per_step_p_cutoff": 0.025,
        "min_source_call_rate": 0.01,
        "min_source_utility": 0.001,
        "calibration_sources": EXPECTED_SOURCES,
        "calibration_decisions": EXPECTED_QA_ROWS,
        "formal_sources": 1471,
        "formal_decisions": 14672,
    }
    if contract != expected_contract:
        raise ValueError("ScreenQA calibration contract mismatch")
    selection = model.get("candidate_selection")
    if not isinstance(selection, Mapping):
        raise ValueError("ScreenQA candidate selection provenance is missing")
    expected_selection = {
        "protocol_sha256": PROTOCOL_SHA256,
        "selected_feature_mode": model["feature_mode"],
        "ranker_report_sha256": report_sha256,
        "ranker_training_outcomes_used": True,
        "calibration_outcomes_used": False,
        "formal_outcomes_used": False,
        "reserve_outcomes_used": False,
    }
    for key, expected in expected_selection.items():
        if selection.get(key) != expected:
            raise ValueError(f"ScreenQA candidate selection {key} mismatch")
    for key in ("ranker_rollouts_sha256", "raw_model_sha256"):
        value = selection.get(key)
        if not isinstance(value, str) or HEX_SHA256.fullmatch(value) is None:
            raise ValueError(f"ScreenQA candidate selection {key} is malformed")
    code_revision = selection.get("ranker_fit_code_revision")
    if not isinstance(code_revision, str) or GIT_REVISION.fullmatch(code_revision) is None:
        raise ValueError("ScreenQA candidate fit revision is malformed")
    return {
        "model": str(model_path.resolve()),
        "model_sha256": model_sha256,
        "report": str(report_path.resolve()),
        "report_sha256": report_sha256,
        "audit": str(audit_path.resolve()),
        "audit_sha256": sha256_file(audit_path),
        "bundle_sha256": sha256_file(candidate_dir / "SHA256SUMS"),
        "feature_mode": model["feature_mode"],
        "threshold_grid": threshold_grid,
    }


def extract_selected_rows(
    path: Path, selected_image_ids: set[str]
) -> tuple[list[dict[str, Any]], list[int], int]:
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
        end = (
            matches[source_index + 1].start()
            if source_index + 1 < len(matches)
            else closing_bracket
        )
        serialized = text[match.start() : end].strip()
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
        raise ValueError(f"selected ScreenQA identities missing: {missing[:10]}")
    return rows, source_indices, len(matches)


def load_calibration_allocation(
    allocation_dir: Path,
) -> tuple[set[str], dict[str, str]]:
    verify_sums(allocation_dir)
    allocation_path = allocation_dir / "allocation.json"
    if sha256_file(allocation_path) != ALLOCATION_SHA256:
        raise ValueError("ScreenQA allocation hash mismatch")
    allocation = _load_json(allocation_path)
    if allocation.get("selection_contract") != {
        "formal_outcomes_opened": False,
        "official_validation_test_untouched": True,
        "outcomes_accessed": False,
        "question_text_accessed": False,
        "reserve_outcomes_opened": False,
        "target_fields_accessed": False,
    }:
        raise ValueError("ScreenQA sealed allocation contract changed")
    role = allocation["roles"]["risk_calibration"]
    selected_ids = {str(image_id) for image_id in role["image_ids"]}
    if (
        len(selected_ids) != EXPECTED_IMAGES
        or int(role["allocated_qa_rows_identity_only"]) != EXPECTED_QA_ROWS
        or int(role["component_count"]) != EXPECTED_SOURCES
    ):
        raise ValueError("ScreenQA calibration allocation counts changed")
    component_by_image: dict[str, str] = {}
    with (allocation_dir / "component_roles.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["role"] != "risk_calibration":
                continue
            component_id = str(row["component_id"])
            for image_id in row["image_ids"]:
                image_key = str(image_id)
                if image_key in component_by_image:
                    raise ValueError("calibration image occurs in multiple components")
                component_by_image[image_key] = component_id
    if set(component_by_image) != selected_ids:
        raise ValueError("calibration component mapping does not cover allocation")
    if len(set(component_by_image.values())) != EXPECTED_SOURCES:
        raise ValueError("calibration source-component count mismatch")
    return selected_ids, component_by_image


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Open and export only the frozen ScreenQA calibration role after "
            "the sole candidate has been serialized"
        )
    )
    parser.add_argument("--allocation-dir", type=Path, required=True)
    parser.add_argument("--short-train", type=Path, required=True)
    parser.add_argument("--rico-images-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--formal-output-dir", type=Path, required=True)
    parser.add_argument("--reserve-output-dir", type=Path, required=True)
    parser.add_argument("--untouched-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    _require_empty(args.formal_output_dir.resolve(), "formal output")
    _require_empty(args.reserve_output_dir.resolve(), "reserve output")
    _require_empty(args.untouched_output_dir.resolve(), "untouched output")
    candidate = verify_candidate(args.candidate_dir.resolve())
    if sha256_file(args.short_train) != SHORT_TRAIN_SHA256:
        raise ValueError("ScreenQA Short train annotation hash mismatch")
    selected_ids, component_by_image = load_calibration_allocation(
        args.allocation_dir.resolve()
    )
    rows, source_indices, source_row_count = extract_selected_rows(
        args.short_train.resolve(), selected_ids
    )
    if len(rows) != EXPECTED_QA_ROWS or source_row_count != EXPECTED_SOURCE_ROWS:
        raise ValueError("ScreenQA selected/source calibration QA count mismatch")

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

    provenance = export_benchmark_manifest(
        export_rows,
        source_indices=source_indices,
        task="screenqa",
        dataset_id="google-research-datasets/screen_qa",
        dataset_revision=DATASET_REVISION,
        dataset_split="train",
        output_dir=args.output_dir,
        seed=20260831,
        state_namespace="screenqa-train-factorized-v1-calibration",
        selection="frozen SHA-256 whole app-plus-duplicate component allocation",
        selection_metadata={
            "allocation": str((args.allocation_dir / "allocation.json").resolve()),
            "allocation_sha256": ALLOCATION_SHA256,
            "role": "risk_calibration",
            "selected_image_count": EXPECTED_IMAGES,
            "selected_qa_count": EXPECTED_QA_ROWS,
            "selected_source_count": EXPECTED_SOURCES,
            "short_train_sha256": SHORT_TRAIN_SHA256,
            "candidate": candidate,
            "candidate_frozen_before_annotation_deserialization": True,
            "only_selected_annotation_objects_deserialized": True,
            "unselected_question_or_target_fields_accessed": False,
            "formal_outcomes_opened": False,
            "reserve_outcomes_opened": False,
            "untouched_outcomes_opened": False,
        },
    )
    manifest_audit = audit_manifest(args.output_dir)
    manifest_rows = [
        json.loads(line)
        for line in (args.output_dir / "manifest.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    exported_ids = {str(row["rico_ui_id"]) for row in manifest_rows}
    expected_sources = {
        f"screenqa:{component_id}" for component_id in set(component_by_image.values())
    }
    if exported_ids != selected_ids:
        raise ValueError("exported ScreenQA calibration identities differ from allocation")
    if manifest_audit["_sources"] != expected_sources:
        raise ValueError("exported ScreenQA calibration sources differ from allocation")
    if manifest_audit["count"] != EXPECTED_QA_ROWS:
        raise ValueError("exported ScreenQA calibration manifest count changed")
    clean_manifest_audit = {
        key: value for key, value in manifest_audit.items() if not key.startswith("_")
    }
    access_audit = {
        "passed": True,
        "scientific_status": (
            "only frozen risk-calibration labels opened after sole candidate freeze"
        ),
        "allocation_sha256": ALLOCATION_SHA256,
        "candidate": candidate,
        "short_train_sha256": SHORT_TRAIN_SHA256,
        "source_annotation_rows_scanned_for_boundaries": source_row_count,
        "annotation_objects_deserialized": len(rows),
        "selected_rico_images": len(selected_ids),
        "selected_source_components": len(expected_sources),
        "unselected_annotation_objects_deserialized": 0,
        "ranker_training_outcomes_previously_used": True,
        "risk_calibration_opened": True,
        "formal_test_opened": False,
        "reserve_opened": False,
        "untouched_opened": False,
        "official_validation_test_opened": False,
        "manifest": clean_manifest_audit,
        "export_provenance": provenance,
    }
    write_json(args.output_dir / "manifest.audit.json", access_audit)
    with (args.output_dir / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in sorted(args.output_dir.iterdir()):
            if path.is_file() and path.name != "SHA256SUMS":
                handle.write(f"{sha256_file(path)}  {path.name}\n")
    print(json.dumps(access_audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
