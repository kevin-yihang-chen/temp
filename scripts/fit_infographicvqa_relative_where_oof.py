#!/usr/bin/env python3
"""Fit the frozen InfographicVQA relative-where family under source OOF."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from beyond_entropy.dataset import read_jsonl
from beyond_entropy.infographicvqa_decar import assemble_decar_dataset
from beyond_entropy.infographicvqa_decar_evaluation import (
    _prediction_forbidden_fields,
)
from beyond_entropy.infographicvqa_relative_where import (
    RELATIVE_WHERE_EPOCHS,
    RELATIVE_WHERE_VARIANTS,
    fit_relative_where_oof,
)
from beyond_entropy.qwen_semantic import load_semantic_feature_dataset
from fit_infographicvqa_decar_oof import (
    _checked,
    _fold_maps,
    _image_geometry,
    _read_jsonl_objects,
    _sha256,
    _write_json,
    _write_jsonl,
)


EXPECTED_DECISIONS = 23_946
EXPECTED_SOURCES = 2_204
EXPECTED_IMAGES = 4_406


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "rollouts",
        "answer-nll",
        "features",
        "image-manifest",
        "outer-folds",
        "inner-folds",
        "protocol",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--expected-{name}-sha256", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=RELATIVE_WHERE_EPOCHS)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    input_names = (
        "rollouts",
        "answer_nll",
        "features",
        "image_manifest",
        "outer_folds",
        "inner_folds",
        "protocol",
    )
    paths = {
        name: _checked(
            getattr(args, name), getattr(args, f"expected_{name}_sha256"), name
        )
        for name in input_names
    }
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite relative-where OOF output: {output_dir}"
        )

    records = read_jsonl(paths["rollouts"])
    nll_rows = _read_jsonl_objects(paths["answer_nll"])
    feature_payload = load_semantic_feature_dataset(paths["features"])
    image_rows = _read_jsonl_objects(paths["image_manifest"])
    outer_rows = _read_jsonl_objects(paths["outer_folds"])
    inner_rows = _read_jsonl_objects(paths["inner_folds"])
    outer, inner = _fold_maps(outer_rows, inner_rows)
    dataset = assemble_decar_dataset(
        records,
        nll_rows,
        feature_payload,
        _image_geometry(image_rows),
    )
    expected_inner = {
        (outer_fold, source_id)
        for outer_fold in range(5)
        for source_id in set(dataset.source_ids)
        if outer[source_id] != outer_fold
    }
    if (
        dataset.decisions != EXPECTED_DECISIONS
        or len(set(dataset.source_ids)) != EXPECTED_SOURCES
        or len(set(dataset.image_ids)) != EXPECTED_IMAGES
        or set(inner) != expected_inner
        or any(value not in range(4) for value in inner.values())
    ):
        raise ValueError("relative-where OOF population or fold audit changed")

    start = time.monotonic()
    rows, audit = fit_relative_where_oof(
        dataset,
        outer,
        device=args.device,
        epochs=args.epochs,
    )
    runtime_seconds = time.monotonic() - start
    if len(rows) != EXPECTED_DECISIONS:
        raise RuntimeError("relative-where OOF output coverage changed")
    forbidden = set().union(*(_prediction_forbidden_fields(row) for row in rows))
    if forbidden:
        raise RuntimeError(
            f"relative-where prediction contains forbidden outcomes: {sorted(forbidden)}"
        )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    run = {
        "code_revision": revision,
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "input_hashes_verified_before_fit": True,
        "validation_or_test_inputs_used": False,
        "prediction_outcomes_included": False,
        "scientific_endpoints_computed": False,
        "device": args.device,
        "epochs": args.epochs,
        "runtime_seconds": runtime_seconds,
    }
    audit["run"] = run
    report: dict[str, Any] = {
        "schema": "infographicvqa_relative_where_oof_fit_report_v1",
        "scientific_endpoints_computed": False,
        "scientific_endpoints_read": False,
        "prediction_outcomes_included": False,
        "population": {
            "decisions": dataset.decisions,
            "sources": len(set(dataset.source_ids)),
            "images": len(set(dataset.image_ids)),
        },
        "prediction_metadata": {
            "schema": "infographicvqa_relative_where_oof_predictions_v1",
            "decisions": len(rows),
            "variants": list(RELATIVE_WHERE_VARIANTS),
            "fits": 5 * len(RELATIVE_WHERE_VARIANTS),
            "epochs": args.epochs,
            "device": args.device,
            "outcomes_included": False,
        },
        "run": run,
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=output_dir.name + ".partial-", dir=output_dir.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        prediction_path = temporary / "predictions.jsonl"
        audit_path = temporary / "audit.json"
        report_path = temporary / "report.json"
        _write_jsonl(prediction_path, rows)
        _write_json(audit_path, audit)
        _write_json(report_path, report)
        completion = {
            "schema": "infographicvqa_relative_where_oof_fit_complete_v1",
            "predictions": {
                "path": "predictions.jsonl",
                "sha256": _sha256(prediction_path),
            },
            "audit": {"path": "audit.json", "sha256": _sha256(audit_path)},
            "report": {"path": "report.json", "sha256": _sha256(report_path)},
            "prediction_rows": len(rows),
            "prediction_outcomes_included": False,
            "scientific_endpoints_computed": False,
            "validation_or_test_inputs_used": False,
        }
        _write_json(temporary / "complete.json", completion)
        temporary.replace(output_dir)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
