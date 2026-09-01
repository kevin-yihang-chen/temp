#!/usr/bin/env python3
"""Evaluate frozen InfographicVQA DECAR OOF predictions and advancement gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np  # type: ignore[import-not-found]

from beyond_entropy.dataset import read_jsonl
from beyond_entropy.infographicvqa_decar_evaluation import (
    DECAR_BOOTSTRAP_RESAMPLES,
    DECAR_BOOTSTRAP_SEED,
    build_decar_outcomes,
    evaluate_decar_oof,
)


EXPECTED_DECISIONS = 23_946
EXPECTED_SOURCES = 2_204
EXPECTED_IMAGES = 4_406


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked(path: Path, expected: str, name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or _sha256(resolved) != expected:
        raise ValueError(f"InfographicVQA DECAR evaluation {name} SHA-256 mismatch")
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _assert_zero_source_overlap(value: object) -> int:
    checked = 0
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "source_overlap":
                if int(child) != 0:
                    raise ValueError("DECAR OOF audit contains source overlap")
                checked += 1
            checked += _assert_zero_source_overlap(child)
    elif isinstance(value, list):
        for child in value:
            checked += _assert_zero_source_overlap(child)
    return checked


def _verify_oof_contract(
    complete: Mapping[str, Any],
    audit: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    predictions: Path,
    audit_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    if complete.get("schema") != "infographicvqa_decar_oof_fit_complete_v1":
        raise ValueError("DECAR OOF completion schema changed")
    if (
        complete.get("prediction_rows") != EXPECTED_DECISIONS
        or complete.get("prediction_outcomes_included") is not False
        or complete.get("predictions", {}).get("sha256") != _sha256(predictions)
        or complete.get("audit", {}).get("sha256") != _sha256(audit_path)
        or complete.get("report", {}).get("sha256") != _sha256(report_path)
    ):
        raise ValueError("DECAR OOF completion contract failed")
    population = report.get("population")
    if not isinstance(population, Mapping) or (
        population.get("decisions") != EXPECTED_DECISIONS
        or population.get("sources") != EXPECTED_SOURCES
        or population.get("images") != EXPECTED_IMAGES
        or report.get("scientific_endpoints_computed") is not False
        or report.get("scientific_endpoints_read") is not False
        or report.get("prediction_outcomes_included") is not False
    ):
        raise ValueError("DECAR OOF fit report contract failed")
    if (
        audit.get("schema") != "infographicvqa_decar_nested_oof_audit_v1"
        or audit.get("prediction_rows") != EXPECTED_DECISIONS
        or audit.get("prediction_outcomes_included") is not False
    ):
        raise ValueError("DECAR nested OOF audit contract failed")
    overlap_checks = _assert_zero_source_overlap(audit)
    if overlap_checks < 25:
        raise ValueError("DECAR nested OOF source-exclusion audit is incomplete")
    return {
        "passed": True,
        "source_overlap_checks": overlap_checks,
        "prediction_rows": EXPECTED_DECISIONS,
        "prediction_outcomes_included": False,
        "scientific_endpoints_computed_during_fit": False,
    }


def _write_bootstrap_indices(path: Path, source_count: int) -> Any:
    values = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.int32,
        shape=(DECAR_BOOTSTRAP_RESAMPLES, source_count),
    )
    rng = np.random.default_rng(DECAR_BOOTSTRAP_SEED)
    batch_size = 256
    for start in range(0, DECAR_BOOTSTRAP_RESAMPLES, batch_size):
        stop = min(DECAR_BOOTSTRAP_RESAMPLES, start + batch_size)
        values[start:stop] = rng.integers(
            0,
            source_count,
            size=(stop - start, source_count),
            dtype=np.int32,
        )
    values.flush()
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "rollouts",
        "predictions",
        "oof-complete",
        "oof-audit",
        "oof-report",
        "protocol",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--expected-{name}-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    input_names = (
        "rollouts",
        "predictions",
        "oof_complete",
        "oof_audit",
        "oof_report",
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
        raise FileExistsError(f"refusing to overwrite DECAR evaluation: {output_dir}")
    complete = _read_json(paths["oof_complete"])
    audit = _read_json(paths["oof_audit"])
    report = _read_json(paths["oof_report"])
    fit_audit = _verify_oof_contract(
        complete,
        audit,
        report,
        predictions=paths["predictions"],
        audit_path=paths["oof_audit"],
        report_path=paths["oof_report"],
    )
    records = read_jsonl(paths["rollouts"])
    outcomes = build_decar_outcomes(
        records,
        expected_decisions=EXPECTED_DECISIONS,
        expected_sources=EXPECTED_SOURCES,
    )
    sources = sorted({row.source_id for row in outcomes.values()})
    if len({row.image_id for row in outcomes.values()}) != EXPECTED_IMAGES:
        raise ValueError("DECAR evaluation image population changed")
    prediction_rows = _read_jsonl_objects(paths["predictions"])

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=output_dir.name + ".partial-", dir=output_dir.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        source_order_path = temporary / "bootstrap-sources.json"
        indices_path = temporary / "bootstrap-indices.npy"
        _write_json(
            source_order_path,
            {
                "schema": "infographicvqa_decar_bootstrap_sources_v1",
                "sources": sources,
            },
        )
        bootstrap_indices = _write_bootstrap_indices(indices_path, len(sources))
        evaluation = evaluate_decar_oof(
            records,
            prediction_rows,
            bootstrap_indices=bootstrap_indices,
            expected_decisions=EXPECTED_DECISIONS,
            expected_sources=EXPECTED_SOURCES,
        )
        del bootstrap_indices
        evaluation["inputs"] = {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        }
        evaluation["fit_audit"] = fit_audit
        evaluation["bootstrap"]["indices"] = {
            "path": "bootstrap-indices.npy",
            "sha256": _sha256(indices_path),
            "shape": [DECAR_BOOTSTRAP_RESAMPLES, len(sources)],
            "dtype": "int32",
        }
        evaluation["bootstrap"]["source_order"] = {
            "path": "bootstrap-sources.json",
            "sha256": _sha256(source_order_path),
        }
        evaluation_path = temporary / "evaluation.json"
        decision_path = temporary / "decision.json"
        _write_json(evaluation_path, evaluation)
        _write_json(
            decision_path,
            {
                "schema": "infographicvqa_decar_train_advancement_decision_v1",
                "decision": evaluation["decision"],
                "selected_operating_point": evaluation["selected_operating_point"],
                "validation_opened": False,
                "test_opened": False,
                "evaluation_sha256": _sha256(evaluation_path),
            },
        )
        completion = {
            "schema": "infographicvqa_decar_oof_evaluation_complete_v1",
            "decision": evaluation["decision"],
            "evaluation": {
                "path": "evaluation.json",
                "sha256": _sha256(evaluation_path),
            },
            "decision_file": {
                "path": "decision.json",
                "sha256": _sha256(decision_path),
            },
            "bootstrap_indices": {
                "path": "bootstrap-indices.npy",
                "sha256": _sha256(indices_path),
            },
            "bootstrap_sources": {
                "path": "bootstrap-sources.json",
                "sha256": _sha256(source_order_path),
            },
            "validation_or_test_inputs_used": False,
        }
        _write_json(temporary / "complete.json", completion)
        temporary.replace(output_dir)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
