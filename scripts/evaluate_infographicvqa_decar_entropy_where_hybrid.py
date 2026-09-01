#!/usr/bin/env python3
"""Evaluate the frozen train-only entropy-when / OOF-where hybrid."""

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
    build_decar_outcomes,
    evaluate_entropy_where_hybrid,
)


EXPECTED_DECISIONS = 23_946
EXPECTED_SOURCES = 2_204
EXPECTED_IMAGES = 4_406
EXPECTED_ACTION_DISAGREEMENTS = {
    "loss_only": 0,
    "no_harm_head": 0,
    "task_value_only": 17_446,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked(path: Path, expected: str, name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or _sha256(resolved) != expected:
        raise ValueError(f"InfographicVQA DECAR hybrid {name} SHA-256 mismatch")
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
                    raise ValueError("DECAR hybrid OOF audit contains source overlap")
                checked += 1
            checked += _assert_zero_source_overlap(child)
    elif isinstance(value, list):
        for child in value:
            checked += _assert_zero_source_overlap(child)
    return checked


def _verify_pre_outcome_contract(paths: Mapping[str, Path]) -> dict[str, Any]:
    oof_complete = _read_json(paths["oof_complete"])
    oof_audit = _read_json(paths["oof_audit"])
    oof_report = _read_json(paths["oof_report"])
    if (
        oof_complete.get("schema") != "infographicvqa_decar_oof_fit_complete_v1"
        or oof_complete.get("prediction_rows") != EXPECTED_DECISIONS
        or oof_complete.get("prediction_outcomes_included") is not False
        or oof_complete.get("predictions", {}).get("sha256")
        != _sha256(paths["predictions"])
        or oof_complete.get("audit", {}).get("sha256") != _sha256(paths["oof_audit"])
        or oof_complete.get("report", {}).get("sha256") != _sha256(paths["oof_report"])
    ):
        raise ValueError("DECAR hybrid OOF completion contract failed")
    population = oof_report.get("population")
    if not isinstance(population, Mapping) or (
        population.get("decisions") != EXPECTED_DECISIONS
        or population.get("sources") != EXPECTED_SOURCES
        or population.get("images") != EXPECTED_IMAGES
        or oof_report.get("scientific_endpoints_computed") is not False
        or oof_report.get("scientific_endpoints_read") is not False
        or oof_report.get("prediction_outcomes_included") is not False
    ):
        raise ValueError("DECAR hybrid OOF report contract failed")
    overlap_checks = _assert_zero_source_overlap(oof_audit)
    if (
        oof_audit.get("schema") != "infographicvqa_decar_nested_oof_audit_v1"
        or oof_audit.get("prediction_rows") != EXPECTED_DECISIONS
        or oof_audit.get("prediction_outcomes_included") is not False
        or overlap_checks < 25
    ):
        raise ValueError("DECAR hybrid OOF audit contract failed")

    formal_complete = _read_json(paths["formal_complete"])
    formal_evaluation = _read_json(paths["formal_evaluation"])
    if (
        formal_complete.get("schema")
        != "infographicvqa_decar_oof_evaluation_complete_v1"
        or formal_complete.get("decision") != "decar_not_advanced"
        or formal_complete.get("validation_or_test_inputs_used") is not False
        or formal_complete.get("evaluation", {}).get("sha256")
        != _sha256(paths["formal_evaluation"])
        or formal_complete.get("bootstrap_indices", {}).get("sha256")
        != _sha256(paths["bootstrap_indices"])
        or formal_complete.get("bootstrap_sources", {}).get("sha256")
        != _sha256(paths["bootstrap_sources"])
        or formal_evaluation.get("decision") != "decar_not_advanced"
        or formal_evaluation.get("selected_operating_point") is not None
        or formal_evaluation.get("validation_or_test_inputs_used") is not False
        or formal_evaluation.get("all_prediction_rows_outcome_free") is not True
        or formal_evaluation.get("fit_audit", {}).get("passed") is not True
    ):
        raise ValueError("DECAR hybrid formal-evaluation contract failed")

    source_payload = _read_json(paths["bootstrap_sources"])
    sources = source_payload.get("sources")
    if (
        source_payload.get("schema") != "infographicvqa_decar_bootstrap_sources_v1"
        or not isinstance(sources, list)
        or len(sources) != EXPECTED_SOURCES
        or len(set(sources)) != EXPECTED_SOURCES
        or sources != sorted(sources)
    ):
        raise ValueError("DECAR hybrid bootstrap source-order contract failed")
    return {
        "passed": True,
        "source_overlap_checks": overlap_checks,
        "formal_decision": "decar_not_advanced",
        "prediction_rows": EXPECTED_DECISIONS,
        "prediction_outcomes_included": False,
        "bootstrap_sources": sources,
        "validation_or_test_inputs_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "rollouts",
        "predictions",
        "oof-complete",
        "oof-audit",
        "oof-report",
        "formal-evaluation",
        "formal-complete",
        "bootstrap-indices",
        "bootstrap-sources",
        "formal-result",
        "protocol",
        "freeze",
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
        "formal_evaluation",
        "formal_complete",
        "bootstrap_indices",
        "bootstrap_sources",
        "formal_result",
        "protocol",
        "freeze",
    )
    paths = {
        name: _checked(
            getattr(args, name), getattr(args, f"expected_{name}_sha256"), name
        )
        for name in input_names
    }
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite DECAR hybrid: {output_dir}")

    contract_audit = _verify_pre_outcome_contract(paths)
    records = read_jsonl(paths["rollouts"])
    outcomes = build_decar_outcomes(
        records,
        expected_decisions=EXPECTED_DECISIONS,
        expected_sources=EXPECTED_SOURCES,
    )
    sources = sorted({row.source_id for row in outcomes.values()})
    if (
        sources != contract_audit["bootstrap_sources"]
        or len({row.image_id for row in outcomes.values()}) != EXPECTED_IMAGES
    ):
        raise ValueError("DECAR hybrid outcome population changed")
    prediction_rows = _read_jsonl_objects(paths["predictions"])
    formal_evaluation = _read_json(paths["formal_evaluation"])
    bootstrap_indices = np.load(paths["bootstrap_indices"], mmap_mode="r")
    if (
        bootstrap_indices.shape != (DECAR_BOOTSTRAP_RESAMPLES, EXPECTED_SOURCES)
        or bootstrap_indices.dtype != np.int32
        or int(bootstrap_indices.min()) != 0
        or int(bootstrap_indices.max()) != EXPECTED_SOURCES - 1
    ):
        raise ValueError("DECAR hybrid bootstrap matrix contract failed")
    evaluation = evaluate_entropy_where_hybrid(
        records,
        prediction_rows,
        formal_evaluation,
        bootstrap_indices=bootstrap_indices,
        expected_decisions=EXPECTED_DECISIONS,
        expected_sources=EXPECTED_SOURCES,
        expected_action_disagreements=EXPECTED_ACTION_DISAGREEMENTS,
    )
    del bootstrap_indices
    evaluation["inputs"] = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in paths.items()
    }
    evaluation["contract_audit"] = {
        key: value
        for key, value in contract_audit.items()
        if key != "bootstrap_sources"
    }
    evaluation["bootstrap"]["indices"] = {
        "path": str(paths["bootstrap_indices"]),
        "sha256": _sha256(paths["bootstrap_indices"]),
        "shape": [DECAR_BOOTSTRAP_RESAMPLES, EXPECTED_SOURCES],
        "dtype": "int32",
        "reused_from_formal_evaluation": True,
    }
    evaluation["bootstrap"]["source_order"] = {
        "path": str(paths["bootstrap_sources"]),
        "sha256": _sha256(paths["bootstrap_sources"]),
        "reused_from_formal_evaluation": True,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=output_dir.name + ".partial-", dir=output_dir.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        evaluation_path = temporary / "evaluation.json"
        decision_path = temporary / "decision.json"
        _write_json(evaluation_path, evaluation)
        _write_json(
            decision_path,
            {
                "schema": "infographicvqa_decar_entropy_where_hybrid_decision_v1",
                "decision": evaluation["decision"],
                "selected_operating_point": evaluation["selected_operating_point"],
                "validation_opened": False,
                "test_opened": False,
                "evaluation_sha256": _sha256(evaluation_path),
            },
        )
        completion = {
            "schema": "infographicvqa_decar_entropy_where_hybrid_complete_v1",
            "decision": evaluation["decision"],
            "evaluation": {
                "path": "evaluation.json",
                "sha256": _sha256(evaluation_path),
            },
            "decision_file": {
                "path": "decision.json",
                "sha256": _sha256(decision_path),
            },
            "formal_bootstrap_reused": True,
            "validation_or_test_inputs_used": False,
        }
        _write_json(temporary / "complete.json", completion)
        temporary.replace(output_dir)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
