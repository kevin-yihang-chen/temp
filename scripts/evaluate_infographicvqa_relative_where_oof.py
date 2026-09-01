#!/usr/bin/env python3
"""Evaluate the frozen relative-where source-OOF predictions on train only."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np  # type: ignore[import-not-found]

from beyond_entropy.dataset import read_jsonl
from beyond_entropy.infographicvqa_decar_evaluation import DECAR_BOOTSTRAP_RESAMPLES
from beyond_entropy.infographicvqa_relative_where import RELATIVE_WHERE_VARIANTS
from beyond_entropy.infographicvqa_relative_where_evaluation import (
    evaluate_relative_where_oof,
)
from evaluate_infographicvqa_decar_entropy_where_hybrid import (
    EXPECTED_DECISIONS,
    EXPECTED_IMAGES,
    EXPECTED_SOURCES,
    _checked,
    _read_json,
    _sha256,
    _write_json,
)
from fit_infographicvqa_decar_oof import _read_jsonl_objects


def _zero_source_overlaps(value: object) -> int:
    checked = 0
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "source_overlap":
                if int(child) != 0:
                    raise ValueError("relative-where audit contains source overlap")
                checked += 1
            checked += _zero_source_overlaps(child)
    elif isinstance(value, list):
        for child in value:
            checked += _zero_source_overlaps(child)
    return checked


def _verify_pre_outcome_contract(paths: Mapping[str, Path]) -> dict[str, Any]:
    fit_complete = _read_json(paths["relative_complete"])
    fit_audit = _read_json(paths["relative_audit"])
    fit_report = _read_json(paths["relative_report"])
    if (
        fit_complete.get("schema")
        != "infographicvqa_relative_where_oof_fit_complete_v1"
        or fit_complete.get("prediction_rows") != EXPECTED_DECISIONS
        or fit_complete.get("prediction_outcomes_included") is not False
        or fit_complete.get("scientific_endpoints_computed") is not False
        or fit_complete.get("validation_or_test_inputs_used") is not False
        or fit_complete.get("predictions", {}).get("sha256")
        != _sha256(paths["relative_predictions"])
        or fit_complete.get("audit", {}).get("sha256")
        != _sha256(paths["relative_audit"])
        or fit_complete.get("report", {}).get("sha256")
        != _sha256(paths["relative_report"])
    ):
        raise ValueError("relative-where fit completion contract failed")
    population = fit_report.get("population")
    metadata = fit_report.get("prediction_metadata")
    overlap_checks = _zero_source_overlaps(fit_audit)
    if (
        fit_audit.get("schema") != "infographicvqa_relative_where_nested_oof_audit_v1"
        or fit_audit.get("prediction_rows") != EXPECTED_DECISIONS
        or fit_audit.get("prediction_outcomes_included") is not False
        or fit_audit.get("fits") != 20
        or tuple(fit_audit.get("variants", ())) != RELATIVE_WHERE_VARIANTS
        or overlap_checks != 5
        or fit_report.get("schema") != "infographicvqa_relative_where_oof_fit_report_v1"
        or fit_report.get("scientific_endpoints_computed") is not False
        or fit_report.get("scientific_endpoints_read") is not False
        or fit_report.get("prediction_outcomes_included") is not False
        or not isinstance(population, Mapping)
        or population.get("decisions") != EXPECTED_DECISIONS
        or population.get("sources") != EXPECTED_SOURCES
        or population.get("images") != EXPECTED_IMAGES
        or not isinstance(metadata, Mapping)
        or metadata.get("fits") != 20
        or tuple(metadata.get("variants", ())) != RELATIVE_WHERE_VARIANTS
        or metadata.get("outcomes_included") is not False
    ):
        raise ValueError("relative-where fit audit/report contract failed")

    decar_complete = _read_json(paths["decar_complete"])
    decar_audit = _read_json(paths["decar_audit"])
    decar_report = _read_json(paths["decar_report"])
    if (
        decar_complete.get("schema") != "infographicvqa_decar_oof_fit_complete_v1"
        or decar_complete.get("prediction_rows") != EXPECTED_DECISIONS
        or decar_complete.get("prediction_outcomes_included") is not False
        or decar_complete.get("predictions", {}).get("sha256")
        != _sha256(paths["decar_predictions"])
        or decar_complete.get("audit", {}).get("sha256")
        != _sha256(paths["decar_audit"])
        or decar_complete.get("report", {}).get("sha256")
        != _sha256(paths["decar_report"])
        or decar_audit.get("prediction_rows") != EXPECTED_DECISIONS
        or decar_audit.get("prediction_outcomes_included") is not False
        or decar_report.get("scientific_endpoints_computed") is not False
        or decar_report.get("prediction_outcomes_included") is not False
    ):
        raise ValueError("relative-where frozen DECAR OOF contract failed")

    hybrid_complete = _read_json(paths["hybrid_complete"])
    hybrid_evaluation = _read_json(paths["hybrid_evaluation"])
    oracle_complete = _read_json(paths["oracle_complete"])
    oracle_evaluation = _read_json(paths["oracle_evaluation"])
    if (
        hybrid_complete.get("decision") != "hybrid_train_not_supported"
        or hybrid_complete.get("validation_or_test_inputs_used") is not False
        or hybrid_complete.get("evaluation", {}).get("sha256")
        != _sha256(paths["hybrid_evaluation"])
        or hybrid_evaluation.get("validation_or_test_inputs_used") is not False
        or oracle_complete.get("decision") != "where_bottleneck_supported"
        or oracle_complete.get("validation_or_test_inputs_used") is not False
        or oracle_complete.get("deployable_method_evidence") is not False
        or oracle_complete.get("evaluation", {}).get("sha256")
        != _sha256(paths["oracle_evaluation"])
        or oracle_evaluation.get("validation_or_test_inputs_used") is not False
        or oracle_evaluation.get("deployable_method_evidence") is not False
    ):
        raise ValueError("relative-where frozen diagnostic contract failed")

    source_payload = _read_json(paths["bootstrap_sources"])
    sources = source_payload.get("sources")
    if (
        source_payload.get("schema") != "infographicvqa_decar_bootstrap_sources_v1"
        or not isinstance(sources, list)
        or len(sources) != EXPECTED_SOURCES
        or len(set(sources)) != EXPECTED_SOURCES
        or sources != sorted(sources)
    ):
        raise ValueError("relative-where bootstrap source-order contract failed")
    return {
        "passed": True,
        "relative_prediction_rows": EXPECTED_DECISIONS,
        "relative_prediction_outcomes_included": False,
        "relative_outer_source_overlap_checks": overlap_checks,
        "decar_prediction_outcomes_included": False,
        "hybrid_decision": "hybrid_train_not_supported",
        "oracle_decision": "where_bottleneck_supported",
        "bootstrap_sources": sources,
        "validation_or_test_inputs_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "rollouts",
        "answer-nll",
        "relative-predictions",
        "relative-complete",
        "relative-audit",
        "relative-report",
        "decar-predictions",
        "decar-complete",
        "decar-audit",
        "decar-report",
        "hybrid-evaluation",
        "hybrid-complete",
        "oracle-evaluation",
        "oracle-complete",
        "bootstrap-indices",
        "bootstrap-sources",
        "protocol",
        "design-audit",
        "oracle-result",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--expected-{name}-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    input_names = (
        "rollouts",
        "answer_nll",
        "relative_predictions",
        "relative_complete",
        "relative_audit",
        "relative_report",
        "decar_predictions",
        "decar_complete",
        "decar_audit",
        "decar_report",
        "hybrid_evaluation",
        "hybrid_complete",
        "oracle_evaluation",
        "oracle_complete",
        "bootstrap_indices",
        "bootstrap_sources",
        "protocol",
        "design_audit",
        "oracle_result",
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
            f"refusing to overwrite relative-where evaluation: {output_dir}"
        )

    contract_audit = _verify_pre_outcome_contract(paths)
    records = read_jsonl(paths["rollouts"])
    relative_rows = _read_jsonl_objects(paths["relative_predictions"])
    decar_rows = _read_jsonl_objects(paths["decar_predictions"])
    nll_rows = _read_jsonl_objects(paths["answer_nll"])
    hybrid_evaluation = _read_json(paths["hybrid_evaluation"])
    oracle_evaluation = _read_json(paths["oracle_evaluation"])
    bootstrap_indices = np.load(paths["bootstrap_indices"], mmap_mode="r")
    if (
        bootstrap_indices.shape != (DECAR_BOOTSTRAP_RESAMPLES, EXPECTED_SOURCES)
        or bootstrap_indices.dtype != np.int32
        or int(bootstrap_indices.min()) != 0
        or int(bootstrap_indices.max()) != EXPECTED_SOURCES - 1
    ):
        raise ValueError("relative-where bootstrap matrix contract failed")
    evaluation = evaluate_relative_where_oof(
        records,
        relative_rows,
        decar_rows,
        nll_rows,
        hybrid_evaluation,
        oracle_evaluation,
        bootstrap_indices=bootstrap_indices,
        expected_decisions=EXPECTED_DECISIONS,
        expected_sources=EXPECTED_SOURCES,
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
                "schema": "infographicvqa_relative_where_oof_decision_v1",
                "decision": evaluation["decision"],
                "selected_operating_point": evaluation["selected_operating_point"],
                "validation_opened": False,
                "test_opened": False,
                "evaluation_sha256": _sha256(evaluation_path),
            },
        )
        completion = {
            "schema": "infographicvqa_relative_where_oof_evaluation_complete_v1",
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
            "relative_prediction_outcomes_included": False,
            "privileged_teacher_used_only_in_evaluation": True,
            "validation_or_test_inputs_used": False,
        }
        _write_json(temporary / "complete.json", completion)
        temporary.replace(output_dir)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
