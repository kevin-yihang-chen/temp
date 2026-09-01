#!/usr/bin/env python3
"""Run the frozen train-only entropy-when / outcome-oracle-where diagnostic."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np  # type: ignore[import-not-found]

from beyond_entropy.dataset import read_jsonl
from beyond_entropy.infographicvqa_decar_evaluation import (
    DECAR_BOOTSTRAP_RESAMPLES,
    DECAR_HYBRID_EVALUATION_SCHEMA,
    build_decar_outcomes,
    evaluate_entropy_oracle_where_factorization,
)
from evaluate_infographicvqa_decar_entropy_where_hybrid import (
    EXPECTED_ACTION_DISAGREEMENTS,
    EXPECTED_DECISIONS,
    EXPECTED_IMAGES,
    EXPECTED_SOURCES,
    _checked,
    _read_json,
    _read_jsonl_objects,
    _sha256,
    _verify_pre_outcome_contract,
    _write_json,
)


def _verify_hybrid_contract(paths: Mapping[str, Path]) -> dict[str, Any]:
    hybrid_complete = _read_json(paths["hybrid_complete"])
    hybrid_decision = _read_json(paths["hybrid_decision"])
    hybrid_evaluation = _read_json(paths["hybrid_evaluation"])
    if (
        hybrid_complete.get("schema")
        != "infographicvqa_decar_entropy_where_hybrid_complete_v1"
        or hybrid_complete.get("decision") != "hybrid_train_not_supported"
        or hybrid_complete.get("validation_or_test_inputs_used") is not False
        or hybrid_complete.get("formal_bootstrap_reused") is not True
        or hybrid_complete.get("evaluation", {}).get("sha256")
        != _sha256(paths["hybrid_evaluation"])
        or hybrid_complete.get("decision_file", {}).get("sha256")
        != _sha256(paths["hybrid_decision"])
    ):
        raise ValueError("oracle-where hybrid completion contract failed")
    if (
        hybrid_decision.get("schema")
        != "infographicvqa_decar_entropy_where_hybrid_decision_v1"
        or hybrid_decision.get("decision") != "hybrid_train_not_supported"
        or hybrid_decision.get("selected_operating_point") is not None
        or hybrid_decision.get("validation_opened") is not False
        or hybrid_decision.get("test_opened") is not False
        or hybrid_decision.get("evaluation_sha256")
        != _sha256(paths["hybrid_evaluation"])
    ):
        raise ValueError("oracle-where hybrid decision contract failed")
    population = hybrid_evaluation.get("population")
    contract_audit = hybrid_evaluation.get("contract_audit")
    action_audit = hybrid_evaluation.get("action_family_audit")
    bootstrap = hybrid_evaluation.get("bootstrap")
    if (
        hybrid_evaluation.get("schema") != DECAR_HYBRID_EVALUATION_SCHEMA
        or hybrid_evaluation.get("decision") != "hybrid_train_not_supported"
        or hybrid_evaluation.get("selected_operating_point") is not None
        or hybrid_evaluation.get("validation_or_test_inputs_used") is not False
        or not isinstance(population, Mapping)
        or population.get("decisions") != EXPECTED_DECISIONS
        or population.get("sources") != EXPECTED_SOURCES
        or population.get("images") != EXPECTED_IMAGES
        or not isinstance(contract_audit, Mapping)
        or contract_audit.get("passed") is not True
        or contract_audit.get("prediction_outcomes_included") is not False
        or contract_audit.get("validation_or_test_inputs_used") is not False
        or not isinstance(action_audit, Mapping)
        or action_audit.get("passed") is not True
        or action_audit.get("disagreements_from_decar") != EXPECTED_ACTION_DISAGREEMENTS
        or not isinstance(bootstrap, Mapping)
        or bootstrap.get("n_resamples") != DECAR_BOOTSTRAP_RESAMPLES
        or bootstrap.get("n_sources") != EXPECTED_SOURCES
        or bootstrap.get("same_indices_for_all_policies_and_differences") is not True
        or bootstrap.get("indices", {}).get("sha256")
        != _sha256(paths["bootstrap_indices"])
        or bootstrap.get("source_order", {}).get("sha256")
        != _sha256(paths["bootstrap_sources"])
    ):
        raise ValueError("oracle-where hybrid evaluation contract failed")
    return {
        "passed": True,
        "hybrid_decision": "hybrid_train_not_supported",
        "hybrid_population": dict(population),
        "hybrid_action_family_audit": dict(action_audit),
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
        "hybrid-evaluation",
        "hybrid-decision",
        "hybrid-complete",
        "hybrid-result",
        "hybrid-freeze",
        "protocol",
        "factorization-freeze",
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
        "hybrid_evaluation",
        "hybrid_decision",
        "hybrid_complete",
        "hybrid_result",
        "hybrid_freeze",
        "protocol",
        "factorization_freeze",
    )
    paths = {
        name: _checked(
            getattr(args, name), getattr(args, f"expected_{name}_sha256"), name
        )
        for name in input_names
    }
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite oracle-where: {output_dir}")

    pre_outcome_audit = _verify_pre_outcome_contract(paths)
    hybrid_audit = _verify_hybrid_contract(paths)
    records = read_jsonl(paths["rollouts"])
    outcomes = build_decar_outcomes(
        records,
        expected_decisions=EXPECTED_DECISIONS,
        expected_sources=EXPECTED_SOURCES,
    )
    sources = sorted({row.source_id for row in outcomes.values()})
    if (
        sources != pre_outcome_audit["bootstrap_sources"]
        or len({row.image_id for row in outcomes.values()}) != EXPECTED_IMAGES
    ):
        raise ValueError("oracle-where outcome population changed")
    prediction_rows = _read_jsonl_objects(paths["predictions"])
    hybrid_evaluation = _read_json(paths["hybrid_evaluation"])
    bootstrap_indices = np.load(paths["bootstrap_indices"], mmap_mode="r")
    if (
        bootstrap_indices.shape != (DECAR_BOOTSTRAP_RESAMPLES, EXPECTED_SOURCES)
        or bootstrap_indices.dtype != np.int32
        or int(bootstrap_indices.min()) != 0
        or int(bootstrap_indices.max()) != EXPECTED_SOURCES - 1
    ):
        raise ValueError("oracle-where bootstrap matrix contract failed")
    evaluation = evaluate_entropy_oracle_where_factorization(
        records,
        prediction_rows,
        hybrid_evaluation,
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
        "passed": True,
        "pre_outcome": {
            key: value
            for key, value in pre_outcome_audit.items()
            if key != "bootstrap_sources"
        },
        "hybrid": hybrid_audit,
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
                "schema": "infographicvqa_entropy_oracle_where_decision_v1",
                "decision": evaluation["decision"],
                "selected_operating_point": evaluation["selected_operating_point"],
                "outcome_oracle_used": True,
                "deployable_method_evidence": False,
                "validation_opened": False,
                "test_opened": False,
                "evaluation_sha256": _sha256(evaluation_path),
            },
        )
        completion = {
            "schema": "infographicvqa_entropy_oracle_where_complete_v1",
            "decision": evaluation["decision"],
            "evaluation": {
                "path": "evaluation.json",
                "sha256": _sha256(evaluation_path),
            },
            "decision_file": {
                "path": "decision.json",
                "sha256": _sha256(decision_path),
            },
            "outcome_oracle_used": True,
            "deployable_method_evidence": False,
            "formal_bootstrap_reused": True,
            "validation_or_test_inputs_used": False,
        }
        _write_json(temporary / "complete.json", completion)
        temporary.replace(output_dir)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
