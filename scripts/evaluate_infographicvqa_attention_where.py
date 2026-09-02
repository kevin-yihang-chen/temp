#!/usr/bin/env python3
"""Evaluate the frozen raw-attention where policy on InfographicVQA train only."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np  # type: ignore[import-not-found]

from beyond_entropy.dataset import read_jsonl
from beyond_entropy.infographicvqa_attention_where import ATTENTION_WHERE_SCHEMA
from beyond_entropy.infographicvqa_attention_where_evaluation import (
    ATTENTION_WHERE_EVALUATION_SCHEMA,
    evaluate_attention_where,
)
from beyond_entropy.infographicvqa_decar_evaluation import DECAR_BOOTSTRAP_RESAMPLES
from beyond_entropy.qwen_semantic import load_semantic_feature_dataset
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


MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
SOURCE_FEATURES_SHA256 = (
    "d0508726a50b4c1e54778392d08329b242a680fc13292cac1ebec8b42a175300"
)


def _verify_contract(paths: Mapping[str, Path]) -> dict[str, Any]:
    feature_complete = _read_json(paths["attention_complete"])
    feature_audit = _read_json(paths["attention_audit"])
    merge_report = _read_json(paths["attention_merge_report"])
    if (
        feature_complete.get("schema")
        != "infographicvqa_attention_where_feature_complete_v1"
        or feature_complete.get("passed") is not True
        or feature_complete.get("decisions") != EXPECTED_DECISIONS
        or feature_complete.get("sources") != EXPECTED_SOURCES
        or feature_complete.get("images") != EXPECTED_IMAGES
        or feature_complete.get("validation_or_test_inputs_used") is not False
        or feature_complete.get("outcomes_included") is not False
        or feature_complete.get("merged_features_sha256")
        != _sha256(paths["attention_features"])
        or feature_complete.get("audit_sha256")
        != _sha256(paths["attention_audit"])
        or feature_audit.get("schema") != ATTENTION_WHERE_SCHEMA
        or feature_audit.get("passed") is not True
        or feature_audit.get("population")
        != {
            "decisions": EXPECTED_DECISIONS,
            "sources": EXPECTED_SOURCES,
            "images": EXPECTED_IMAGES,
        }
        or feature_audit.get("outcomes_included") is not False
        or feature_audit.get("candidate_actions_executed") is not False
        or feature_audit.get("validation_or_test_inputs_used") is not False
        or merge_report.get("passed") is not True
        or merge_report.get("stage") != "attention"
        or merge_report.get("decisions") != EXPECTED_DECISIONS
        or merge_report.get("sources") != EXPECTED_SOURCES
        or merge_report.get("source_disjoint") is not True
        or merge_report.get("outcomes_included") is not False
        or merge_report.get("output_sha256")
        != _sha256(paths["attention_features"])
    ):
        raise ValueError("attention-where feature completion contract failed")

    hybrid_complete = _read_json(paths["hybrid_complete"])
    oracle_complete = _read_json(paths["oracle_complete"])
    relative_complete = _read_json(paths["relative_complete"])
    hybrid_evaluation = _read_json(paths["hybrid_evaluation"])
    oracle_evaluation = _read_json(paths["oracle_evaluation"])
    relative_evaluation = _read_json(paths["relative_evaluation"])
    if (
        hybrid_complete.get("decision") != "hybrid_train_not_supported"
        or hybrid_complete.get("evaluation", {}).get("sha256")
        != _sha256(paths["hybrid_evaluation"])
        or oracle_complete.get("decision") != "where_bottleneck_supported"
        or oracle_complete.get("evaluation", {}).get("sha256")
        != _sha256(paths["oracle_evaluation"])
        or relative_complete.get("decision")
        != "relative_where_train_not_supported"
        or relative_complete.get("evaluation", {}).get("sha256")
        != _sha256(paths["relative_evaluation"])
        or hybrid_evaluation.get("validation_or_test_inputs_used") is not False
        or oracle_evaluation.get("validation_or_test_inputs_used") is not False
        or relative_evaluation.get("validation_or_test_inputs_used") is not False
    ):
        raise ValueError("attention-where frozen comparator contract failed")

    source_payload = _read_json(paths["bootstrap_sources"])
    sources = source_payload.get("sources")
    if (
        source_payload.get("schema")
        != "infographicvqa_decar_bootstrap_sources_v1"
        or not isinstance(sources, list)
        or len(sources) != EXPECTED_SOURCES
        or len(set(sources)) != EXPECTED_SOURCES
        or sources != sorted(sources)
    ):
        raise ValueError("attention-where bootstrap source-order contract failed")
    return {
        "passed": True,
        "attention_features_outcomes_included": False,
        "candidate_actions_executed": False,
        "frozen_comparator_evaluations": 3,
        "validation_or_test_inputs_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    input_names = (
        "attention-features",
        "attention-complete",
        "attention-audit",
        "attention-merge-report",
        "rollouts",
        "answer-nll",
        "decar-predictions",
        "relative-predictions",
        "hybrid-evaluation",
        "hybrid-complete",
        "oracle-evaluation",
        "oracle-complete",
        "relative-evaluation",
        "relative-complete",
        "bootstrap-indices",
        "bootstrap-sources",
        "protocol",
        "resource-amendment",
        "action-diagnostic",
    )
    for name in input_names:
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--expected-{name}-sha256", required=True)
    parser.add_argument("--expected-attention-code-revision", required=True)
    parser.add_argument("--expected-model-revision", default=MODEL_REVISION)
    parser.add_argument(
        "--expected-source-features-sha256", default=SOURCE_FEATURES_SHA256
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    normalized_names = tuple(name.replace("-", "_") for name in input_names)
    paths = {
        name: _checked(
            getattr(args, name), getattr(args, f"expected_{name}_sha256"), name
        )
        for name in normalized_names
    }
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite attention-where evaluation: {output_dir}"
        )

    contract_audit = _verify_contract(paths)
    bootstrap_indices = np.load(paths["bootstrap_indices"], mmap_mode="r")
    if (
        bootstrap_indices.shape != (DECAR_BOOTSTRAP_RESAMPLES, EXPECTED_SOURCES)
        or bootstrap_indices.dtype != np.int32
        or int(bootstrap_indices.min()) != 0
        or int(bootstrap_indices.max()) != EXPECTED_SOURCES - 1
    ):
        raise ValueError("attention-where bootstrap matrix contract failed")
    evaluation = evaluate_attention_where(
        read_jsonl(paths["rollouts"]),
        load_semantic_feature_dataset(paths["attention_features"]),
        _read_jsonl_objects(paths["decar_predictions"]),
        _read_jsonl_objects(paths["relative_predictions"]),
        _read_jsonl_objects(paths["answer_nll"]),
        _read_json(paths["hybrid_evaluation"]),
        _read_json(paths["oracle_evaluation"]),
        _read_json(paths["relative_evaluation"]),
        expected_attention_code_revision=args.expected_attention_code_revision,
        expected_model_revision=args.expected_model_revision,
        expected_source_features_sha256=args.expected_source_features_sha256,
        expected_rollouts_sha256=_sha256(paths["rollouts"]),
        bootstrap_indices=bootstrap_indices,
        expected_decisions=EXPECTED_DECISIONS,
        expected_sources=EXPECTED_SOURCES,
    )
    del bootstrap_indices
    if evaluation.get("schema") != ATTENTION_WHERE_EVALUATION_SCHEMA:
        raise RuntimeError("attention-where evaluation schema changed")
    evaluation["inputs"] = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in paths.items()
    }
    evaluation["contract_audit"] = contract_audit
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
                "schema": "infographicvqa_attention_where_decision_v1",
                "decision": evaluation["decision"],
                "selected_operating_point": evaluation[
                    "selected_operating_point"
                ],
                "validation_opened": False,
                "test_opened": False,
                "evaluation_sha256": _sha256(evaluation_path),
            },
        )
        completion = {
            "schema": "infographicvqa_attention_where_evaluation_complete_v1",
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
            "attention_features_outcomes_included": False,
            "privileged_teacher_used_only_in_evaluation": True,
            "validation_or_test_inputs_used": False,
        }
        _write_json(temporary / "complete.json", completion)
        temporary.replace(output_dir)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
