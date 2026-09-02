#!/usr/bin/env python3
"""Evaluate the frozen literature-attention where policies on train only."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np  # type: ignore[import-not-found]
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

from beyond_entropy.dataset import read_jsonl
from beyond_entropy.infographicvqa_decar_evaluation import DECAR_BOOTSTRAP_RESAMPLES
from beyond_entropy.infographicvqa_literature_attention_evaluation import (
    LITERATURE_ATTENTION_EVALUATION_SCHEMA,
    evaluate_literature_attention_where,
)
from beyond_entropy.infographicvqa_literature_attention_where import (
    LITERATURE_ATTENTION_AUDIT_SCHEMA,
)
from beyond_entropy.qwen_semantic import load_semantic_feature_dataset

MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
SOURCE_FEATURES_SHA256 = (
    "d0508726a50b4c1e54778392d08329b242a680fc13292cac1ebec8b42a175300"
)


def _verify_contract(paths: Mapping[str, Path]) -> dict[str, Any]:
    literature_complete = _read_json(paths["literature_complete"])
    literature_audit = _read_json(paths["literature_audit"])
    literature_merge = _read_json(paths["literature_merge_report"])
    if (
        literature_complete.get("schema")
        != "infographicvqa_literature_attention_where_feature_complete_v1"
        or literature_complete.get("passed") is not True
        or literature_complete.get("decisions") != EXPECTED_DECISIONS
        or literature_complete.get("sources") != EXPECTED_SOURCES
        or literature_complete.get("images") != EXPECTED_IMAGES
        or literature_complete.get("validation_or_test_inputs_used") is not False
        or literature_complete.get("outcomes_included") is not False
        or literature_complete.get("merged_features_sha256")
        != _sha256(paths["literature_features"])
        or literature_complete.get("audit_sha256") != _sha256(paths["literature_audit"])
        or literature_audit.get("schema") != LITERATURE_ATTENTION_AUDIT_SCHEMA
        or literature_audit.get("passed") is not True
        or literature_audit.get("population")
        != {
            "decisions": EXPECTED_DECISIONS,
            "sources": EXPECTED_SOURCES,
            "images": EXPECTED_IMAGES,
        }
        or literature_audit.get("outcomes_included") is not False
        or literature_audit.get("candidate_actions_executed") is not False
        or literature_audit.get("validation_or_test_inputs_used") is not False
        or literature_merge.get("schema")
        != "infographicvqa_literature_attention_where_merge_v1"
        or literature_merge.get("passed") is not True
        or literature_merge.get("decisions") != EXPECTED_DECISIONS
        or literature_merge.get("sources") != EXPECTED_SOURCES
        or literature_merge.get("source_disjoint") is not True
        or literature_merge.get("outcomes_included") is not False
        or literature_merge.get("validation_or_test_inputs_used") is not False
        or literature_merge.get("output_sha256")
        != _sha256(paths["literature_features"])
    ):
        raise ValueError("literature-attention feature completion contract failed")

    raw_complete = _read_json(paths["raw_attention_complete"])
    raw_audit = _read_json(paths["raw_attention_audit"])
    raw_evaluation = _read_json(paths["raw_attention_evaluation"])
    raw_evaluation_complete = _read_json(paths["raw_attention_evaluation_complete"])
    if (
        raw_complete.get("schema")
        != "infographicvqa_attention_where_feature_complete_v1"
        or raw_complete.get("passed") is not True
        or raw_complete.get("merged_features_sha256")
        != _sha256(paths["raw_attention_features"])
        or raw_complete.get("audit_sha256") != _sha256(paths["raw_attention_audit"])
        or raw_audit.get("passed") is not True
        or raw_audit.get("outcomes_included") is not False
        or raw_evaluation.get("validation_or_test_inputs_used") is not False
        or raw_evaluation_complete.get("decision")
        not in (
            "attention_where_train_supported",
            "attention_where_train_not_supported",
        )
        or raw_evaluation_complete.get("evaluation", {}).get("sha256")
        != _sha256(paths["raw_attention_evaluation"])
    ):
        raise ValueError("literature-attention raw comparator contract failed")

    for prefix, expected_decision in (
        ("hybrid", "hybrid_train_not_supported"),
        ("oracle", "where_bottleneck_supported"),
        ("relative", "relative_where_train_not_supported"),
    ):
        complete = _read_json(paths[f"{prefix}_complete"])
        evaluation = _read_json(paths[f"{prefix}_evaluation"])
        if (
            complete.get("decision") != expected_decision
            or complete.get("evaluation", {}).get("sha256")
            != _sha256(paths[f"{prefix}_evaluation"])
            or evaluation.get("validation_or_test_inputs_used") is not False
        ):
            raise ValueError(f"literature-attention frozen {prefix} comparator changed")
    source_payload = _read_json(paths["bootstrap_sources"])
    sources = source_payload.get("sources")
    if (
        source_payload.get("schema") != "infographicvqa_decar_bootstrap_sources_v1"
        or not isinstance(sources, list)
        or len(sources) != EXPECTED_SOURCES
        or len(set(sources)) != EXPECTED_SOURCES
        or sources != sorted(sources)
    ):
        raise ValueError("literature-attention bootstrap source order changed")
    return {
        "passed": True,
        "literature_features_outcomes_included": False,
        "raw_attention_reproduced": True,
        "frozen_comparator_evaluations": 4,
        "validation_or_test_inputs_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    input_names = (
        "literature-features",
        "literature-complete",
        "literature-audit",
        "literature-merge-report",
        "raw-attention-features",
        "raw-attention-complete",
        "raw-attention-audit",
        "raw-attention-evaluation",
        "raw-attention-evaluation-complete",
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
    )
    for name in input_names:
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--expected-{name}-sha256", required=True)
    parser.add_argument("--expected-literature-code-revision", required=True)
    parser.add_argument("--expected-raw-attention-code-revision", required=True)
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
            f"refusing to overwrite literature-attention evaluation: {output_dir}"
        )
    contract_audit = _verify_contract(paths)
    bootstrap_indices = np.load(paths["bootstrap_indices"], mmap_mode="r")
    if (
        bootstrap_indices.shape != (DECAR_BOOTSTRAP_RESAMPLES, EXPECTED_SOURCES)
        or bootstrap_indices.dtype != np.int32
        or int(bootstrap_indices.min()) != 0
        or int(bootstrap_indices.max()) != EXPECTED_SOURCES - 1
    ):
        raise ValueError("literature-attention bootstrap matrix contract failed")
    evaluation = evaluate_literature_attention_where(
        read_jsonl(paths["rollouts"]),
        load_semantic_feature_dataset(paths["literature_features"]),
        load_semantic_feature_dataset(paths["raw_attention_features"]),
        _read_jsonl_objects(paths["decar_predictions"]),
        _read_jsonl_objects(paths["relative_predictions"]),
        _read_jsonl_objects(paths["answer_nll"]),
        _read_json(paths["hybrid_evaluation"]),
        _read_json(paths["oracle_evaluation"]),
        _read_json(paths["relative_evaluation"]),
        _read_json(paths["raw_attention_evaluation"]),
        expected_literature_code_revision=args.expected_literature_code_revision,
        expected_raw_attention_code_revision=args.expected_raw_attention_code_revision,
        expected_model_revision=args.expected_model_revision,
        expected_source_features_sha256=args.expected_source_features_sha256,
        expected_rollouts_sha256=_sha256(paths["rollouts"]),
        bootstrap_indices=bootstrap_indices,
        expected_decisions=EXPECTED_DECISIONS,
        expected_sources=EXPECTED_SOURCES,
    )
    del bootstrap_indices
    if evaluation.get("schema") != LITERATURE_ATTENTION_EVALUATION_SCHEMA:
        raise RuntimeError("literature-attention evaluation schema changed")
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
                "schema": "infographicvqa_literature_attention_where_decision_v1",
                "decision": evaluation["decision"],
                "selected_variant_and_operating_point": evaluation[
                    "selected_variant_and_operating_point"
                ],
                "validation_opened": False,
                "test_opened": False,
                "evaluation_sha256": _sha256(evaluation_path),
            },
        )
        completion = {
            "schema": (
                "infographicvqa_literature_attention_where_evaluation_complete_v1"
            ),
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
            "multiplicity_corrected": True,
            "literature_features_outcomes_included": False,
            "privileged_teacher_used_only_in_evaluation": True,
            "validation_or_test_inputs_used": False,
        }
        _write_json(temporary / "complete.json", completion)
        temporary.replace(output_dir)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
