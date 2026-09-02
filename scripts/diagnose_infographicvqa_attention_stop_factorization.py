#!/usr/bin/env python3
"""Run the opened-train raw-attention stop-versus-where diagnostic."""

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

from beyond_entropy.dataset import read_jsonl
from beyond_entropy.infographicvqa_attention_stop_diagnostic import (
    ATTENTION_STOP_DIAGNOSTIC_SCHEMA,
    evaluate_attention_stop_factorization,
)
from beyond_entropy.infographicvqa_decar_evaluation import DECAR_BOOTSTRAP_RESAMPLES
from beyond_entropy.infographicvqa_relative_where_evaluation import (
    _first_frozen_difference,
)
from beyond_entropy.qwen_semantic import load_semantic_feature_dataset

MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
SOURCE_FEATURES_SHA256 = (
    "d0508726a50b4c1e54778392d08329b242a680fc13292cac1ebec8b42a175300"
)


def _verify_contract(paths: Mapping[str, Path]) -> dict[str, Any]:
    feature_complete = _read_json(paths["attention_complete"])
    feature_audit = _read_json(paths["attention_audit"])
    evaluation = _read_json(paths["attention_evaluation"])
    evaluation_complete = _read_json(paths["attention_evaluation_complete"])
    source_payload = _read_json(paths["bootstrap_sources"])
    sources = source_payload.get("sources")
    if (
        feature_complete.get("schema")
        != "infographicvqa_attention_where_feature_complete_v1"
        or feature_complete.get("passed") is not True
        or feature_complete.get("decisions") != EXPECTED_DECISIONS
        or feature_complete.get("sources") != EXPECTED_SOURCES
        or feature_complete.get("images") != EXPECTED_IMAGES
        or feature_complete.get("merged_features_sha256")
        != _sha256(paths["attention_features"])
        or feature_complete.get("audit_sha256") != _sha256(paths["attention_audit"])
        or feature_audit.get("passed") is not True
        or feature_audit.get("outcomes_included") is not False
        or feature_audit.get("validation_or_test_inputs_used") is not False
        or evaluation.get("decision") != "attention_where_train_not_supported"
        or evaluation.get("validation_or_test_inputs_used") is not False
        or evaluation_complete.get("decision") != "attention_where_train_not_supported"
        or evaluation_complete.get("evaluation", {}).get("sha256")
        != _sha256(paths["attention_evaluation"])
        or source_payload.get("schema") != "infographicvqa_decar_bootstrap_sources_v1"
        or not isinstance(sources, list)
        or len(sources) != EXPECTED_SOURCES
        or len(set(sources)) != EXPECTED_SOURCES
        or sources != sorted(sources)
    ):
        raise ValueError("attention-stop diagnostic input contract failed")
    return {
        "passed": True,
        "raw_negative_decision_preserved": True,
        "feature_outcomes_included": False,
        "validation_or_test_inputs_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    input_names = (
        "rollouts",
        "attention-features",
        "attention-complete",
        "attention-audit",
        "attention-evaluation",
        "attention-evaluation-complete",
        "bootstrap-indices",
        "bootstrap-sources",
        "protocol",
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
            f"refusing to overwrite attention-stop diagnostic: {output_dir}"
        )
    contract = _verify_contract(paths)
    bootstrap_indices = np.load(paths["bootstrap_indices"], mmap_mode="r")
    if (
        bootstrap_indices.shape != (DECAR_BOOTSTRAP_RESAMPLES, EXPECTED_SOURCES)
        or bootstrap_indices.dtype != np.int32
        or int(bootstrap_indices.min()) != 0
        or int(bootstrap_indices.max()) != EXPECTED_SOURCES - 1
    ):
        raise ValueError("attention-stop diagnostic bootstrap contract failed")
    diagnostic = evaluate_attention_stop_factorization(
        read_jsonl(paths["rollouts"]),
        load_semantic_feature_dataset(paths["attention_features"]),
        expected_attention_code_revision=args.expected_attention_code_revision,
        expected_model_revision=args.expected_model_revision,
        expected_source_features_sha256=args.expected_source_features_sha256,
        expected_rollouts_sha256=_sha256(paths["rollouts"]),
        bootstrap_indices=bootstrap_indices,
        expected_decisions=EXPECTED_DECISIONS,
        expected_sources=EXPECTED_SOURCES,
    )
    del bootstrap_indices
    if diagnostic.get("schema") != ATTENTION_STOP_DIAGNOSTIC_SCHEMA:
        raise RuntimeError("attention-stop diagnostic schema changed")
    raw_points = _read_json(paths["attention_evaluation"]).get("operating_points")
    diagnostic_points = diagnostic.get("operating_points")
    if (
        not isinstance(raw_points, list)
        or not isinstance(diagnostic_points, list)
        or len(raw_points) != len(diagnostic_points)
    ):
        raise ValueError("attention-stop raw operating-point family changed")
    for raw_point, diagnostic_point in zip(raw_points, diagnostic_points, strict=True):
        if raw_point.get("name") != diagnostic_point.get("name"):
            raise ValueError("attention-stop raw operating-point identity changed")
        difference = _first_frozen_difference(
            diagnostic_point["policies"]["entropy_stop"],
            raw_point["policies"]["attention_where"],
        )
        if difference is not None:
            raise ValueError(
                f"attention-stop entropy policy failed reproduction: {difference}"
            )
    diagnostic["inputs"] = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in paths.items()
    }
    diagnostic["contract_audit"] = contract
    diagnostic["contract_audit"]["raw_entropy_policy_reproduced"] = True
    diagnostic["bootstrap"]["indices"] = {
        "path": str(paths["bootstrap_indices"]),
        "sha256": _sha256(paths["bootstrap_indices"]),
        "shape": [DECAR_BOOTSTRAP_RESAMPLES, EXPECTED_SOURCES],
        "dtype": "int32",
        "reused_from_formal_evaluation": True,
    }
    diagnostic["bootstrap"]["source_order"] = {
        "path": str(paths["bootstrap_sources"]),
        "sha256": _sha256(paths["bootstrap_sources"]),
        "reused_from_formal_evaluation": True,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=output_dir.name + ".partial-", dir=output_dir.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        diagnostic_path = temporary / "diagnostic.json"
        _write_json(diagnostic_path, diagnostic)
        completion = {
            "schema": "infographicvqa_attention_stop_factorization_complete_v1",
            "passed": True,
            "diagnostic": {
                "path": "diagnostic.json",
                "sha256": _sha256(diagnostic_path),
            },
            "raw_negative_decision_preserved": True,
            "valid_for_formal_selection": False,
            "validation_or_test_inputs_used": False,
        }
        _write_json(temporary / "complete.json", completion)
        temporary.replace(output_dir)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
