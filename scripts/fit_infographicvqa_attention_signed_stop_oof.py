#!/usr/bin/env python3
"""Fit the frozen InfographicVQA fixed-action signed-value OOF stop."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
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
from beyond_entropy.infographicvqa_attention_signed_stop import (
    ATTENTION_SIGNED_STOP_PRIMARY_CALLS,
    ATTENTION_SIGNED_STOP_SCHEMA,
    evaluate_attention_signed_stop,
    smoke_attention_signed_stop,
)
from beyond_entropy.infographicvqa_decar_evaluation import DECAR_BOOTSTRAP_RESAMPLES
from beyond_entropy.qwen_semantic import load_semantic_feature_dataset


MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
SOURCE_FEATURES_SHA256 = (
    "d0508726a50b4c1e54778392d08329b242a680fc13292cac1ebec8b42a175300"
)
EXPECTED_POSITIVE_NET_STATES = 1_023


def _verify_contract(paths: Mapping[str, Path]) -> dict[str, Any]:
    feature_complete = _read_json(paths["attention_complete"])
    feature_audit = _read_json(paths["attention_audit"])
    attention_evaluation = _read_json(paths["attention_evaluation"])
    attention_evaluation_complete = _read_json(
        paths["attention_evaluation_complete"]
    )
    diagnostic = _read_json(paths["stop_diagnostic"])
    diagnostic_complete = _read_json(paths["stop_diagnostic_complete"])
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
        or feature_complete.get("audit_sha256")
        != _sha256(paths["attention_audit"])
        or feature_audit.get("passed") is not True
        or feature_audit.get("outcomes_included") is not False
        or feature_audit.get("validation_or_test_inputs_used") is not False
        or attention_evaluation.get("decision")
        != "attention_where_train_not_supported"
        or attention_evaluation.get("validation_or_test_inputs_used") is not False
        or attention_evaluation_complete.get("decision")
        != "attention_where_train_not_supported"
        or diagnostic.get("schema")
        != "infographicvqa_attention_stop_factorization_diagnostic_v1"
        or diagnostic.get("raw_action_positive_net", {}).get(
            "positive_net_states"
        )
        != EXPECTED_POSITIVE_NET_STATES
        or diagnostic.get("valid_for_formal_selection") is not False
        or diagnostic.get("validation_or_test_inputs_used") is not False
        or diagnostic_complete.get("schema")
        != "infographicvqa_attention_stop_factorization_complete_v1"
        or diagnostic_complete.get("passed") is not True
        or diagnostic_complete.get("diagnostic", {}).get("sha256")
        != _sha256(paths["stop_diagnostic"])
        or source_payload.get("schema")
        != "infographicvqa_decar_bootstrap_sources_v1"
        or not isinstance(sources, list)
        or len(sources) != EXPECTED_SOURCES
        or sources != sorted(sources)
        or len(set(sources)) != EXPECTED_SOURCES
    ):
        raise ValueError("attention signed stop input contract failed")
    return {
        "passed": True,
        "raw_negative_decision_preserved": True,
        "diagnostic_is_privileged_only": True,
        "feature_outcomes_included": False,
        "validation_or_test_inputs_used": False,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    input_names = (
        "rollouts",
        "attention-features",
        "attention-complete",
        "attention-audit",
        "attention-evaluation",
        "attention-evaluation-complete",
        "stop-diagnostic",
        "stop-diagnostic-complete",
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
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    normalized_names = tuple(name.replace("-", "_") for name in input_names)
    paths = {
        name: _checked(
            getattr(args, name), getattr(args, f"expected_{name}_sha256"), name
        )
        for name in normalized_names
    }
    contract = _verify_contract(paths)
    records = read_jsonl(paths["rollouts"])
    feature_payload = load_semantic_feature_dataset(paths["attention_features"])
    shared = {
        "expected_attention_code_revision": args.expected_attention_code_revision,
        "expected_model_revision": args.expected_model_revision,
        "expected_source_features_sha256": args.expected_source_features_sha256,
        "expected_rollouts_sha256": _sha256(paths["rollouts"]),
        "expected_decisions": EXPECTED_DECISIONS,
        "expected_sources": EXPECTED_SOURCES,
        "expected_positive_net_states": EXPECTED_POSITIVE_NET_STATES,
    }
    if args.smoke_only:
        if args.output_dir is not None:
            raise ValueError("attention signed stop smoke does not write output")
        smoke = smoke_attention_signed_stop(records, feature_payload, **shared)
        smoke["contract_audit"] = contract
        smoke["inputs_verified"] = True
        print(json.dumps(smoke, indent=2, sort_keys=True))
        return
    if args.output_dir is None:
        raise ValueError("attention signed stop full fit requires --output-dir")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite attention signed stop output: {output_dir}"
        )
    bootstrap_indices = np.load(paths["bootstrap_indices"], mmap_mode="r")
    if (
        bootstrap_indices.shape
        != (DECAR_BOOTSTRAP_RESAMPLES, EXPECTED_SOURCES)
        or bootstrap_indices.dtype != np.int32
        or int(bootstrap_indices.min()) != 0
        or int(bootstrap_indices.max()) != EXPECTED_SOURCES - 1
    ):
        raise ValueError("attention signed stop bootstrap contract failed")
    report, model, score_rows = evaluate_attention_signed_stop(
        records,
        feature_payload,
        bootstrap_indices=bootstrap_indices,
        expected_bootstrap_resamples=DECAR_BOOTSTRAP_RESAMPLES,
        **shared,
    )
    del bootstrap_indices
    if (
        report.get("schema") != ATTENTION_SIGNED_STOP_SCHEMA
        or report.get("primary_calls") != ATTENTION_SIGNED_STOP_PRIMARY_CALLS
        or report.get("validation_or_test_inputs_used") is not False
        or report.get("valid_for_formal_claim") is not False
    ):
        raise RuntimeError("attention signed stop output contract changed")
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
        "contract_audit": contract,
        "validation_or_test_inputs_used": False,
        "protected_role_inputs_used": False,
    }
    report["run"] = run
    model["run"] = run

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=output_dir.name + ".partial-", dir=output_dir.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        report_path = temporary / "report.json"
        model_path = temporary / "model.json"
        scores_path = temporary / "scores.jsonl"
        _write_json(report_path, report)
        _write_json(model_path, model)
        _write_jsonl(scores_path, score_rows)
        completion = {
            "schema": "infographicvqa_attention_signed_stop_oof_complete_v1",
            "decision": report["decision"],
            "report": {"path": "report.json", "sha256": _sha256(report_path)},
            "model": {"path": "model.json", "sha256": _sha256(model_path)},
            "scores": {"path": "scores.jsonl", "sha256": _sha256(scores_path)},
            "primary_calls": ATTENTION_SIGNED_STOP_PRIMARY_CALLS,
            "validation_or_test_inputs_used": False,
            "valid_for_formal_claim": False,
        }
        _write_json(temporary / "complete.json", completion)
        temporary.replace(output_dir)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
