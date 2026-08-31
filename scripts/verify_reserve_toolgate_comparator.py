#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping

from beyond_entropy.dataset import read_jsonl
from beyond_entropy.qwen_semantic import (
    load_semantic_feature_dataset,
    validate_semantic_feature_dataset,
)
from beyond_entropy.reserve_toolgate import fit_reserve_toolgate_comparator


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mapping(path: Path, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _compare_value(
    expected: Any,
    actual: Any,
    *,
    name: str,
    numeric_differences: list[float],
) -> None:
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected is not actual:
            raise ValueError(f"frozen comparator differs at {name}")
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        left = float(expected)
        right = float(actual)
        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError(f"frozen comparator has non-finite value at {name}")
        difference = abs(left - right)
        numeric_differences.append(difference)
        if difference > 1e-12:
            raise ValueError(
                f"frozen comparator differs at {name}: {left!r} != {right!r}"
            )
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            raise ValueError(f"frozen comparator length differs at {name}")
        for index, (left, right) in enumerate(zip(expected, actual)):
            _compare_value(
                left,
                right,
                name=f"{name}[{index}]",
                numeric_differences=numeric_differences,
            )
        return
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        if set(expected) != set(actual):
            raise ValueError(f"frozen comparator keys differ at {name}")
        for key in sorted(expected):
            _compare_value(
                expected[key],
                actual[key],
                name=f"{name}.{key}",
                numeric_differences=numeric_differences,
            )
        return
    if expected != actual:
        raise ValueError(f"frozen comparator differs at {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce and verify the outcome-sealed ToolGate comparator"
    )
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--policy-a-model", type=Path, required=True)
    parser.add_argument("--frozen-model", type=Path, required=True)
    parser.add_argument("--frozen-report", type=Path, required=True)
    parser.add_argument("--expected-rollouts-sha256", required=True)
    parser.add_argument("--expected-features-sha256", required=True)
    parser.add_argument("--expected-policy-a-model-sha256", required=True)
    parser.add_argument("--expected-frozen-model-sha256", required=True)
    parser.add_argument("--expected-frozen-report-sha256", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    paths = {
        "rollouts": args.rollouts.resolve(),
        "features": args.features.resolve(),
        "policy_a_model": args.policy_a_model.resolve(),
        "frozen_model": args.frozen_model.resolve(),
        "frozen_report": args.frozen_report.resolve(),
    }
    expected_hashes = {
        "rollouts": args.expected_rollouts_sha256,
        "features": args.expected_features_sha256,
        "policy_a_model": args.expected_policy_a_model_sha256,
        "frozen_model": args.expected_frozen_model_sha256,
        "frozen_report": args.expected_frozen_report_sha256,
    }
    observed_hashes: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")
        observed_hashes[name] = _sha256(path)
        if observed_hashes[name] != expected_hashes[name]:
            raise ValueError(f"{name} SHA-256 mismatch")

    records = read_jsonl(paths["rollouts"])
    feature_payload = load_semantic_feature_dataset(paths["features"])
    validate_semantic_feature_dataset(feature_payload, records)
    semantic = {
        (str(item["state_id"]), str(item["replicate_id"])): item
        for item in feature_payload["decisions"]
    }
    reproduced_report, reproduced_model = fit_reserve_toolgate_comparator(
        {"docvqa": records},
        semantic_decisions_by_domain={"docvqa": semantic},
    )
    frozen_model = _load_mapping(paths["frozen_model"], "frozen model")
    frozen_report = _load_mapping(paths["frozen_report"], "frozen report")
    provenance_only = {
        "code_revision",
        "development_inputs",
        "policy_a_model",
    }
    scientific_frozen_model = {
        key: value for key, value in frozen_model.items() if key not in provenance_only
    }
    differences: list[float] = []
    _compare_value(
        scientific_frozen_model,
        reproduced_model,
        name="model",
        numeric_differences=differences,
    )
    if frozen_report.get("run", {}).get("reserve_outcomes_used") is not False:
        raise ValueError("frozen comparator report used reserve outcomes")
    if frozen_report.get("run", {}).get("formal_outcomes_used") is not False:
        raise ValueError("frozen comparator report used formal outcomes")
    if reproduced_report.get("reserve_outcomes_used") is not False:
        raise ValueError("reproduced comparator unexpectedly used reserve outcomes")
    repo = Path(__file__).resolve().parents[1]
    result = {
        "passed": True,
        "scientific_status": (
            "frozen ToolGate comparator exactly reproduced from development inputs"
        ),
        "code_revision": subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "input_sha256": observed_hashes,
        "development_decisions": len(semantic),
        "maximum_numeric_difference": max(differences, default=0.0),
        "reserve_outcomes_used": False,
        "formal_outcomes_used": False,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
