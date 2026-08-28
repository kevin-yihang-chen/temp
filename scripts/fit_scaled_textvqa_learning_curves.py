from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from beyond_entropy.dataset import read_jsonl
from beyond_entropy.qwen_semantic import (
    load_semantic_feature_dataset,
    validate_semantic_feature_dataset,
)
from beyond_entropy.scaled_action_value import fit_scaled_pairwise_action_value_model
from beyond_entropy.schema import ActionRecord


SOURCE_COUNTS = (200, 1000, 3000, 5000)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def registered_source_prefixes(
    allocation: Mapping[str, Any],
    counts: Sequence[int] = SOURCE_COUNTS,
) -> dict[int, tuple[str, ...]]:
    body = allocation.get("allocation")
    if not isinstance(body, Mapping):
        raise ValueError("allocation body must be a mapping")
    roles = body.get("roles")
    if not isinstance(roles, Mapping):
        raise ValueError("allocation roles must be a mapping")
    role = roles.get("ranker_training")
    if not isinstance(role, Mapping) or not isinstance(role.get("assignments"), list):
        raise ValueError("allocation is missing ranker-training assignments")
    assignments = sorted(
        role["assignments"],
        key=lambda item: (int(item["source_rank"]), str(item["source_group_id"])),
    )
    ordered = tuple(str(item["source_group_id"]) for item in assignments)
    if len(ordered) != len(set(ordered)):
        raise ValueError("ranker-training source assignments are not unique")
    requested = tuple(int(count) for count in counts)
    if not requested or any(count <= 0 for count in requested):
        raise ValueError("learning-curve source counts must be positive")
    if tuple(sorted(set(requested))) != requested:
        raise ValueError("learning-curve source counts must be unique and increasing")
    if requested[-1] > len(ordered):
        raise ValueError("learning-curve prefix exceeds the allocated source role")
    return {count: ordered[:count] for count in requested}


def _subset_records(
    records: Sequence[ActionRecord],
    source_groups: Sequence[str],
) -> list[ActionRecord]:
    source_ids = {f"textvqa:{group_id}" for group_id in source_groups}
    selected = [record for record in records if record.source_id in source_ids]
    observed = {record.source_id for record in selected}
    if observed != source_ids:
        raise ValueError("learning-curve rollouts do not cover the registered prefix")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit preregistered secondary TextVQA source learning curves"
    )
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--expected-allocation-sha256", required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--expected-rollouts-sha256", required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--expected-features-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parents[1]
    code_revision = os.environ.get("BE_CODE_REVISION")
    if not code_revision:
        code_revision = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    actual_hashes = {
        "allocation": _sha256(args.allocation),
        "rollouts": _sha256(args.rollouts),
        "features": _sha256(args.features),
    }
    for name, expected in (
        ("allocation", args.expected_allocation_sha256),
        ("rollouts", args.expected_rollouts_sha256),
        ("features", args.expected_features_sha256),
    ):
        if actual_hashes[name] != expected:
            raise ValueError(f"{name} SHA-256 mismatch")
    allocation = json.loads(args.allocation.read_text(encoding="utf-8"))
    prefixes = registered_source_prefixes(allocation)
    records = read_jsonl(args.rollouts)
    features = load_semantic_feature_dataset(args.features)
    validate_semantic_feature_dataset(features, records)
    if bool(features["metadata"].get("outcomes_included", True)):
        raise ValueError("learning curves require label-free feature storage")
    all_semantic = {
        (str(decision["state_id"]), str(decision["replicate_id"])): decision
        for decision in features["decisions"]
    }
    curves: list[dict[str, Any]] = []
    for count in SOURCE_COUNTS:
        selected_records = _subset_records(records, prefixes[count])
        selected_keys = {
            (record.state_id, record.replicate_id) for record in selected_records
        }
        semantic = {key: all_semantic[key] for key in selected_keys}
        report, _ = fit_scaled_pairwise_action_value_model(
            selected_records,
            feature_mode="semantic-context",
            semantic_decisions=semantic,
            n_folds=5,
            lambda_cost=0.05,
            ranker_c_values=(0.01, 0.1, 1.0),
            call_alpha_values=(1.0, 10.0, 100.0),
            max_thresholds=32,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=20260828,
        )
        curves.append(
            {
                "source_count": count,
                "decision_count": report["n_decisions"],
                "selected_ranker": report["selected_ranker"],
                "selected_call_value": report["selected_call_value"],
                "nested_ranking": report["nested_ranking"],
                "oof_zero_threshold_policy": report["oof_zero_threshold_policy"],
                "threshold_count": len(report["threshold_grid"]),
            }
        )
    result = {
        "scientific_status": (
            "secondary preregistered source-prefix learning curves; cannot select "
            "or replace the full-scale primary policy"
        ),
        "feature_mode": "semantic-context",
        "source_counts": list(SOURCE_COUNTS),
        "curves": curves,
        "run": {
            "code_revision": code_revision,
            "allocation": str(args.allocation.resolve()),
            "allocation_sha256": actual_hashes["allocation"],
            "rollouts": str(args.rollouts.resolve()),
            "rollouts_sha256": actual_hashes["rollouts"],
            "features": str(args.features.resolve()),
            "features_sha256": actual_hashes["features"],
            "risk_calibration_outcomes_used": False,
            "formal_outcomes_used": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
