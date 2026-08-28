#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from statistics import mean
from typing import Any

from beyond_entropy.action_value import (
    select_frozen_factorized_action_value_actions,
)
from beyond_entropy.dataset import group_by_decision, read_jsonl
from beyond_entropy.qwen_semantic import (
    load_semantic_feature_dataset,
    validate_semantic_feature_dataset,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc stopping-versus-ranking decomposition of a frozen policy"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--expected-rollouts-sha256", required=True)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--expected-features-sha256")
    parser.add_argument(
        "--require-label-free-features",
        action="store_true",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_path = args.model.resolve()
    rollout_path = args.rollouts.resolve()
    hashes = {
        "model": _sha256(model_path),
        "rollouts": _sha256(rollout_path),
    }
    if hashes["model"] != args.expected_model_sha256:
        raise ValueError("model SHA-256 mismatch")
    if hashes["rollouts"] != args.expected_rollouts_sha256:
        raise ValueError("rollout SHA-256 mismatch")
    model: dict[str, Any] = json.loads(model_path.read_text(encoding="utf-8"))
    records = read_jsonl(rollout_path)
    semantic_decisions = None
    feature_mode = str(model.get("feature_mode"))
    if feature_mode in {"semantic-context", "hybrid-context-semantic"}:
        if args.features is None or not args.expected_features_sha256:
            raise ValueError("semantic decomposition requires a frozen feature hash")
        feature_path = args.features.resolve()
        hashes["features"] = _sha256(feature_path)
        if hashes["features"] != args.expected_features_sha256:
            raise ValueError("features SHA-256 mismatch")
        payload = load_semantic_feature_dataset(feature_path)
        validate_semantic_feature_dataset(
            payload,
            records,
            require_outcomes=False if args.require_label_free_features else None,
        )
        semantic_decisions = {
            (str(decision["state_id"]), str(decision["replicate_id"])): decision
            for decision in payload["decisions"]
        }
    elif args.features is not None:
        raise ValueError("context-only decomposition must not receive features")
    grouped = group_by_decision(records)
    selected, scores = select_frozen_factorized_action_value_actions(
        model,
        records,
        semantic_decisions=semantic_decisions,
    )
    always_call_model = dict(model)
    always_call_model["threshold"] = float("-inf")
    top_actions, _ = select_frozen_factorized_action_value_actions(
        always_call_model,
        records,
        semantic_decisions=semantic_decisions,
    )
    lambda_cost = float(model["lambda_cost"])
    rows = []
    for key, siblings in grouped.items():
        zooms = {record.action_id: record for record in siblings if record.action_type == "ZOOM"}
        top_action_id = top_actions[key]
        if not zooms or top_action_id is None:
            raise ValueError(f"decision {key!r} lacks ranked crop actions")
        top = zooms[top_action_id]
        selected_action_id = selected[key]
        called = selected_action_id is not None
        chosen = zooms[selected_action_id] if selected_action_id is not None else None
        oracle_gain = max(record.delta_success for record in zooms.values())
        oracle_utility = max(0.0, *(record.voi(lambda_cost) for record in zooms.values()))
        rows.append(
            {
                "called": called,
                "score": scores[key],
                "top_gain": top.delta_success,
                "top_utility": top.voi(lambda_cost),
                "chosen_gain": chosen.delta_success if chosen is not None else 0.0,
                "chosen_utility": chosen.voi(lambda_cost) if chosen is not None else 0.0,
                "oracle_gain": oracle_gain,
                "oracle_utility": oracle_utility,
            }
        )
    calls = sum(bool(row["called"]) for row in rows)
    helpful = sum(float(row["oracle_gain"]) > 0.0 for row in rows)
    worthwhile = sum(float(row["oracle_utility"]) > 0.0 for row in rows)
    calls_on_helpful = sum(
        bool(row["called"]) and float(row["oracle_gain"]) > 0.0 for row in rows
    )
    calls_on_worthwhile = sum(
        bool(row["called"]) and float(row["oracle_utility"]) > 0.0 for row in rows
    )
    realized_worthwhile_calls = sum(
        bool(row["called"]) and float(row["chosen_utility"]) > 0.0 for row in rows
    )
    report = {
        "scientific_status": (
            "post-hoc failure decomposition; no threshold, model, or formal claim is "
            "changed"
        ),
        "decisions": len(rows),
        "lambda_cost": lambda_cost,
        "frozen_threshold": float(model["threshold"]),
        "calls": calls,
        "tool_rate": calls / len(rows),
        "helpful_states": helpful,
        "oracle_worthwhile_states": worthwhile,
        "stopping": {
            "call_precision_for_any_positive_gain": _ratio(calls_on_helpful, calls),
            "call_recall_for_any_positive_gain": _ratio(calls_on_helpful, helpful),
            "call_precision_for_oracle_positive_utility": _ratio(
                calls_on_worthwhile, calls
            ),
            "call_recall_for_oracle_positive_utility": _ratio(
                calls_on_worthwhile, worthwhile
            ),
            "realized_positive_utility_call_precision": _ratio(
                realized_worthwhile_calls, calls
            ),
        },
        "ranking": {
            "always_top1_mean_gain": mean(float(row["top_gain"]) for row in rows),
            "top1_rescue_rate_within_helpful_states": mean(
                float(row["top_gain"]) > 0.0
                for row in rows
                if float(row["oracle_gain"]) > 0.0
            ),
            "top1_positive_utility_rate_within_oracle_worthwhile_states": mean(
                float(row["top_utility"]) > 0.0
                for row in rows
                if float(row["oracle_utility"]) > 0.0
            ),
        },
        "utility_decomposition": {
            "frozen_stopping_and_ranking": mean(
                float(row["chosen_utility"]) for row in rows
            ),
            "frozen_stopping_oracle_action": mean(
                (
                    float(row["oracle_gain"]) - lambda_cost
                    if bool(row["called"])
                    else 0.0
                )
                for row in rows
            ),
            "oracle_stopping_frozen_ranking": mean(
                max(0.0, float(row["top_utility"])) for row in rows
            ),
            "oracle_stopping_and_action": mean(
                float(row["oracle_utility"]) for row in rows
            ),
        },
        "run": {
            "code_revision": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "paths": {
                "model": str(model_path),
                "rollouts": str(rollout_path),
                **(
                    {"features": str(args.features.resolve())}
                    if args.features is not None
                    else {}
                ),
            },
            "sha256": hashes,
            "required_label_free_features": args.require_label_free_features,
            "formal_outcomes_used": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
