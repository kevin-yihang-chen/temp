#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from beyond_entropy.action_value import (
    select_frozen_factorized_action_value_actions,
)
from beyond_entropy.dataset import read_jsonl
from beyond_entropy.qwen_semantic import (
    load_semantic_feature_dataset,
    validate_semantic_feature_dataset,
)


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("comparison summary requires values")
    ordered = sorted(values)
    percentile_index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    return {
        "minimum": min(values),
        "mean": mean(values),
        "p95": ordered[percentile_index],
        "maximum": max(values),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare separate and joint frozen question-attention passes"
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--joint", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if _sha256(args.model) != args.expected_model_sha256:
        raise ValueError("frozen model SHA-256 mismatch")

    import torch  # type: ignore[import-not-found]
    import torch.nn.functional as functional  # type: ignore[import-not-found]

    records = read_jsonl(args.rollouts)
    reference = load_semantic_feature_dataset(args.reference)
    joint = load_semantic_feature_dataset(args.joint)
    validate_semantic_feature_dataset(reference, records)
    validate_semantic_feature_dataset(joint, records)
    reference_by_key = {
        (str(item["state_id"]), str(item["replicate_id"])): item
        for item in reference["decisions"]
    }
    joint_by_key = {
        (str(item["state_id"]), str(item["replicate_id"])): item
        for item in joint["decisions"]
    }
    if set(reference_by_key) != set(joint_by_key):
        raise ValueError("feature files do not cover identical decisions")

    question_cosine: list[float] = []
    question_max_abs: list[float] = []
    attention_max_abs: list[float] = []
    for key in sorted(reference_by_key):
        left = reference_by_key[key]
        right = joint_by_key[key]
        if list(left["action_ids"]) != list(right["action_ids"]):
            raise ValueError(f"action IDs differ for decision {key!r}")
        left_question = left["question_embedding"].float()
        right_question = right["question_embedding"].float()
        if left_question.shape != right_question.shape:
            raise ValueError(f"question dimensions differ for decision {key!r}")
        question_cosine.append(
            float(functional.cosine_similarity(left_question, right_question, dim=0))
        )
        question_max_abs.append(float((left_question - right_question).abs().max()))
        left_attention = left.get("question_region_attention")
        right_attention = right.get("question_region_attention")
        if left_attention is None or right_attention is None:
            raise ValueError(f"attention is missing for decision {key!r}")
        if left_attention.shape != right_attention.shape:
            raise ValueError(f"attention dimensions differ for decision {key!r}")
        attention_max_abs.append(float((left_attention - right_attention).abs().max()))

    model: dict[str, Any] = json.loads(args.model.read_text(encoding="utf-8"))
    reference_actions, reference_scores = (
        select_frozen_factorized_action_value_actions(
            model,
            records,
            semantic_decisions=reference_by_key,
        )
    )
    joint_actions, joint_scores = select_frozen_factorized_action_value_actions(
        model,
        records,
        semantic_decisions=joint_by_key,
    )
    keys = sorted(reference_actions)
    score_abs = [abs(reference_scores[key] - joint_scores[key]) for key in keys]
    report = {
        "scientific_status": (
            "development-only engineering equivalence check; no outcome metric used"
        ),
        "decisions": len(keys),
        "question_embedding_cosine": _summary(question_cosine),
        "question_embedding_max_absolute_difference": _summary(question_max_abs),
        "question_region_attention_max_absolute_difference": _summary(
            attention_max_abs
        ),
        "predicted_net_value_absolute_difference": _summary(score_abs),
        "frozen_policy": {
            "exact_decision_agreement": mean(
                reference_actions[key] == joint_actions[key] for key in keys
            ),
            "reference_calls": sum(reference_actions[key] is not None for key in keys),
            "joint_calls": sum(joint_actions[key] is not None for key in keys),
        },
        "run": {
            "reference": str(args.reference.resolve()),
            "reference_sha256": _sha256(args.reference),
            "joint": str(args.joint.resolve()),
            "joint_sha256": _sha256(args.joint),
            "rollouts": str(args.rollouts.resolve()),
            "rollouts_sha256": _sha256(args.rollouts),
            "model": str(args.model.resolve()),
            "model_sha256": _sha256(args.model),
            "formal_outcomes_used": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
