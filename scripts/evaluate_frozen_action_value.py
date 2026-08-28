#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from beyond_entropy.action_value import (
    evaluate_frozen_action_value_model,
    evaluate_frozen_factorized_action_value_model,
)
from beyond_entropy.dataset import read_jsonl
from beyond_entropy.qwen_semantic import (
    load_semantic_feature_dataset,
    validate_semantic_feature_dataset,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one serialized direct or factorized action-value model"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-model-sha256")
    parser.add_argument("--expected-rollouts-sha256")
    parser.add_argument("--expected-features-sha256")
    parser.add_argument(
        "--require-label-free-features",
        action="store_true",
        help="reject semantic feature files that contain target outcome fields",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--cluster-by",
        choices=("state_id", "image_id", "source_id"),
        default="state_id",
    )
    args = parser.parse_args()

    paths = {"model": args.model.resolve(), "rollouts": args.rollouts.resolve()}
    if args.features is not None:
        paths["features"] = args.features.resolve()
    expected = {
        "model": args.expected_model_sha256,
        "rollouts": args.expected_rollouts_sha256,
        "features": args.expected_features_sha256,
    }
    hashes = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[name] = _sha256(path)
        if expected[name] and hashes[name] != expected[name]:
            raise ValueError(f"{name} SHA-256 mismatch")

    model: dict[str, Any] = json.loads(paths["model"].read_text(encoding="utf-8"))
    records = read_jsonl(paths["rollouts"])
    semantic_decisions = None
    if model.get("feature_mode") in {
        "semantic-context",
        "hybrid-context-semantic",
    }:
        if "features" not in paths:
            raise ValueError("semantic action model requires --features")
        payload = load_semantic_feature_dataset(paths["features"])
        validate_semantic_feature_dataset(
            payload,
            records,
            require_outcomes=False if args.require_label_free_features else None,
        )
        semantic_decisions = {
            (str(decision["state_id"]), str(decision["replicate_id"])): decision
            for decision in payload["decisions"]
        }
    elif "features" in paths:
        raise ValueError("context-geometry model must not receive semantic features")

    model_type = model.get("model_type")
    if model_type == "multidomain_direct_action_value":
        result = evaluate_frozen_action_value_model(
            model,
            records,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
            cluster_by=args.cluster_by,
            semantic_decisions=semantic_decisions,
        )
    elif model_type == "multidomain_factorized_action_value":
        result = evaluate_frozen_factorized_action_value_model(
            model,
            records,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
            cluster_by=args.cluster_by,
            semantic_decisions=semantic_decisions,
        )
    else:
        raise ValueError(f"unsupported model type: {model_type!r}")
    report = {
        "scientific_status": "frozen model evaluation; no target fitting",
        "model_type": model_type,
        "feature_mode": model.get("feature_mode"),
        "evaluation": result,
        "run": {
            "code_revision": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "paths": {name: str(path) for name, path in paths.items()},
            "sha256": hashes,
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed": args.bootstrap_seed,
            "cluster_by": args.cluster_by,
            "required_label_free_features": args.require_label_free_features,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
