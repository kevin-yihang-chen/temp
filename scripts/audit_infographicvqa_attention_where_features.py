#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from beyond_entropy.dataset import read_jsonl
from beyond_entropy.infographicvqa_attention_where import (
    assemble_attention_where_features,
)
from beyond_entropy.qwen_semantic import load_semantic_feature_dataset


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit outcome-free InfographicVQA raw attention crop scores"
    )
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--expected-features-sha256", required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--expected-rollouts-sha256", required=True)
    parser.add_argument("--expected-code-revision", required=True)
    parser.add_argument("--expected-model-revision", required=True)
    parser.add_argument("--source-features-sha256", required=True)
    args = parser.parse_args()

    features_path = args.features.resolve()
    rollouts_path = args.rollouts.resolve()
    features_sha256 = sha256_file(features_path)
    rollouts_sha256 = sha256_file(rollouts_path)
    if features_sha256 != args.expected_features_sha256:
        raise ValueError("attention-where feature SHA-256 changed")
    if rollouts_sha256 != args.expected_rollouts_sha256:
        raise ValueError("attention-where rollout SHA-256 changed")
    _, audit = assemble_attention_where_features(
        read_jsonl(rollouts_path),
        load_semantic_feature_dataset(features_path),
        expected_code_revision=args.expected_code_revision,
        expected_model_revision=args.expected_model_revision,
        expected_source_features_sha256=args.source_features_sha256,
        expected_rollouts_sha256=rollouts_sha256,
    )
    audit["features"] = {
        "path": str(features_path),
        "sha256": features_sha256,
    }
    audit["rollouts"] = {
        "path": str(rollouts_path),
        "sha256": rollouts_sha256,
    }
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
