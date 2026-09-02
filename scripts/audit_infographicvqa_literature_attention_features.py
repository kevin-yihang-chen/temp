#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from beyond_entropy.dataset import read_jsonl
from beyond_entropy.infographicvqa_literature_attention_merge import sha256_file
from beyond_entropy.infographicvqa_literature_attention_where import (
    assemble_literature_attention_where_features,
)
from beyond_entropy.qwen_semantic import load_semantic_feature_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit outcome-free InfographicVQA literature-attention features"
    )
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--expected-features-sha256", required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--expected-rollouts-sha256", required=True)
    parser.add_argument("--expected-code-revision", required=True)
    parser.add_argument("--expected-model-revision", required=True)
    parser.add_argument("--source-features-sha256", required=True)
    args = parser.parse_args()
    features = args.features.resolve()
    rollouts = args.rollouts.resolve()
    features_sha = sha256_file(features)
    rollouts_sha = sha256_file(rollouts)
    if features_sha != args.expected_features_sha256:
        raise ValueError("literature attention feature SHA-256 changed")
    if rollouts_sha != args.expected_rollouts_sha256:
        raise ValueError("literature attention rollout SHA-256 changed")
    _, audit = assemble_literature_attention_where_features(
        read_jsonl(rollouts),
        load_semantic_feature_dataset(features),
        expected_code_revision=args.expected_code_revision,
        expected_model_revision=args.expected_model_revision,
        expected_source_features_sha256=args.source_features_sha256,
        expected_rollouts_sha256=rollouts_sha,
    )
    audit["features"] = {"path": str(features), "sha256": features_sha}
    audit["rollouts"] = {"path": str(rollouts), "sha256": rollouts_sha}
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
