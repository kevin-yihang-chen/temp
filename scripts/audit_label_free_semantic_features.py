#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from beyond_entropy.dataset import read_jsonl
from beyond_entropy.qwen_semantic import (
    SEMANTIC_OUTCOME_FIELDS,
    load_semantic_feature_dataset,
    validate_semantic_feature_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that a frozen semantic feature file contains no outcomes"
    )
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    args = parser.parse_args()

    records = read_jsonl(args.rollouts)
    payload = load_semantic_feature_dataset(args.features)
    validate_semantic_feature_dataset(
        payload,
        records,
        require_outcomes=False,
    )
    present = sorted(
        {
            field
            for decision in payload["decisions"]
            for field in SEMANTIC_OUTCOME_FIELDS & set(decision)
        }
    )
    if present:
        raise ValueError(f"forbidden outcome fields are present: {present}")
    print(
        json.dumps(
            {
                "features": str(args.features.resolve()),
                "features_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
                "rollouts": str(args.rollouts.resolve()),
                "rollouts_sha256": hashlib.sha256(args.rollouts.read_bytes()).hexdigest(),
                "decisions": len(payload["decisions"]),
                "outcome_fields_present": present,
                "outcomes_included_metadata": payload["metadata"].get(
                    "outcomes_included"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
