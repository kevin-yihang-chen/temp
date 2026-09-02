#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from beyond_entropy.infographicvqa_literature_attention_extraction import (
    LITERATURE_ATTENTION_METADATA_KEY,
    augment_literature_attention_where,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract frozen InfographicVQA literature-attention features"
    )
    parser.add_argument("--source-features", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--checkpoint-interval", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = augment_literature_attention_where(
        source_features_path=args.source_features,
        rollouts_path=args.rollouts,
        output_path=args.output,
        model_name_or_path=args.model,
        revision=args.revision,
        device_map=args.device_map,
        dtype=args.dtype,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "completed_decisions": result["metadata"][
                    LITERATURE_ATTENTION_METADATA_KEY
                ]["completed_decisions"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
