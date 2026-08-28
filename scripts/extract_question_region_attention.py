#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from beyond_entropy.attention_features import augment_question_region_attention


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract frozen question-to-region attention features"
    )
    parser.add_argument("--source-features", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--top-layers", type=int, default=4)
    parser.add_argument("--checkpoint-interval", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = augment_question_region_attention(
        source_features_path=args.source_features,
        rollouts_path=args.rollouts,
        output_path=args.output,
        model_name_or_path=args.model,
        revision=args.revision,
        device_map=args.device_map,
        dtype=args.dtype,
        top_layers=args.top_layers,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "completed_decisions": result["metadata"][
                    "question_region_attention"
                ]["completed_decisions"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
