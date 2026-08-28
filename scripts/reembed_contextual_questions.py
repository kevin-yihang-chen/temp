from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from beyond_entropy.question_reembed import (
    reembed_contextual_questions,
    reembed_multimodal_questions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-embed frozen semantic questions")
    parser.add_argument("--source-features", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention-implementation", default="sdpa")
    parser.add_argument(
        "--mode",
        choices=("text-only", "multimodal-original"),
        default="text-only",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint-interval", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    reembed = (
        reembed_multimodal_questions
        if args.mode == "multimodal-original"
        else reembed_contextual_questions
    )
    result = reembed(
        source_features_path=args.source_features,
        rollouts_path=args.rollouts,
        output_path=args.output,
        model_name_or_path=args.model,
        revision=args.revision,
        device_map=args.device_map,
        dtype=args.dtype,
        attention_implementation=args.attention_implementation,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "completed_decisions": result["metadata"]["question_reembedding"][
                    "completed_decisions"
                ],
                "question_feature_mode": result["metadata"]["question_feature_mode"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
