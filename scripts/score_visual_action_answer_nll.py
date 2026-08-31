from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from beyond_entropy.answer_likelihood import (
    Qwen25VLAnswerLikelihood,
    score_rollout_answer_likelihood,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score frozen visual-action siblings by target-answer NLL"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--expected-rollouts-sha256")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--checkpoint-interval", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention-implementation", default="sdpa")
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=768 * 28 * 28)
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--code-revision", default=os.environ.get("BE_CODE_REVISION", "unknown"))
    parser.add_argument(
        "--scientific-status",
        default=(
            "opened-development proxy-to-outcome audit only; not a replacement "
            "ScreenQA ranker candidate"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scorer = Qwen25VLAnswerLikelihood(
        args.model,
        revision=args.model_revision,
        device_map=args.device_map,
        dtype=args.dtype,
        attention_implementation=args.attention_implementation,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        local_files_only=not args.allow_download,
        system_prompt=args.system_prompt,
    )
    result = score_rollout_answer_likelihood(
        manifest=args.manifest,
        rollouts=args.rollouts,
        output=args.output,
        score_request=scorer.score,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_rollouts_sha256=args.expected_rollouts_sha256,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
        model=args.model,
        model_revision=args.model_revision,
        code_revision=args.code_revision,
        scientific_status=args.scientific_status,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
