from __future__ import annotations

import argparse

from beyond_entropy.predictability_features import (
    extract_predictability_feature_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract frozen pre-action L0--L3 features"
    )
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--dataset-role",
        choices=("train", "validation", "test", "retrospective_smoke"),
        required=True,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention-implementation", default="sdpa")
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=768 * 28 * 28)
    parser.add_argument("--checkpoint-interval", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-missing-prompt-hash", action="store_true")
    parser.add_argument("--allow-online", action="store_true")
    args = parser.parse_args()
    extract_predictability_feature_dataset(
        rollouts_path=args.rollouts,
        manifest_path=args.manifest,
        output_path=args.output,
        dataset_role=args.dataset_role,
        model_name_or_path=args.model,
        revision=args.revision,
        device_map=args.device_map,
        dtype=args.dtype,
        attention_implementation=args.attention_implementation,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        local_files_only=not args.allow_online,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
        require_prompt_hash=not args.allow_missing_prompt_hash,
    )


if __name__ == "__main__":
    main()
