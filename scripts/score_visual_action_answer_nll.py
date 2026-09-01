from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from beyond_entropy.answer_likelihood import (
    Qwen25VLAnswerLikelihood,
    score_rollout_answer_likelihood,
)
from beyond_entropy.qwen_backend import merge_runtime_measurements


def _existing_runtime_measurement(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("runtime_measurement")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("answer-likelihood runtime measurement is malformed")
    return value


def _rewrite_provenance(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(
            f"answer-likelihood provenance staging exists: {temporary}"
        )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score frozen visual-action siblings by target-answer NLL"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--expected-rollouts-sha256")
    parser.add_argument("--manifest-limit", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--shard-key",
        choices=("decision_index", "source_id"),
        default="decision_index",
    )
    parser.add_argument("--shard-namespace", default="")
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
    parser.add_argument(
        "--code-revision", default=os.environ.get("BE_CODE_REVISION", "unknown")
    )
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
    provenance_path = args.output.resolve().with_suffix(".provenance.json")
    previous_runtime = _existing_runtime_measurement(provenance_path)
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
        manifest_limit=args.manifest_limit,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
        shard_key=args.shard_key,
        shard_namespace=args.shard_namespace,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
        model=args.model,
        model_revision=args.model_revision,
        measurement_config=scorer.measurement_config(),
        code_revision=args.code_revision,
        scientific_status=args.scientific_status,
    )
    result["runtime_measurement"] = merge_runtime_measurements(
        previous_runtime,
        scorer.runtime_measurement(),
    )
    _rewrite_provenance(provenance_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
