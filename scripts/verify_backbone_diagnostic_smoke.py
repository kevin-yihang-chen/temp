#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from beyond_entropy.backbone_smoke import verify_backbone_engineering_smoke


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify an endpoint-blind backbone rollout/NLL engineering smoke."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--rollout-provenance", type=Path, required=True)
    parser.add_argument("--rollout-resume-audit", type=Path, required=True)
    parser.add_argument("--answer-nll", type=Path, required=True)
    parser.add_argument("--answer-nll-provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-decisions", type=int, default=32)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--expected-model-revision", required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--expected-code-revision", required=True)
    parser.add_argument("--rollout-seconds", type=float, required=True)
    parser.add_argument("--answer-nll-seconds", type=float, required=True)
    args = parser.parse_args()
    result = verify_backbone_engineering_smoke(
        manifest=args.manifest,
        rollouts=args.rollouts,
        rollout_provenance=args.rollout_provenance,
        rollout_resume_audit=args.rollout_resume_audit,
        answer_nll=args.answer_nll,
        answer_nll_provenance=args.answer_nll_provenance,
        output=args.output,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_decisions=args.expected_decisions,
        expected_model=args.expected_model,
        expected_model_revision=args.expected_model_revision,
        expected_gpu_name=args.expected_gpu_name,
        expected_code_revision=args.expected_code_revision,
        rollout_seconds=args.rollout_seconds,
        answer_nll_seconds=args.answer_nll_seconds,
    )
    print(json.dumps(result["timing_seconds"], sort_keys=True))


if __name__ == "__main__":
    main()
