#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from beyond_entropy.proxy_outcome_audit import compare_proxy_nll_hardware


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare matched 4090/H800 proxy-NLL scores")
    parser.add_argument("--first-scores", type=Path, required=True)
    parser.add_argument("--first-benchmark", type=Path, required=True)
    parser.add_argument("--second-scores", type=Path, required=True)
    parser.add_argument("--second-benchmark", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-first-scores-sha256")
    parser.add_argument("--expected-second-scores-sha256")
    parser.add_argument("--expected-protocol-sha256")
    parser.add_argument("--expected-decisions", type=int, default=64)
    parser.add_argument("--remaining-gpu-minutes", type=float, required=True)
    parser.add_argument("--code-revision", required=True)
    args = parser.parse_args()
    result = compare_proxy_nll_hardware(
        first_scores=args.first_scores,
        first_benchmark=args.first_benchmark,
        second_scores=args.second_scores,
        second_benchmark=args.second_benchmark,
        protocol=args.protocol,
        output_dir=args.output_dir,
        expected_first_scores_sha256=args.expected_first_scores_sha256,
        expected_second_scores_sha256=args.expected_second_scores_sha256,
        expected_protocol_sha256=args.expected_protocol_sha256,
        expected_decisions=args.expected_decisions,
        remaining_gpu_minutes=args.remaining_gpu_minutes,
        code_revision=args.code_revision,
    )
    print(json.dumps(result["hardware_decision"], sort_keys=True))


if __name__ == "__main__":
    main()
