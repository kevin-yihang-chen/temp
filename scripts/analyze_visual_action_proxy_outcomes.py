#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from beyond_entropy.proxy_outcome_audit import analyze_proxy_outcomes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen visual-action proxy-to-outcome audit"
    )
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--implementation-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-scores-sha256")
    parser.add_argument("--expected-protocol-sha256")
    parser.add_argument("--expected-implementation-contract-sha256")
    parser.add_argument("--expected-decisions", type=int)
    parser.add_argument("--expected-sources", type=int)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260831)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--code-revision", required=True)
    args = parser.parse_args()
    result = analyze_proxy_outcomes(
        scores=args.scores,
        protocol=args.protocol,
        implementation_contract=args.implementation_contract,
        output_dir=args.output_dir,
        expected_scores_sha256=args.expected_scores_sha256,
        expected_protocol_sha256=args.expected_protocol_sha256,
        expected_implementation_contract_sha256=(
            args.expected_implementation_contract_sha256
        ),
        expected_decisions=args.expected_decisions,
        expected_sources=args.expected_sources,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_confidence=args.bootstrap_confidence,
        code_revision=args.code_revision,
    )
    print(json.dumps(result["population"], sort_keys=True))


if __name__ == "__main__":
    main()
