#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from beyond_entropy.proxy_outcome_audit import merge_answer_likelihood_shards


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify and merge state-aligned visual-action answer-NLL shards"
    )
    parser.add_argument("--shard", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-shard-count", type=int, default=4)
    parser.add_argument("--expected-decisions", type=int)
    parser.add_argument("--expected-records", type=int)
    parser.add_argument("--expected-sources", type=int)
    args = parser.parse_args()
    result = merge_answer_likelihood_shards(
        shards=args.shard,
        output=args.output,
        expected_shard_count=args.expected_shard_count,
        expected_decisions=args.expected_decisions,
        expected_records=args.expected_records,
        expected_sources=args.expected_sources,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
