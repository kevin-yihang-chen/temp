#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from beyond_entropy.dataset import read_jsonl
from beyond_entropy.decoding_sensitivity import export_capped_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze states whose Qwen outputs reached a decoding token cap"
    )
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--token-cap", type=int, required=True)
    args = parser.parse_args()
    result = export_capped_manifest(
        records=read_jsonl(args.rollouts),
        source_manifest=args.source_manifest,
        source_rollouts=args.rollouts,
        output_manifest=args.output_manifest,
        token_cap=args.token_cap,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
