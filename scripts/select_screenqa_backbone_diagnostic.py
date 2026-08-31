#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from beyond_entropy.backbone_diagnostic import select_source_disjoint_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select one outcome-blind state from each hash-ranked ScreenQA source."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--source-count", type=int, default=512)
    parser.add_argument(
        "--namespace", default="beyond-entropy-screenqa-backbone-7b-v1"
    )
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--code-revision", required=True)
    args = parser.parse_args()
    result = select_source_disjoint_manifest(
        manifest=args.manifest,
        output=args.output,
        report=args.report,
        expected_manifest_sha256=args.expected_manifest_sha256,
        source_count=args.source_count,
        namespace=args.namespace,
        seed=args.seed,
        code_revision=args.code_revision,
    )
    print(
        json.dumps(
            {
                "manifest": result["output"]["manifest"],
                "manifest_sha256": result["output"]["manifest_sha256"],
                "sources": result["output"]["sources"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
