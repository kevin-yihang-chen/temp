#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from beyond_entropy.manifest_audit import audit_manifest_pair


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit frozen development/formal manifests for source leakage"
    )
    parser.add_argument("--development-dir", type=Path, required=True)
    parser.add_argument("--formal-dir", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_manifest_pair(
        args.development_dir,
        args.formal_dir,
        task=args.task,
        expected_revision=args.expected_revision,
    )
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
