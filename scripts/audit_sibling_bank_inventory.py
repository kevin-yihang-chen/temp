#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from beyond_entropy.sibling_bank_inventory import (
    build_n1_inventory,
    default_n1_bank_specs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit existing sibling banks for the N1 regret benchmark gate."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    report = build_n1_inventory(default_n1_bank_specs(repo_root), repo_root=repo_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "checks_passed": sum(report["gate_checks"].values()),
                "checks_total": len(report["gate_checks"]),
                "main_records": report["summary"]["main_records"],
                "main_decisions": report["summary"]["main_decisions"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
