from __future__ import annotations

import argparse
import json
from typing import Sequence

from beyond_entropy.predictability_test_transaction import start_test_transaction


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Durably start the one-shot test transaction before test access"
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    result = start_test_transaction(
        args.plan,
        expected_sha256=args.expected_plan_sha256,
        repo_root=args.repo_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
