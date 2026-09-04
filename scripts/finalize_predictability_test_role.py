from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from beyond_entropy.predictability_matrix_artifacts import (
    TEST_ACCESS_SCHEMA,
    atomic_json_write_exclusive,
    load_hashed_json,
    load_role_artifacts,
    sha256_file,
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate and seal one role inside the one-shot test transaction"
    )
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--rollout-provenance", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--access-ledger", required=True)
    parser.add_argument("--expected-access-ledger-sha256", required=True)
    parser.add_argument("--test-transaction-plan-sha256", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--expected-states", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    # The durable access record is verified before any test role artifact.
    _, access = load_hashed_json(
        args.access_ledger,
        expected_sha256=args.expected_access_ledger_sha256,
        schema=TEST_ACCESS_SCHEMA,
    )
    if (
        access.get("status") != "started"
        or access.get("test_transaction_plan_sha256")
        != args.test_transaction_plan_sha256
        or access.get("code_revision") != args.code_revision
        or access.get("protocol_sha256") != args.expected_protocol_sha256
        or access.get("test_artifacts_accessed_before_this_record") is not False
        or access.get("automatic_retry_allowed") is not False
    ):
        raise ValueError("test role access ledger binding differs")
    protocol_path = Path(args.protocol).resolve()
    if sha256_file(protocol_path) != args.expected_protocol_sha256:
        raise ValueError("formal test role protocol SHA-256 mismatch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    loaded = load_role_artifacts(
        {
            "manifest": {
                "path": str(Path(args.manifest).resolve()),
                "sha256": sha256_file(args.manifest),
            },
            "rollouts": {
                "path": str(Path(args.rollouts).resolve()),
                "sha256": sha256_file(args.rollouts),
            },
            "rollout_provenance": {
                "path": str(Path(args.rollout_provenance).resolve()),
                "sha256": sha256_file(args.rollout_provenance),
            },
            "features": {
                "path": str(Path(args.features).resolve()),
                "sha256": sha256_file(args.features),
            },
        },
        benchmark=args.benchmark,
        role="test",
        code_revision=args.code_revision,
        protocol=protocol,
    )
    if len(loaded.examples) != args.expected_states:
        raise ValueError("formal test role state count differs from frozen plan")
    report = {
        "schema": "predictability_formal_test_role_v1",
        "passed": True,
        "benchmark": args.benchmark,
        "role": "test",
        "states": len(loaded.examples),
        "sibling_records": len(loaded.siblings),
        "post_action_examples": len(loaded.post_action_examples),
        "code_revision": args.code_revision,
        "protocol": str(protocol_path),
        "protocol_sha256": args.expected_protocol_sha256,
        "test_transaction_plan_sha256": args.test_transaction_plan_sha256,
        "access_ledger": str(Path(args.access_ledger).resolve()),
        "access_ledger_sha256": args.expected_access_ledger_sha256,
        "artifacts": loaded.hashes,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json_write_exclusive(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
