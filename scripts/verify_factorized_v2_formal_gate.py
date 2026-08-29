from __future__ import annotations

import argparse
import json
from pathlib import Path

from beyond_entropy.factorized_formal import validate_materialized_formal_gate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the materialized factorized-v2 formal gate"
    )
    parser.add_argument("--policy-freeze", type=Path, required=True)
    parser.add_argument("--expected-policy-freeze-sha256", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--manifest-provenance", type=Path, required=True)
    parser.add_argument("--expected-manifest-provenance-sha256", required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--expected-audit-sha256", required=True)
    args = parser.parse_args()
    freeze = validate_materialized_formal_gate(
        policy_freeze_path=args.policy_freeze,
        expected_policy_freeze_sha256=args.expected_policy_freeze_sha256,
        model_path=args.model,
        expected_model_sha256=args.expected_model_sha256,
        manifest_path=args.manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
        manifest_provenance_path=args.manifest_provenance,
        expected_manifest_provenance_sha256=(
            args.expected_manifest_provenance_sha256
        ),
        audit_path=args.audit,
        expected_audit_sha256=args.expected_audit_sha256,
    )
    print(
        json.dumps(
            {
                "passed": True,
                "policy_freeze_sha256": args.expected_policy_freeze_sha256,
                "model_sha256": args.expected_model_sha256,
                "manifest_sha256": args.expected_manifest_sha256,
                "formal_sources": freeze["formal_test"]["allocated_sources"],
                "formal_outcomes_used": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
