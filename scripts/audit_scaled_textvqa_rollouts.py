from __future__ import annotations

import argparse
import json
from pathlib import Path

from beyond_entropy.rollout_audit import audit_sibling_rollout_bank


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit one scaled TextVQA rollout bank")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--expected-states", type=int, required=True)
    parser.add_argument("--expected-model-revision", required=True)
    parser.add_argument("--expected-scientific-status", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = audit_sibling_rollout_bank(
        args.manifest,
        args.rollouts,
        expected_manifest_sha256=args.manifest_sha256,
        expected_states=args.expected_states,
        expected_candidate_count=4,
        expected_model_revision=args.expected_model_revision,
        expected_scientific_status=args.expected_scientific_status,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
