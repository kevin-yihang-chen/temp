from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from beyond_entropy.predictability_matrix_artifacts import (
    atomic_json_write_exclusive,
    load_role_artifacts,
    sha256_file,
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate and seal one formal predictability development role"
    )
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--role", choices=("train", "validation"), required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--rollout-provenance", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--expected-states", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    protocol_path = Path(args.protocol).resolve()
    if sha256_file(protocol_path) != args.expected_protocol_sha256:
        raise ValueError("formal role protocol SHA-256 mismatch")
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
        role=args.role,
        code_revision=args.code_revision,
        protocol=protocol,
    )
    if len(loaded.examples) != args.expected_states:
        raise ValueError("formal role state count differs from frozen allocation")
    report = {
        "schema": "predictability_formal_development_role_v1",
        "passed": True,
        "scientific_status": "development_feature_generation_not_test_result",
        "benchmark": args.benchmark,
        "role": args.role,
        "states": len(loaded.examples),
        "sibling_records": len(loaded.siblings),
        "post_action_examples": len(loaded.post_action_examples),
        "code_revision": args.code_revision,
        "protocol": str(protocol_path),
        "protocol_sha256": args.expected_protocol_sha256,
        "artifacts": loaded.hashes,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json_write_exclusive(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
