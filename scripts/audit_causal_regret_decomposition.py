#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from beyond_entropy.causal_regret_decomposition_audit import (
    audit_causal_regret_candidate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit identifiability and additivity of the N2 regret candidate."
    )
    parser.add_argument("--n1-report", type=Path, required=True)
    parser.add_argument("--expected-n1-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n1_sha256 = hashlib.sha256(args.n1_report.read_bytes()).hexdigest()
    if n1_sha256 != args.expected_n1_sha256:
        raise ValueError("N2 N1-report SHA-256 mismatch")
    n1 = json.loads(args.n1_report.read_text(encoding="utf-8"))
    expected_n1_decision = (
        "n1_existing_assets_insufficient_for_top_tier_regret_benchmark"
    )
    if n1.get("decision") != expected_n1_decision:
        raise ValueError("N2 requires the frozen N1 insufficiency decision")
    audit = audit_causal_regret_candidate()
    report = audit.to_dict()
    report["n1_report"] = str(args.n1_report)
    report["n1_report_sha256"] = n1_sha256
    report["n1_decision"] = n1["decision"]
    report["literature_sources"] = {
        "the_illusion_of_visual_tool_use": "https://arxiv.org/abs/2608.06270",
        "gapsight": "https://arxiv.org/abs/2608.21762",
    }
    report["resource_audit_status"] = (
        "not_run_because_identifiability_and_novelty_gate_failed"
    )
    report["authorized_new_gpu_jobs"] = 0
    report["authorized_new_checkpoints"] = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "decision": audit.decision,
                "checks_passed": sum(audit.checks.values()),
                "checks_total": len(audit.checks),
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
