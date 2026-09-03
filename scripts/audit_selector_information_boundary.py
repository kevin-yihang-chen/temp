#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from beyond_entropy.selector_information_boundary import (
    audit_selector_information_boundary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the N4 selector-information-boundary candidate."
    )
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--n3-report", type=Path, required=True)
    parser.add_argument("--expected-n3-sha256", required=True)
    parser.add_argument("--rico-integrity-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    n3_sha256 = _sha256(args.n3_report)
    if n3_sha256 != args.expected_n3_sha256:
        raise ValueError("N4 N3-report SHA-256 mismatch")
    n3 = json.loads(args.n3_report.read_text(encoding="utf-8"))
    if n3.get("decision") != (
        "n3_public_initializer_exists_but_joint_gate_failed_before_download"
    ):
        raise ValueError("N4 requires the frozen N3 joint-gate rejection")

    registry_bytes = args.registry.read_bytes()
    registry = json.loads(registry_bytes)
    report = audit_selector_information_boundary(registry)
    rico_report = json.loads(args.rico_integrity_report.read_text(encoding="utf-8"))
    rico_gates = rico_report.get("gates")
    required_rico_gates = (
        "all_required_images_decode",
        "all_required_jpg_and_json_files_present",
        "allocation_components_cover_all_nonbad_train_images",
    )
    if not isinstance(rico_gates, dict) or not all(
        rico_gates.get(key) is True for key in required_rico_gates
    ):
        raise ValueError("RICO integrity report fails a required availability gate")
    rico_counts = rico_report.get("counts")
    if not isinstance(rico_counts, dict):
        raise ValueError("RICO integrity report lacks counts")
    report["registry_sha256"] = hashlib.sha256(registry_bytes).hexdigest()
    report["n3_report_sha256"] = n3_sha256
    report["n3_decision"] = n3["decision"]
    report["existing_label_free_real_image_seed"] = {
        "report": str(args.rico_integrity_report),
        "report_sha256": _sha256(args.rico_integrity_report),
        "required_availability_gates": {
            key: rico_gates[key] for key in required_rico_gates
        },
        "required_images": rico_counts.get("required_images"),
        "decoded_images": rico_counts.get("decoded_images"),
        "dimension_mismatches_retained_as_qc_risk": rico_counts.get(
            "dimension_mismatches"
        ),
        "role": "future N5 construction audit only; no outcome was read",
    }
    checks = report.get("checks")
    alias_decomposition = report.get("conflicting_alias_decomposition")
    if not isinstance(checks, dict) or not isinstance(alias_decomposition, dict):
        raise ValueError("N4 report lacks machine-readable audit fields")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "checks_passed": sum(value is True for value in checks.values()),
                "checks_total": len(checks),
                "aliasing_regret": alias_decomposition["aliasing_regret"],
                "opened_existing_outcomes": report["opened_existing_outcomes"],
                "authorized_new_gpu_jobs": report["authorized_new_gpu_jobs"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
