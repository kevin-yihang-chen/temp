#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from beyond_entropy.tool_checkpoint_novelty_audit import (
    audit_checkpoint_and_novelty,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the N3 public checkpoint and independent-novelty gate."
    )
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--n2-report", type=Path, required=True)
    parser.add_argument("--expected-n2-sha256", required=True)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--old-vtool-repo", type=Path, required=True)
    parser.add_argument("--training-v2-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _license_is_apache_2(path: Path) -> bool:
    license_text = (path / "LICENSE").read_text(encoding="utf-8")
    return "Apache License" in license_text and "Version 2.0" in license_text


def _cached_model_ids(cache_root: Path, model_ids: list[str]) -> list[str]:
    cached: list[str] = []
    for model_id in model_ids:
        cache_name = "models--" + model_id.replace("/", "--")
        if (cache_root / cache_name).is_dir():
            cached.append(model_id)
    return cached


def main() -> None:
    args = parse_args()
    n2_sha256 = hashlib.sha256(args.n2_report.read_bytes()).hexdigest()
    if n2_sha256 != args.expected_n2_sha256:
        raise ValueError("N3 N2-report SHA-256 mismatch")
    n2 = json.loads(args.n2_report.read_text(encoding="utf-8"))
    if n2.get("decision") != (
        "n2_additive_causal_regret_candidate_not_identified_and_not_novel"
    ):
        raise ValueError("N3 requires the frozen N2 rejection decision")

    registry_bytes = args.registry.read_bytes()
    registry = json.loads(registry_bytes)
    model_ids = [str(item["model_id"]) for item in registry["checkpoint_candidates"]]
    cached_model_ids = _cached_model_ids(args.hf_cache, model_ids)
    report = audit_checkpoint_and_novelty(
        registry, local_cache_model_ids=cached_model_ids
    )
    report["registry_sha256"] = hashlib.sha256(registry_bytes).hexdigest()
    report["n2_report_sha256"] = n2_sha256
    report["n2_decision"] = n2["decision"]
    report["local_code_repositories"] = {
        "old_vtool": {
            "revision": _git_revision(args.old_vtool_repo),
            "apache_2_license": _license_is_apache_2(args.old_vtool_repo),
        },
        "training_v2": {
            "revision": _git_revision(args.training_v2_repo),
            "apache_2_license": _license_is_apache_2(args.training_v2_repo),
        },
    }
    report["local_cache_candidate_count"] = len(cached_model_ids)
    report["resource_audit_status"] = (
        "not_run_because_joint_baseline_and_novelty_gate_failed"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "baseline_checks_passed": sum(report["baseline_checks"].values()),
                "baseline_checks_total": len(report["baseline_checks"]),
                "novelty_checks_passed": sum(report["novelty_checks"].values()),
                "novelty_checks_total": len(report["novelty_checks"]),
                "selected_candidate": report[
                    "selected_candidate_if_scientifically_authorized"
                ],
                "downloaded_checkpoint_bytes": report[
                    "downloaded_checkpoint_bytes"
                ],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
