#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from beyond_entropy.dataset import read_jsonl
from beyond_entropy.factorized_evaluation import (
    evaluate_factorized_risk_controlled_policy,
)
from scripts.verify_screenqa_calibration_result import verify_result
from scripts.verify_screenqa_formal_manifest import verify_manifest
from scripts.verify_screenqa_formal_rollouts import verify_rollouts


EXPECTED_STATES = 14672
EXPECTED_SOURCES = 1471
BOOTSTRAP_RESAMPLES = 20000
BOOTSTRAP_CONFIDENCE = 0.975
BOOTSTRAP_SEED = 20260831


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mapping(path: Path, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"ScreenQA formal {name} must be a JSON object")
    return payload


def _require(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"ScreenQA formal evaluation mismatch for {name}")


def _git_revision(repo_dir: Path) -> str:
    status = subprocess.run(
        ["git", "-C", str(repo_dir), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("tracked worktree must be clean before ScreenQA formal evaluation")
    return subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen ScreenQA factorized policy exactly once"
    )
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--expected-candidate-bundle-sha256", required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--expected-calibration-bundle-sha256", required=True)
    parser.add_argument("--formal-manifest-dir", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-manifest-audit-sha256", required=True)
    parser.add_argument("--formal-run-root", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--expected-rollouts-sha256", required=True)
    parser.add_argument("--merge-audit", type=Path, required=True)
    parser.add_argument("--expected-merge-audit-sha256", required=True)
    parser.add_argument("--rollout-audit", type=Path, required=True)
    parser.add_argument("--expected-rollout-audit-sha256", required=True)
    parser.add_argument("--bank-completion", type=Path, required=True)
    parser.add_argument("--expected-bank-completion-sha256", required=True)
    parser.add_argument("--expected-code-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, required=True)
    parser.add_argument("--bootstrap-confidence", type=float, required=True)
    parser.add_argument("--bootstrap-seed", type=int, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite ScreenQA one-shot formal report: {args.output}"
        )
    _require(args.bootstrap_resamples, BOOTSTRAP_RESAMPLES, "bootstrap resamples")
    _require(args.bootstrap_confidence, BOOTSTRAP_CONFIDENCE, "bootstrap confidence")
    _require(args.bootstrap_seed, BOOTSTRAP_SEED, "bootstrap seed")
    repo_dir = Path(__file__).resolve().parents[1]
    code_revision = _git_revision(repo_dir)
    _require(code_revision, args.expected_code_revision, "code revision")

    candidate_dir = args.candidate_dir.resolve()
    calibration_dir = args.calibration_dir.resolve()
    manifest_dir = args.formal_manifest_dir.resolve()
    run_root = args.formal_run_root.resolve()
    rollouts = args.rollouts.resolve()
    merge_audit = args.merge_audit.resolve()
    rollout_audit = args.rollout_audit.resolve()
    bank_completion = args.bank_completion.resolve()
    manifest_info = verify_manifest(
        manifest_dir,
        candidate_dir=candidate_dir,
        expected_candidate_bundle_sha256=args.expected_candidate_bundle_sha256,
        calibration_dir=calibration_dir,
        expected_calibration_bundle_sha256=(
            args.expected_calibration_bundle_sha256
        ),
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_audit_sha256=args.expected_manifest_audit_sha256,
    )
    calibration_info = verify_result(calibration_dir, candidate_dir)
    _require(calibration_info["formal_allowed"], True, "formal gate")
    _require(
        calibration_info["bundle_sha256"],
        args.expected_calibration_bundle_sha256,
        "calibration bundle",
    )
    if sha256_file(rollout_audit) != args.expected_rollout_audit_sha256:
        raise ValueError("ScreenQA formal rollout-audit SHA-256 mismatch")
    if sha256_file(bank_completion) != args.expected_bank_completion_sha256:
        raise ValueError("ScreenQA formal bank-completion SHA-256 mismatch")
    fresh_rollout_audit_path = args.output.parent / "rollouts.reverified.json"
    rollout_info = verify_rollouts(
        formal_manifest_dir=manifest_dir,
        candidate_dir=candidate_dir,
        expected_candidate_bundle_sha256=args.expected_candidate_bundle_sha256,
        calibration_dir=calibration_dir,
        expected_calibration_bundle_sha256=(
            args.expected_calibration_bundle_sha256
        ),
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_manifest_audit_sha256=args.expected_manifest_audit_sha256,
        run_root=run_root,
        rollouts=rollouts,
        expected_rollouts_sha256=args.expected_rollouts_sha256,
        merge_audit=merge_audit,
        expected_merge_audit_sha256=args.expected_merge_audit_sha256,
        expected_bank_code_revision=args.expected_code_revision,
        output=fresh_rollout_audit_path,
        resume=True,
    )
    stored_rollout_audit = _load_mapping(rollout_audit, "rollout audit")
    for key in (
        "passed",
        "scientific_status",
        "one_shot_formal_bank_complete",
        "rollouts_sha256",
        "merge_audit_sha256",
        "manifest_sha256",
        "manifest_audit_sha256",
        "candidate_bundle_sha256",
        "calibration_bundle_sha256",
        "selected_threshold",
        "bank_code_revision",
        "states",
        "records",
        "source_components",
        "one_shot_completed_shards",
        "completion_marker_sha256s",
        "formal_outcomes_used_for_tuning",
    ):
        _require(stored_rollout_audit.get(key), rollout_info.get(key), f"audit {key}")
    bank = _load_mapping(bank_completion, "bank completion")
    expected_bank = {
        "passed": True,
        "one_shot_formal_bank_complete": True,
        "formal_outcomes_used_for_tuning": False,
        "states": EXPECTED_STATES,
        "records": EXPECTED_STATES * 5,
        "completed_shards": 4,
        "rollouts_sha256": args.expected_rollouts_sha256,
        "merge_audit_sha256": args.expected_merge_audit_sha256,
        "rollout_audit_sha256": args.expected_rollout_audit_sha256,
        "manifest_sha256": args.expected_manifest_sha256,
        "manifest_audit_sha256": args.expected_manifest_audit_sha256,
        "candidate_bundle_sha256": args.expected_candidate_bundle_sha256,
        "calibration_bundle_sha256": args.expected_calibration_bundle_sha256,
        "code_revision": args.expected_code_revision,
    }
    for key, expected_value in expected_bank.items():
        _require(bank.get(key), expected_value, f"bank completion {key}")

    model_path = calibration_dir / "model.json"
    model = _load_mapping(model_path, "calibrated model")
    _require(model.get("domains"), ["screenqa"], "model domain")
    if model.get("feature_mode") not in {
        "context-geometry",
        "spatial-context-geometry",
    }:
        raise ValueError("ScreenQA formal model feature mode is not preregistered")
    _require(
        float(model["threshold"]),
        float(manifest_info["selected_threshold"]),
        "calibrated threshold",
    )
    records = read_jsonl(rollouts)
    report = evaluate_factorized_risk_controlled_policy(
        model,
        records,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_confidence=args.bootstrap_confidence,
        bootstrap_seed=args.bootstrap_seed,
    )
    _require(report.get("n_sources"), EXPECTED_SOURCES, "report sources")
    _require(report.get("n_decisions"), EXPECTED_STATES, "report decisions")
    _require(
        report.get("threshold"),
        float(manifest_info["selected_threshold"]),
        "report threshold",
    )
    pass_rule = report.get("pass_rule")
    if not isinstance(pass_rule, dict):
        raise ValueError("ScreenQA formal report lacks its pass rule")
    pass_rule["threshold_matches_calibration_choice"] = True
    pass_rule["all_frozen_hashes_and_identity_audits_match"] = True
    report["passed"] = all(bool(value) for value in pass_rule.values())
    report["scientific_status"] = (
        "one-shot ScreenQA evaluation of the frozen low-capacity factorized policy"
    )
    report["run"] = {
        "code_revision": code_revision,
        "formal_outcomes_used": True,
        "no_target_derived_tuning": True,
        "candidate_bundle_sha256": args.expected_candidate_bundle_sha256,
        "calibration_bundle_sha256": args.expected_calibration_bundle_sha256,
        "calibrated_model": str(model_path),
        "calibrated_model_sha256": calibration_info["model_sha256"],
        "manifest": str((manifest_dir / "manifest.jsonl").resolve()),
        "manifest_sha256": args.expected_manifest_sha256,
        "manifest_audit_sha256": args.expected_manifest_audit_sha256,
        "rollouts": str(rollouts),
        "rollouts_sha256": args.expected_rollouts_sha256,
        "merge_audit_sha256": args.expected_merge_audit_sha256,
        "rollout_audit_sha256": args.expected_rollout_audit_sha256,
        "fresh_rollout_audit_sha256": sha256_file(fresh_rollout_audit_path),
        "bank_completion_sha256": args.expected_bank_completion_sha256,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_confidence": BOOTSTRAP_CONFIDENCE,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "source_utility": report["source_balanced"]["utility"],
                "source_utility_ci_low": report["source_bootstrap"]["metrics"][
                    "utility"
                ]["ci_low"],
                "question_utility": report["question_weighted"]["utility"],
                "source_call_rate": report["source_balanced"]["call"],
                "n_sources": report["n_sources"],
                "n_decisions": report["n_decisions"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
