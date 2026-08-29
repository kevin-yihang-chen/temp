from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from beyond_entropy.dataset import read_jsonl
from beyond_entropy.docvqa_formal import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    FORMAL_SCIENTIFIC_STATUS,
    FORMAL_SOURCES,
    MODEL_REVISION,
    check_hash,
    validate_materialized_formal_gate,
)
from beyond_entropy.factorized_evaluation import (
    evaluate_factorized_risk_controlled_policy,
)
from beyond_entropy.qwen_semantic import (
    load_semantic_feature_dataset,
    validate_semantic_feature_dataset,
)
from beyond_entropy.rollout_audit import audit_sibling_rollout_bank


def _load_mapping(path: Path, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"DocVQA formal {name} must be a JSON object")
    return payload


def _require(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"DocVQA formal evaluation mismatch for {name}")


def _git_revision(repo_dir: Path) -> str:
    status = subprocess.run(
        ["git", "-C", str(repo_dir), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("tracked worktree must be clean before DocVQA formal evaluation")
    return subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen DocVQA-train factorized-v2 policy once"
    )
    parser.add_argument("--policy-freeze", type=Path, required=True)
    parser.add_argument("--expected-policy-freeze-sha256", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--manifest-provenance", type=Path, required=True)
    parser.add_argument("--expected-manifest-provenance-sha256", required=True)
    parser.add_argument("--formal-audit", type=Path, required=True)
    parser.add_argument("--expected-formal-audit-sha256", required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--expected-rollouts-sha256", required=True)
    parser.add_argument("--rollout-audit", type=Path, required=True)
    parser.add_argument("--expected-rollout-audit-sha256", required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--expected-features-sha256", required=True)
    parser.add_argument("--expected-states", type=int, required=True)
    parser.add_argument(
        "--expected-scientific-status",
        default=FORMAL_SCIENTIFIC_STATUS,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-confidence", type=float, default=BOOTSTRAP_CONFIDENCE)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    policy_freeze_path = args.policy_freeze.resolve()
    model_path = args.model.resolve()
    manifest_path = args.manifest.resolve()
    manifest_provenance_path = args.manifest_provenance.resolve()
    formal_audit_path = args.formal_audit.resolve()
    rollouts_path = args.rollouts.resolve()
    rollout_audit_path = args.rollout_audit.resolve()
    features_path = args.features.resolve()
    output_path = args.output.resolve()
    freeze = validate_materialized_formal_gate(
        policy_freeze_path=policy_freeze_path,
        expected_policy_freeze_sha256=args.expected_policy_freeze_sha256,
        model_path=model_path,
        expected_model_sha256=args.expected_model_sha256,
        manifest_path=manifest_path,
        expected_manifest_sha256=args.expected_manifest_sha256,
        manifest_provenance_path=manifest_provenance_path,
        expected_manifest_provenance_sha256=(
            args.expected_manifest_provenance_sha256
        ),
        audit_path=formal_audit_path,
        expected_audit_sha256=args.expected_formal_audit_sha256,
    )
    _require(args.bootstrap_resamples, BOOTSTRAP_RESAMPLES, "bootstrap resamples")
    _require(args.bootstrap_confidence, BOOTSTRAP_CONFIDENCE, "bootstrap confidence")
    _require(args.bootstrap_seed, BOOTSTRAP_SEED, "bootstrap seed")
    _require(
        args.expected_scientific_status,
        FORMAL_SCIENTIFIC_STATUS,
        "scientific status",
    )
    if args.expected_states < FORMAL_SOURCES:
        raise ValueError("DocVQA formal state count is smaller than its source count")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite DocVQA formal report: {output_path}")

    check_hash(rollouts_path, args.expected_rollouts_sha256, "formal rollouts")
    check_hash(
        rollout_audit_path,
        args.expected_rollout_audit_sha256,
        "formal rollout audit",
    )
    check_hash(features_path, args.expected_features_sha256, "formal features")
    fresh_rollout_audit = audit_sibling_rollout_bank(
        manifest_path,
        rollouts_path,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_states=args.expected_states,
        expected_candidate_count=4,
        expected_model_revision=MODEL_REVISION,
        expected_scientific_status=args.expected_scientific_status,
    )
    _require(
        fresh_rollout_audit.get("rollouts_sha256"),
        args.expected_rollouts_sha256,
        "fresh rollout hash",
    )
    stored_rollout_audit = _load_mapping(rollout_audit_path, "rollout audit")
    expected_stored = {
        "passed": True,
        "policy_freeze_sha256": args.expected_policy_freeze_sha256,
        "model_sha256": args.expected_model_sha256,
        "manifest_sha256": args.expected_manifest_sha256,
        "manifest_provenance_sha256": args.expected_manifest_provenance_sha256,
        "formal_audit_sha256": args.expected_formal_audit_sha256,
        "rollouts_sha256": args.expected_rollouts_sha256,
        "protocol_sha256": freeze["artifacts"]["protocol"]["sha256"],
        "code_revision": freeze["code_revision"],
        "model_revision": MODEL_REVISION,
        "scientific_status": FORMAL_SCIENTIFIC_STATUS,
        "states": args.expected_states,
        "records": args.expected_states * 5,
        "unique_sources": FORMAL_SOURCES,
        "unique_images": FORMAL_SOURCES,
        "candidate_count": 4,
        "answer_records": args.expected_states,
        "zoom_records": args.expected_states * 4,
        "formal_outcomes_collected": True,
        "formal_outcomes_used_for_tuning": False,
    }
    for name, expected in expected_stored.items():
        _require(stored_rollout_audit.get(name), expected, f"rollout audit {name}")
    for name in (
        "manifest_sha256",
        "model_revision",
        "scientific_status",
        "states",
        "records",
        "unique_sources",
        "unique_images",
        "candidate_count",
        "answer_records",
        "zoom_records",
        "rollouts_sha256",
    ):
        _require(
            stored_rollout_audit.get(name),
            fresh_rollout_audit.get(name),
            f"stored/fresh rollout audit {name}",
        )

    records = read_jsonl(rollouts_path)
    features = load_semantic_feature_dataset(features_path)
    validate_semantic_feature_dataset(features, records)
    metadata = features.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("DocVQA formal feature metadata is missing")
    _require(metadata.get("outcomes_included"), False, "feature outcome exclusion")
    _require(metadata.get("rollouts_sha256"), args.expected_rollouts_sha256, "feature rollouts")
    _require(metadata.get("model_revision"), MODEL_REVISION, "feature model revision")
    _require(len(features["decisions"]), args.expected_states, "feature decisions")
    semantic_decisions = {
        (str(decision["state_id"]), str(decision["replicate_id"])): decision
        for decision in features["decisions"]
    }
    model = _load_mapping(model_path, "calibrated model")
    _require(model.get("domains"), ["docvqa"], "model domain")
    _require(model.get("feature_mode"), "hybrid-context-semantic", "feature mode")
    selected_threshold = float(freeze["calibration"]["selected_threshold"])
    _require(float(model["threshold"]), selected_threshold, "calibrated threshold")
    report = evaluate_factorized_risk_controlled_policy(
        model,
        records,
        semantic_decisions=semantic_decisions,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_confidence=args.bootstrap_confidence,
        bootstrap_seed=args.bootstrap_seed,
    )
    _require(report.get("n_sources"), FORMAL_SOURCES, "report source count")
    _require(report.get("n_decisions"), args.expected_states, "report decision count")
    _require(report.get("threshold"), selected_threshold, "report threshold")
    pass_rule = report.get("pass_rule")
    if not isinstance(pass_rule, dict):
        raise ValueError("DocVQA formal report lacks its pass rule")
    pass_rule["threshold_matches_calibration_choice"] = True
    pass_rule["all_frozen_hashes_and_identity_audits_match"] = True
    report["passed"] = all(bool(value) for value in pass_rule.values())
    report["scientific_status"] = (
        "one-shot DocVQA-train evaluation of the frozen factorized-v2 policy"
    )

    repo_dir = Path(__file__).resolve().parents[1]
    code_revision = _git_revision(repo_dir)
    _require(code_revision, freeze["code_revision"], "evaluator code revision")
    report["run"] = {
        "code_revision": code_revision,
        "formal_outcomes_used": True,
        "no_target_derived_tuning": True,
        "policy_freeze": str(policy_freeze_path),
        "policy_freeze_sha256": args.expected_policy_freeze_sha256,
        "model": str(model_path),
        "model_sha256": args.expected_model_sha256,
        "manifest": str(manifest_path),
        "manifest_sha256": args.expected_manifest_sha256,
        "manifest_provenance": str(manifest_provenance_path),
        "manifest_provenance_sha256": args.expected_manifest_provenance_sha256,
        "formal_audit": str(formal_audit_path),
        "formal_audit_sha256": args.expected_formal_audit_sha256,
        "rollouts": str(rollouts_path),
        "rollouts_sha256": args.expected_rollouts_sha256,
        "rollout_audit": str(rollout_audit_path),
        "rollout_audit_sha256": args.expected_rollout_audit_sha256,
        "features": str(features_path),
        "features_sha256": args.expected_features_sha256,
        "feature_outcomes_included": False,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_confidence": BOOTSTRAP_CONFIDENCE,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "fresh_rollout_audit": fresh_rollout_audit,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
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
