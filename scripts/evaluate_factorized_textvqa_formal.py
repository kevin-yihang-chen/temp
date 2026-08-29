from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from beyond_entropy.dataset import read_jsonl
from beyond_entropy.factorized_evaluation import (
    evaluate_factorized_risk_controlled_policy,
)
from beyond_entropy.factorized_formal import (
    MODEL_REVISION,
    check_hash,
    load_mapping,
    validate_materialized_formal_gate,
)
from beyond_entropy.qwen_semantic import (
    load_semantic_feature_dataset,
    validate_semantic_feature_dataset,
)
from beyond_entropy.rollout_audit import audit_sibling_rollout_bank


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-shot evaluation of the frozen factorized-v2 formal policy"
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
    parser.add_argument("--expected-scientific-status", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.975)
    parser.add_argument("--bootstrap-seed", type=int, default=20260828)
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parents[1]
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
        audit_path=args.formal_audit,
        expected_audit_sha256=args.expected_formal_audit_sha256,
    )
    if args.bootstrap_resamples != 20000:
        raise ValueError("formal evaluation requires exactly 20,000 resamples")
    if args.bootstrap_confidence != 0.975:
        raise ValueError("formal evaluation requires a two-sided 97.5% interval")
    if args.bootstrap_seed != 20260828:
        raise ValueError("formal evaluation bootstrap seed is frozen at 20260828")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite formal report: {args.output}")

    check_hash(args.rollouts, args.expected_rollouts_sha256, "formal rollouts")
    check_hash(
        args.rollout_audit,
        args.expected_rollout_audit_sha256,
        "formal rollout audit",
    )
    check_hash(args.features, args.expected_features_sha256, "formal features")
    rollout_audit = audit_sibling_rollout_bank(
        args.manifest,
        args.rollouts,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_states=args.expected_states,
        expected_candidate_count=4,
        expected_model_revision=MODEL_REVISION,
        expected_scientific_status=args.expected_scientific_status,
    )
    if rollout_audit["rollouts_sha256"] != args.expected_rollouts_sha256:
        raise ValueError("fresh rollout audit does not match the frozen rollout hash")
    stored_rollout_audit = load_mapping(args.rollout_audit, "rollout audit")
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
    ):
        if stored_rollout_audit.get(name) != rollout_audit.get(name):
            raise ValueError(f"stored rollout audit differs for {name}")

    records = read_jsonl(args.rollouts)
    features = load_semantic_feature_dataset(args.features)
    validate_semantic_feature_dataset(features, records)
    metadata = features.get("metadata")
    if not isinstance(metadata, Mapping) or bool(metadata.get("outcomes_included", True)):
        raise ValueError("formal evaluation requires label-free semantic features")
    semantic_decisions = {
        (str(decision["state_id"]), str(decision["replicate_id"])): decision
        for decision in features["decisions"]
    }
    model: dict[str, Any] = load_mapping(args.model, "calibrated model")
    if float(model["threshold"]) != float(
        freeze["calibration"]["selected_threshold"]
    ):
        raise ValueError("formal model threshold differs from policy freeze")
    report = evaluate_factorized_risk_controlled_policy(
        model,
        records,
        semantic_decisions=semantic_decisions,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_confidence=args.bootstrap_confidence,
        bootstrap_seed=args.bootstrap_seed,
    )
    if report["n_sources"] != 5953 or report["n_decisions"] != args.expected_states:
        raise ValueError("formal report does not cover the frozen formal population")
    code_revision = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if code_revision != freeze["code_revision"]:
        raise ValueError("formal evaluator revision differs from policy freeze")
    report["run"] = {
        "code_revision": code_revision,
        "formal_outcomes_used": True,
        "no_target_derived_tuning": True,
        "policy_freeze": str(args.policy_freeze.resolve()),
        "policy_freeze_sha256": args.expected_policy_freeze_sha256,
        "model": str(args.model.resolve()),
        "model_sha256": args.expected_model_sha256,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": args.expected_manifest_sha256,
        "manifest_provenance_sha256": args.expected_manifest_provenance_sha256,
        "formal_audit_sha256": args.expected_formal_audit_sha256,
        "rollouts": str(args.rollouts.resolve()),
        "rollouts_sha256": args.expected_rollouts_sha256,
        "rollout_audit_sha256": args.expected_rollout_audit_sha256,
        "features": str(args.features.resolve()),
        "features_sha256": args.expected_features_sha256,
        "feature_outcomes_included": False,
        "rollout_audit": rollout_audit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
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
