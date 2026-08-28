from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from beyond_entropy.dataset import read_jsonl
from beyond_entropy.qwen_semantic import (
    load_semantic_feature_dataset,
    validate_semantic_feature_dataset,
)
from beyond_entropy.rollout_audit import audit_sibling_rollout_bank
from beyond_entropy.scaled_evaluation import evaluate_scaled_risk_controlled_policy


MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_hash(path: Path, expected: str, name: str) -> str:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{name} SHA-256 mismatch")
    return actual


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-shot evaluation of the frozen scaled TextVQA policy"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--expected-rollouts-sha256", required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--expected-features-sha256", required=True)
    parser.add_argument("--expected-states", type=int, required=True)
    parser.add_argument("--expected-scientific-status", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-evaluator-module-sha256", required=True)
    parser.add_argument("--expected-evaluator-script-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.975)
    parser.add_argument("--bootstrap-seed", type=int, default=20260828)
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parents[1]
    evaluator_module = repo_dir / "src/beyond_entropy/scaled_evaluation.py"
    paths_and_hashes = (
        (args.model, args.expected_model_sha256, "model"),
        (args.manifest, args.expected_manifest_sha256, "manifest"),
        (args.rollouts, args.expected_rollouts_sha256, "rollouts"),
        (args.features, args.expected_features_sha256, "features"),
        (args.protocol, args.expected_protocol_sha256, "protocol"),
        (
            evaluator_module,
            args.expected_evaluator_module_sha256,
            "evaluator module",
        ),
        (
            Path(__file__).resolve(),
            args.expected_evaluator_script_sha256,
            "evaluator script",
        ),
    )
    actual_hashes = {
        name: _check_hash(path, expected, name)
        for path, expected, name in paths_and_hashes
    }
    if args.bootstrap_resamples != 20000:
        raise ValueError("formal evaluation requires exactly 20,000 resamples")
    if args.bootstrap_confidence != 0.975:
        raise ValueError("formal evaluation requires a two-sided 97.5% interval")
    if args.bootstrap_seed != 20260828:
        raise ValueError("formal evaluation bootstrap seed is frozen at 20260828")

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
        raise ValueError("rollout audit SHA-256 does not match frozen input")

    records = read_jsonl(args.rollouts)
    features = load_semantic_feature_dataset(args.features)
    validate_semantic_feature_dataset(features, records)
    metadata = features.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("semantic feature metadata must be a mapping")
    if bool(metadata.get("outcomes_included", True)):
        raise ValueError("formal evaluation requires label-free feature storage")
    semantic_decisions = {
        (str(decision["state_id"]), str(decision["replicate_id"])): decision
        for decision in features["decisions"]
    }
    model: dict[str, Any] = json.loads(args.model.read_text(encoding="utf-8"))
    report = evaluate_scaled_risk_controlled_policy(
        model,
        records,
        semantic_decisions=semantic_decisions,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_confidence=args.bootstrap_confidence,
        bootstrap_seed=args.bootstrap_seed,
    )
    code_revision = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report["run"] = {
        "code_revision": code_revision,
        "formal_outcomes_used": True,
        "no_target_derived_tuning": True,
        "model": str(args.model.resolve()),
        "model_sha256": actual_hashes["model"],
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": actual_hashes["manifest"],
        "rollouts": str(args.rollouts.resolve()),
        "rollouts_sha256": actual_hashes["rollouts"],
        "features": str(args.features.resolve()),
        "features_sha256": actual_hashes["features"],
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": actual_hashes["protocol"],
        "evaluator_module_sha256": actual_hashes["evaluator module"],
        "evaluator_script_sha256": actual_hashes["evaluator script"],
        "feature_outcomes_included": False,
        "rollout_audit": rollout_audit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
