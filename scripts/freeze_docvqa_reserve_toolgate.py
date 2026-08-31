#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
from pathlib import Path

from beyond_entropy.docvqa_reserve import (
    RESERVE_END_EXCLUSIVE,
    RESERVE_SOURCES,
    RESERVE_START,
)
from beyond_entropy.reserve_freeze import sha256_file, validate_reserve_freeze


def _revision(repo_dir: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean(repo_dir: Path) -> None:
    status = subprocess.run(
        ["git", "-C", str(repo_dir), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("tracked worktree must be clean before reserve freeze")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze all outcome-sealed DocVQA reserve comparator inputs"
    )
    parser.add_argument("--comparator-verification", type=Path, required=True)
    parser.add_argument("--identity-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    _require_clean(repo)
    verification = args.comparator_verification.resolve()
    verification_payload = json.loads(verification.read_text(encoding="utf-8"))
    if not isinstance(verification_payload, dict) or (
        verification_payload.get("passed") is not True
        or verification_payload.get("reserve_outcomes_used") is not False
        or verification_payload.get("formal_outcomes_used") is not False
    ):
        raise ValueError("comparator verification is not a passing sealed result")
    identity_audit = args.identity_audit.resolve()
    identity_payload = json.loads(identity_audit.read_text(encoding="utf-8"))
    if not isinstance(identity_payload, dict) or (
        identity_payload.get("passed") is not True
        or identity_payload.get("source_group_count") != RESERVE_SOURCES
        or identity_payload.get("selection_target_fields_accessed") is not False
        or identity_payload.get("reserve_outcomes_used") is not False
    ):
        raise ValueError("reserve identity audit is not a passing sealed result")

    relative_components = {
        "allocation": "artifacts/docvqa-train-factorized-v2/allocation/allocation.json",
        "allocation_audit": "artifacts/docvqa-train-factorized-v2/allocation/allocation.audit.json",
        "allocation_protocol": "docs/docvqa_train_factorized_v2_preregistration.md",
        "comparator_protocol": "artifacts/docvqa-train-factorized-v2/ops/reserve-toolgate-comparator-protocol-20260830.md",
        "implementation_specification": "artifacts/docvqa-train-factorized-v2/ops/reserve-toolgate-comparator-implementation-spec-20260830.md",
        "development_rollouts": "artifacts/docvqa-train-factorized-v2/ranker-training/qwen3b-c4-seed0/rollouts.jsonl",
        "development_features": "artifacts/docvqa-train-factorized-v2/ranker-training/attention-semantic-v1/features-question-region-attention-label-free.pt",
        "policy_a_model": "artifacts/docvqa-train-factorized-v2/ranker-training/factorized-oof-v1/model.json",
        "policy_a_report": "artifacts/docvqa-train-factorized-v2/ranker-training/factorized-oof-v1/report.json",
        "policy_b_model": "artifacts/docvqa-train-factorized-v2/reserve-comparator/toolgate-proxy-v1/model.json",
        "policy_b_report": "artifacts/docvqa-train-factorized-v2/reserve-comparator/toolgate-proxy-v1/report.json",
        "reserve_identity_module": "src/beyond_entropy/docvqa_reserve.py",
        "reserve_comparator_module": "src/beyond_entropy/reserve_toolgate.py",
        "reserve_freeze_module": "src/beyond_entropy/reserve_freeze.py",
        "reserve_exporter": "scripts/export_docvqa_reserve_toolgate.py",
        "reserve_scorer": "scripts/score_docvqa_reserve_toolgate.py",
        "reserve_evaluator": "scripts/evaluate_docvqa_reserve_toolgate.py",
        "reserve_gate_verifier": "scripts/verify_docvqa_reserve_gate.py",
        "reserve_export_worker": "scripts/slurm_docvqa_reserve_toolgate_export.sh",
        "reserve_pipeline_worker": "scripts/slurm_docvqa_reserve_toolgate_pipeline.sh",
        "reserve_submitter": "scripts/submit_docvqa_reserve_toolgate.sh",
        "rollout_shard_preparer": "artifacts/docvqa-train-factorized-v2/ops/prepare_docvqa_formal_multigpu_shards.py",
        "rollout_shard_merger": "artifacts/docvqa-train-factorized-v2/ops/merge_docvqa_formal_multigpu_rollouts.py",
        "semantic_shard_preparer": "scripts/prepare_semantic_feature_batch_shards.py",
        "semantic_shard_merger": "scripts/merge_semantic_feature_shards.py",
        "comparator_verifier": "scripts/verify_reserve_toolgate_comparator.py",
        "reserve_identity_auditor": "scripts/audit_docvqa_reserve_identities.py",
    }
    paths = {name: (repo / path).resolve() for name, path in relative_components.items()}
    paths["comparator_verification"] = verification
    paths["reserve_identity_audit"] = identity_audit
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"reserve freeze component {name} is absent: {path}")
    document = {
        "schema_version": 1,
        "scientific_status": (
            "outcome-sealed DocVQA reserve ToolGate comparator implementation freeze"
        ),
        "code_revision": _revision(repo),
        "population": {
            "rank_start": RESERVE_START,
            "rank_end_exclusive": RESERVE_END_EXCLUSIVE,
            "expected_source_groups": RESERVE_SOURCES,
            "manifest_materialized": False,
            "rollouts_collected": False,
            "outcomes_used": False,
        },
        "components": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sorted(paths.items())
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scikit-learn", "torch")
        },
        "formal_outcomes_used": False,
        "reserve_outcomes_used": False,
    }
    validate_reserve_freeze(
        document,
        expected_code_revision=document["code_revision"],
        verify_components=True,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"freeze": str(output), "sha256": sha256_file(output)}, indent=2))


if __name__ == "__main__":
    main()
