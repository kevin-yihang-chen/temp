from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from beyond_entropy.docvqa_formal import (
    FORMAL_SCIENTIFIC_STATUS,
    FORMAL_SOURCES,
    MODEL_REVISION,
    validate_materialized_formal_gate,
)
from beyond_entropy.docvqa_train_allocation import sha256_file
from beyond_entropy.rollout_audit import audit_sibling_rollout_bank


def _load_mapping(path: Path, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"DocVQA formal {name} must be a JSON object")
    return payload


def _tracked_revision(repo_dir: Path) -> str:
    status = subprocess.run(
        ["git", "-C", str(repo_dir), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("tracked worktree must be clean before DocVQA formal audit")
    return subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_frozen(path: Path, payload: Mapping[str, Any], *, resume: bool) -> None:
    serialized = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if not resume:
            raise FileExistsError(f"refusing to overwrite DocVQA formal audit: {path}")
        if path.read_text(encoding="utf-8") != serialized:
            raise ValueError("existing DocVQA formal rollout audit differs")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the one-shot DocVQA-train formal sibling bank"
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
    parser.add_argument("--expected-states", type=int, required=True)
    parser.add_argument("--expected-code-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    policy_freeze_path = args.policy_freeze.resolve()
    model_path = args.model.resolve()
    manifest_path = args.manifest.resolve()
    manifest_provenance_path = args.manifest_provenance.resolve()
    formal_audit_path = args.formal_audit.resolve()
    rollouts_path = args.rollouts.resolve()
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
    if args.expected_states < FORMAL_SOURCES:
        raise ValueError("DocVQA formal state count is smaller than source count")
    repo_dir = Path(__file__).resolve().parents[1]
    code_revision = _tracked_revision(repo_dir)
    if code_revision != args.expected_code_revision or code_revision != freeze.get(
        "code_revision"
    ):
        raise ValueError("DocVQA formal audit revision differs from policy freeze")
    report = audit_sibling_rollout_bank(
        manifest_path,
        rollouts_path,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_states=args.expected_states,
        expected_candidate_count=4,
        expected_model_revision=MODEL_REVISION,
        expected_scientific_status=FORMAL_SCIENTIFIC_STATUS,
    )
    expected_report = {
        "passed": True,
        "manifest_sha256": args.expected_manifest_sha256,
        "model_revision": MODEL_REVISION,
        "code_revision": code_revision,
        "scientific_status": FORMAL_SCIENTIFIC_STATUS,
        "states": args.expected_states,
        "records": args.expected_states * 5,
        "unique_sources": FORMAL_SOURCES,
        "unique_images": FORMAL_SOURCES,
        "candidate_count": 4,
        "answer_records": args.expected_states,
        "zoom_records": args.expected_states * 4,
    }
    for name, expected in expected_report.items():
        if report.get(name) != expected:
            raise ValueError(f"DocVQA formal rollout audit mismatch for {name}")
    report.update(
        {
            "policy_freeze_sha256": args.expected_policy_freeze_sha256,
            "model_sha256": args.expected_model_sha256,
            "manifest_provenance_sha256": (
                args.expected_manifest_provenance_sha256
            ),
            "formal_audit_sha256": args.expected_formal_audit_sha256,
            "protocol_sha256": freeze["artifacts"]["protocol"]["sha256"],
            "formal_outcomes_collected": True,
            "formal_outcomes_used_for_tuning": False,
        }
    )
    if report.get("rollouts_sha256") != sha256_file(rollouts_path):
        raise ValueError("DocVQA formal rollout hash changed during audit")
    _write_frozen(output_path, report, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
