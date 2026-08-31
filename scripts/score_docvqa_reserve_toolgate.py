#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from beyond_entropy.qwen_semantic import (
    load_semantic_feature_dataset,
    validate_semantic_feature_dataset,
)
from beyond_entropy.reserve_freeze import (
    component_path,
    sha256_file,
    validate_reserve_freeze,
)
from beyond_entropy.reserve_toolgate import score_reserve_policies
from beyond_entropy.schema import ActionRecord


def _load(path: Path, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"reserve {name} must be a JSON object")
    return payload


def _revision(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _read_redacted_rollouts(path: Path) -> list[ActionRecord]:
    """Mask all task/post-action fields before model-record construction."""

    records: list[ActionRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"reserve rollout line {line_number} is not an object")
            payload["correct_before"] = 0.0
            payload["correct_after"] = 0.0
            payload["answer_after"] = payload.get("answer_before", "")
            payload["entropy_after"] = payload.get("entropy_before", 0.0)
            metadata = payload.get("metadata")
            if isinstance(metadata, dict) and payload.get("action_type") == "ZOOM":
                payload["metadata"] = {}
            records.append(ActionRecord.from_dict(payload))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze outcome-blind reserve Policy A/B scores before evaluation"
    )
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--reserve-audit", type=Path, required=True)
    parser.add_argument("--expected-reserve-audit-sha256", required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--expected-rollouts-sha256", required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--expected-features-sha256", required=True)
    parser.add_argument("--scores-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    revision = _revision(repo)
    freeze_path = args.freeze.resolve()
    if sha256_file(freeze_path) != args.expected_freeze_sha256:
        raise ValueError("reserve freeze SHA-256 mismatch")
    freeze = _load(freeze_path, "freeze")
    validate_reserve_freeze(
        freeze, expected_code_revision=revision, verify_components=True
    )
    inputs = {
        "manifest": (args.manifest.resolve(), args.expected_manifest_sha256),
        "reserve_audit": (
            args.reserve_audit.resolve(),
            args.expected_reserve_audit_sha256,
        ),
        "rollouts": (args.rollouts.resolve(), args.expected_rollouts_sha256),
        "features": (args.features.resolve(), args.expected_features_sha256),
    }
    for name, (path, expected) in inputs.items():
        if sha256_file(path) != expected:
            raise ValueError(f"reserve {name} SHA-256 mismatch")
    audit = _load(inputs["reserve_audit"][0], "manifest audit")
    if (
        audit.get("passed") is not True
        or audit.get("freeze_sha256") != args.expected_freeze_sha256
        or audit.get("reserve_outcomes_collected") is not False
        or audit.get("manifest", {}).get("manifest_sha256")
        != args.expected_manifest_sha256
    ):
        raise ValueError("reserve manifest audit is not bound to the freeze")

    redacted_records = _read_redacted_rollouts(inputs["rollouts"][0])
    features = load_semantic_feature_dataset(inputs["features"][0])
    validate_semantic_feature_dataset(
        features,
        redacted_records,
        require_outcomes=False,
    )
    semantic = {
        (str(item["state_id"]), str(item["replicate_id"])): item
        for item in features["decisions"]
    }
    policy_a = _load(component_path(freeze, "policy_a_model"), "Policy A model")
    policy_b = _load(component_path(freeze, "policy_b_model"), "Policy B model")
    report, rows = score_reserve_policies(
        policy_a,
        policy_b,
        redacted_records,
        semantic_decisions=semantic,
    )
    report.update(
        {
            "code_revision": revision,
            "freeze_sha256": args.expected_freeze_sha256,
            "inputs": {
                name: {"path": str(path), "sha256": expected}
                for name, (path, expected) in inputs.items()
            },
            "raw_rollout_outcomes_redacted_before_record_construction": True,
        }
    )
    scores_output = args.scores_output.resolve()
    report_output = args.report_output.resolve()
    for path in (scores_output, report_output):
        if path.exists():
            raise FileExistsError(f"reserve score output exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    scores_staging = scores_output.with_suffix(scores_output.suffix + ".tmp")
    report_staging = report_output.with_suffix(report_output.suffix + ".tmp")
    for path in (scores_staging, report_staging):
        if path.exists():
            raise FileExistsError(f"reserve score staging output exists: {path}")
    with scores_staging.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(scores_staging, scores_output)
    report["scores"] = {
        "path": str(scores_output),
        "sha256": sha256_file(scores_output),
        "rows": len(rows),
    }
    with report_staging.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(report_staging, report_output)
    print(
        json.dumps(
            {
                "scores": str(scores_output),
                "scores_sha256": sha256_file(scores_output),
                "report": str(report_output),
                "report_sha256": sha256_file(report_output),
                "decisions": len(rows),
                "selection_uses_outcomes": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
