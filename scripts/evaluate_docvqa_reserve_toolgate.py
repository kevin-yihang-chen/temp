#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from beyond_entropy.dataset import read_jsonl
from beyond_entropy.reserve_freeze import sha256_file, validate_reserve_freeze
from beyond_entropy.reserve_toolgate import evaluate_reserve_policies


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the one-shot paired reserve ToolGate evaluation"
    )
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--expected-rollouts-sha256", required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--expected-scores-sha256", required=True)
    parser.add_argument("--score-report", type=Path, required=True)
    parser.add_argument("--expected-score-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260829)
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
        "rollouts": (args.rollouts.resolve(), args.expected_rollouts_sha256),
        "scores": (args.scores.resolve(), args.expected_scores_sha256),
        "score_report": (
            args.score_report.resolve(),
            args.expected_score_report_sha256,
        ),
    }
    for name, (path, expected) in inputs.items():
        if sha256_file(path) != expected:
            raise ValueError(f"reserve {name} SHA-256 mismatch")
    score_report = _load(inputs["score_report"][0], "score report")
    if (
        score_report.get("selection_uses_outcomes") is not False
        or score_report.get("raw_rollout_outcomes_redacted_before_record_construction")
        is not True
        or score_report.get("scores", {}).get("sha256")
        != args.expected_scores_sha256
        or score_report.get("freeze_sha256") != args.expected_freeze_sha256
    ):
        raise ValueError("reserve policy scores are not outcome-blind and freeze-bound")
    score_rows: list[dict[str, Any]] = []
    with inputs["scores"][0].open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"reserve score line {line_number} is not an object")
            score_rows.append(payload)
    records = read_jsonl(inputs["rollouts"][0])
    report = evaluate_reserve_policies(
        records,
        score_rows,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    report.update(
        {
            "code_revision": revision,
            "freeze_sha256": args.expected_freeze_sha256,
            "inputs": {
                name: {"path": str(path), "sha256": expected}
                for name, (path, expected) in inputs.items()
            },
        }
    )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite one-shot reserve result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_suffix(output.suffix + ".tmp")
    if staging.exists():
        raise FileExistsError(f"one-shot reserve staging result exists: {staging}")
    with staging.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(staging, output)
    print(
        json.dumps(
            {
                "report": str(output),
                "report_sha256": sha256_file(output),
                "supports_policy_a_over_policy_b": report[
                    "supports_policy_a_over_policy_b"
                ],
                "primary_estimand": report["primary_estimand"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
