from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from beyond_entropy.action_value import predict_frozen_factorized_action_values
from beyond_entropy.dataset import read_jsonl
from beyond_entropy.docvqa_candidate_freeze import (
    PROTOCOL_SHA256,
    build_frozen_candidate,
    validate_candidate_freeze_gate,
)
from beyond_entropy.docvqa_train_allocation import sha256_file
from beyond_entropy.qwen_semantic import (
    load_semantic_feature_dataset,
    validate_semantic_feature_dataset,
)


def _load_mapping(path: Path, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _require(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"DocVQA candidate input mismatch for {name}")


def _require_empty(path: Path, name: str) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"{name} must remain unmaterialized before candidate freeze")


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")


def _tracked_revision(repo_dir: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("tracked worktree must be clean before candidate freeze")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the sole DocVQA-train factorized-v2 candidate"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--allocation-audit", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--calibration-output-dir", type=Path, required=True)
    parser.add_argument("--formal-output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "raw_model": args.model.resolve(),
        "development_report": args.report.resolve(),
        "development_rollouts": args.rollouts.resolve(),
        "development_features": args.features.resolve(),
        "allocation": args.allocation.resolve(),
        "allocation_audit": args.allocation_audit.resolve(),
        "protocol": args.protocol.resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"candidate input does not exist: {name}={path}")
    input_hashes = {name: sha256_file(path) for name, path in paths.items()}
    _require(input_hashes["protocol"], PROTOCOL_SHA256, "protocol SHA-256")
    _require_empty(args.calibration_output_dir.resolve(), "calibration output")
    _require_empty(args.formal_output_dir.resolve(), "formal output")
    if args.output.exists() or args.audit_output.exists():
        raise FileExistsError("candidate or candidate audit output already exists")

    allocation = _load_mapping(paths["allocation"], "allocation")
    allocation_audit = _load_mapping(paths["allocation_audit"], "allocation audit")
    validate_candidate_freeze_gate(
        allocation,
        allocation_audit,
        allocation_sha256=input_hashes["allocation"],
    )
    model = _load_mapping(paths["raw_model"], "raw model")
    report = _load_mapping(paths["development_report"], "development report")
    report_run = report.get("run")
    if not isinstance(report_run, Mapping):
        raise ValueError("development report is missing run provenance")
    development_inputs = report_run.get("development_inputs")
    semantic_features = report_run.get("semantic_features")
    if not isinstance(development_inputs, Mapping) or not isinstance(
        semantic_features, Mapping
    ):
        raise ValueError("development report is missing DocVQA input provenance")
    docvqa_input = development_inputs.get("docvqa")
    docvqa_features = semantic_features.get("docvqa")
    if not isinstance(docvqa_input, Mapping) or not isinstance(
        docvqa_features, Mapping
    ):
        raise ValueError("development report is missing the DocVQA domain")
    _require(
        docvqa_input.get("sha256"),
        input_hashes["development_rollouts"],
        "development rollout SHA-256",
    )
    _require(
        docvqa_features.get("sha256"),
        input_hashes["development_features"],
        "development feature SHA-256",
    )

    records = read_jsonl(paths["development_rollouts"])
    features = load_semantic_feature_dataset(paths["development_features"])
    validate_semantic_feature_dataset(features, records)
    if bool(features["metadata"].get("outcomes_included", True)):
        raise ValueError("candidate freeze requires label-free semantic features")
    semantic_decisions = {
        (str(decision["state_id"]), str(decision["replicate_id"])): decision
        for decision in features["decisions"]
    }
    actions, scores_by_key = predict_frozen_factorized_action_values(
        model,
        records,
        semantic_decisions=semantic_decisions,
    )
    source_by_key = {
        (record.state_id, record.replicate_id): record.source_id
        for record in records
        if record.action_type == "ANSWER"
    }
    if set(actions) != set(scores_by_key) or set(actions) != set(source_by_key):
        raise RuntimeError("candidate predictions do not exactly cover development")
    _require(docvqa_input.get("records"), len(records), "development record count")

    repo_dir = Path(__file__).resolve().parents[1]
    code_revision = _tracked_revision(repo_dir)
    provenance = {
        key: value
        for name, path in paths.items()
        for key, value in (
            (name, str(path)),
            (f"{name}_sha256", input_hashes[name]),
        )
    }
    candidate, audit = build_frozen_candidate(
        model,
        report,
        scores_by_key=scores_by_key,
        source_by_key=source_by_key,
        provenance=provenance,
        code_revision=code_revision,
    )
    audit["candidate"] = str(args.output.resolve())
    audit["action_predictions"] = len(actions)
    _write_exclusive(args.output.resolve(), candidate)
    if sha256_file(args.output.resolve()) != audit["candidate_sha256"]:
        raise RuntimeError("serialized candidate SHA-256 differs from freeze audit")
    _write_exclusive(args.audit_output.resolve(), audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
