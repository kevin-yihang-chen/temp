from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from beyond_entropy.dataset import read_jsonl
from beyond_entropy.stopping import FrozenWhenToCallGate
from beyond_entropy.vtool_adapter import (
    VTOOL_GATE_SCHEMA_VERSION,
    VToolGateControl,
    build_vtool_gate_manifest_rows,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export frozen when-to-call controls for VTool training-v2",
    )
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rollouts-sha256")
    parser.add_argument("--expected-model-sha256")
    parser.add_argument("--registered-lambda-cost", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rollout_sha256 = _sha256(args.rollouts)
    if (
        args.expected_rollouts_sha256 is not None
        and rollout_sha256 != args.expected_rollouts_sha256
    ):
        raise ValueError(
            "rollout SHA-256 mismatch: "
            f"{rollout_sha256} != {args.expected_rollouts_sha256}"
        )
    gate = FrozenWhenToCallGate.load(
        args.model,
        expected_sha256=args.expected_model_sha256,
        registered_lambda_cost=args.registered_lambda_cost,
    )
    rows = build_vtool_gate_manifest_rows(read_jsonl(args.rollouts), gate)
    _write_jsonl(rows, args.output)
    calls = sum(
        VToolGateControl.from_tools_metadata(row["tools_kwargs_metadata"])
        .should_call_tool
        for row in rows
    )
    output_sha256 = _sha256(args.output)
    provenance_path = args.output.with_suffix(".provenance.json")
    _write_json(
        {
            "scientific_status": "integration artifact; not a new model result",
            "schema_version": VTOOL_GATE_SCHEMA_VERSION,
            "rollouts": str(args.rollouts.resolve()),
            "rollouts_sha256": rollout_sha256,
            "model": str(args.model.resolve()),
            "model_sha256": gate.model_sha256,
            "registered_lambda_cost": gate.registered_lambda_cost,
            "decisions": len(rows),
            "tool_calls": calls,
            "tool_call_rate": calls / len(rows) if rows else 0.0,
            "output": str(args.output.resolve()),
            "output_sha256": output_sha256,
        },
        provenance_path,
    )
    print(json.dumps({"output": str(args.output), "provenance": str(provenance_path)}))


if __name__ == "__main__":
    main()
