#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from beyond_entropy.infographicvqa_relative_where_diagnostics import (
    ACTION_GENERALIZATION_SCHEMA,
    audit_relative_where_action_generalization,
)


EXPECTED_DECISIONS = 23_946
EXPECTED_SOURCES = 2_204


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked(path: Path, expected: str, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or _sha256(resolved) != expected:
        raise ValueError(f"relative-where diagnostic {label} SHA-256 mismatch")
    return resolved


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("relative-where diagnostic JSONL row is not an object")
            rows.append(value)
    return rows


def _read_answer_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("relative-where diagnostic rollout row is invalid")
            if value.get("action_type") == "ANSWER":
                rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("predictions", "answer_nll", "rollouts", "parent_result", "protocol"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
        parser.add_argument(
            f"--expected-{name.replace('_', '-')}-sha256", required=True
        )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    names = ("predictions", "answer_nll", "rollouts", "parent_result", "protocol")
    paths = {
        name: _checked(
            getattr(args, name), getattr(args, f"expected_{name}_sha256"), name
        )
        for name in names
    }
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite relative-where diagnostic: {output_dir}"
        )

    predictions = _read_jsonl(paths["predictions"])
    nll_rows = _read_jsonl(paths["answer_nll"])
    answer_rows = _read_answer_rows(paths["rollouts"])
    audit = audit_relative_where_action_generalization(
        predictions,
        nll_rows,
        answer_rows,
        expected_decisions=EXPECTED_DECISIONS,
        expected_sources=EXPECTED_SOURCES,
    )
    if audit["schema"] != ACTION_GENERALIZATION_SCHEMA:
        raise RuntimeError("relative-where diagnostic schema changed")
    audit["inputs"] = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in paths.items()
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=output_dir.name + ".partial-", dir=output_dir.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        audit_path = temporary / "audit.json"
        _write_json(audit_path, audit)
        completion = {
            "schema": "infographicvqa_relative_where_action_generalization_complete_v1",
            "audit": {"path": "audit.json", "sha256": _sha256(audit_path)},
            "decisions": audit["population"]["decisions"],
            "sources": audit["population"]["sources"],
            "prediction_outcomes_included": False,
            "validation_or_test_inputs_used": False,
            "changes_parent_train_gate": False,
        }
        _write_json(temporary / "complete.json", completion)
        temporary.replace(output_dir)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
