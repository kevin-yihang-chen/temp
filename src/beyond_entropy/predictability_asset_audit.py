from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .dataset import read_jsonl
from .predictability_audit import (
    collapse_fixed_entropy_tool,
    fixed_tool_headroom_summary,
    matrix_completion_report,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_path(root: Path, value: Mapping[str, Any], *, name: str) -> Path:
    path = (root / str(value["path"])).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} is missing: {path}")
    actual = _sha256(path)
    if actual != value["sha256"]:
        raise ValueError(
            f"{name} SHA-256 mismatch: expected {value['sha256']}, got {actual}"
        )
    return path


def audit_retrospective_assets(
    *, config_path: str | Path, repository_root: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config_file = Path(config_path).resolve()
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if config.get("schema") != "predictability_retrospective_assets_config_v1":
        raise ValueError("unexpected retrospective asset config schema")
    banks = config.get("banks")
    if not isinstance(banks, Mapping) or not banks:
        raise ValueError("asset config requires non-empty banks")

    reports: dict[str, Any] = {}
    for name, raw_spec in banks.items():
        if not isinstance(raw_spec, Mapping):
            raise ValueError(f"bank {name!r} must be a mapping")
        if raw_spec.get("final_test_eligible") is not False:
            raise ValueError(
                f"retrospective bank {name!r} must be marked final-test ineligible"
            )
        rollouts = _checked_path(root, raw_spec["rollouts"], name=f"{name} rollouts")
        features = _checked_path(root, raw_spec["features"], name=f"{name} features")
        records = read_jsonl(rollouts)
        outcomes = collapse_fixed_entropy_tool(
            records, expected_zoom_actions=int(config["fixed_tool_candidate_count"])
        )
        reports[str(name)] = {
            "dataset_role": raw_spec["dataset_role"],
            "final_test_eligible": False,
            "rollouts": {
                "path": str(rollouts.relative_to(root)),
                "sha256": _sha256(rollouts),
                "action_records": len(records),
            },
            "features": {
                "path": str(features.relative_to(root)),
                "sha256": _sha256(features),
                "declared_available_levels": list(
                    raw_spec["declared_available_levels"]
                ),
                "declared_missing_requirements": list(
                    raw_spec["declared_missing_requirements"]
                ),
                "contents_not_loaded": True,
            },
            "identity": {
                "decisions": len(outcomes),
                "unique_image_ids": len({item.image_id for item in outcomes}),
                "unique_source_ids": len({item.source_id for item in outcomes}),
            },
            "fixed_tool_headroom": fixed_tool_headroom_summary(
                outcomes, lambda_cost=float(config["lambda_cost"])
            ),
        }

    return {
        "schema": "predictability_retrospective_assets_report_v1",
        "config": {
            "path": str(config_file.relative_to(root)),
            "sha256": _sha256(config_file),
        },
        "fixed_tool": "four-crop entropy search",
        "banks": reports,
        "formal_matrix": matrix_completion_report([]),
        "decision": "retrospective_assets_only_formal_matrix_incomplete",
        "scientific_boundary": (
            "headroom values validate label construction only; historically opened data, "
            "missing L0/L3 features, and absent HRBench prevent a formal verdict"
        ),
    }


def _atomic_json(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Audit opened assets for the fixed-tool study"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = audit_retrospective_assets(
        config_path=args.config,
        repository_root=args.repository_root,
    )
    _atomic_json(report, Path(args.output))


if __name__ == "__main__":
    main()
