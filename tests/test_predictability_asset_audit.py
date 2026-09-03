from __future__ import annotations

import hashlib
import json
from pathlib import Path

from beyond_entropy.dataset import write_jsonl
from beyond_entropy.predictability_asset_audit import audit_retrospective_assets
from test_predictability_audit import _siblings


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_retrospective_asset_audit_never_claims_formal_completion(
    tmp_path: Path,
) -> None:
    rollouts = tmp_path / "rollouts.jsonl"
    features = tmp_path / "features.pt"
    config = tmp_path / "config.json"
    write_jsonl(_siblings(), rollouts)
    features.write_bytes(b"opaque feature bundle")
    config.write_text(
        json.dumps(
            {
                "schema": "predictability_retrospective_assets_config_v1",
                "lambda_cost": 0.05,
                "fixed_tool_candidate_count": 4,
                "banks": {
                    "opened": {
                        "dataset_role": "retrospective_smoke_only",
                        "final_test_eligible": False,
                        "rollouts": {
                            "path": rollouts.name,
                            "sha256": _digest(rollouts),
                        },
                        "features": {
                            "path": features.name,
                            "sha256": _digest(features),
                        },
                        "declared_available_levels": ["partial_l0_entropy_only"],
                        "declared_missing_requirements": ["untouched_test"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    report = audit_retrospective_assets(config_path=config, repository_root=tmp_path)
    assert report["decision"] == "retrospective_assets_only_formal_matrix_incomplete"
    assert report["formal_matrix"]["completed_cells"] == 0
    assert report["formal_matrix"]["complete"] is False
    assert report["banks"]["opened"]["features"]["contents_not_loaded"] is True
    assert report["banks"]["opened"]["fixed_tool_headroom"]["mean_tool_cost"] == 4.0
