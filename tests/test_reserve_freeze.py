from __future__ import annotations

from pathlib import Path

import pytest

from beyond_entropy.docvqa_reserve import (
    RESERVE_END_EXCLUSIVE,
    RESERVE_SOURCES,
    RESERVE_START,
)
from beyond_entropy.reserve_freeze import (
    MANDATORY_COMPONENTS,
    sha256_file,
    validate_reserve_freeze,
)


def _freeze(tmp_path: Path) -> dict[str, object]:
    components = {}
    for name in MANDATORY_COMPONENTS:
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        components[name] = {"path": str(path), "sha256": sha256_file(path)}
    return {
        "schema_version": 1,
        "code_revision": "a" * 40,
        "population": {
            "rank_start": RESERVE_START,
            "rank_end_exclusive": RESERVE_END_EXCLUSIVE,
            "expected_source_groups": RESERVE_SOURCES,
            "manifest_materialized": False,
            "rollouts_collected": False,
            "outcomes_used": False,
        },
        "components": components,
        "formal_outcomes_used": False,
        "reserve_outcomes_used": False,
    }


def test_validate_reserve_freeze_binds_all_components(tmp_path: Path):
    freeze = _freeze(tmp_path)
    validate_reserve_freeze(
        freeze, expected_code_revision="a" * 40, verify_components=True
    )


def test_validate_reserve_freeze_detects_component_drift(tmp_path: Path):
    freeze = _freeze(tmp_path)
    component = Path(str(freeze["components"]["allocation"]["path"]))  # type: ignore[index]
    component.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_reserve_freeze(freeze, verify_components=True)


def test_reserve_scripts_bind_hashes_and_keep_scoring_outcome_blind():
    repo = Path(__file__).resolve().parents[1]
    exporter = (repo / "scripts/export_docvqa_reserve_toolgate.py").read_text(
        encoding="utf-8"
    )
    scorer = (repo / "scripts/score_docvqa_reserve_toolgate.py").read_text(
        encoding="utf-8"
    )
    evaluator = (repo / "scripts/evaluate_docvqa_reserve_toolgate.py").read_text(
        encoding="utf-8"
    )
    assert "--expected-freeze-sha256" in exporter
    assert "select_reserve_identities" in exporter
    assert "_read_redacted_rollouts" in scorer
    assert 'payload["correct_before"] = 0.0' in scorer
    assert "--expected-scores-sha256" in evaluator
    assert "refusing to overwrite one-shot reserve result" in evaluator


def test_reserve_slurm_pipeline_uses_four_h800s_and_all_state_email():
    repo = Path(__file__).resolve().parents[1]
    exporter = (repo / "scripts/slurm_docvqa_reserve_toolgate_export.sh").read_text(
        encoding="utf-8"
    )
    pipeline = (
        repo / "scripts/slurm_docvqa_reserve_toolgate_pipeline.sh"
    ).read_text(encoding="utf-8")
    submitter = (repo / "scripts/submit_docvqa_reserve_toolgate.sh").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --mail-type=ALL" in exporter
    assert "#SBATCH --mail-type=ALL" in pipeline
    assert "#SBATCH --gres=gpu:h800:4" in pipeline
    assert "--bootstrap-resamples 20000" in pipeline
    assert "--mail-type=ALL" in submitter
    assert "--gres=gpu:h800:4" in submitter
