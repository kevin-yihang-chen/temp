from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_screenqa_ranker_fit_is_frozen_low_capacity_source_oof():
    content = (ROOT / "scripts/slurm_screenqa_ranker_fit.sh").read_text()
    assert "#SBATCH --gres=gpu:rtx_4090:1" in content
    assert "#SBATCH --time=04:00:00" in content
    assert "#SBATCH --mail-user=yihangc@connect.hku.hk" in content
    assert "#SBATCH --mail-type=ALL" in content
    assert "--model-family factorized-oof" in content
    assert "--oof-folds 5" in content
    assert "--feature-mode context-geometry" in content
    assert "--feature-mode spatial-context-geometry" in content
    assert "--bootstrap-resamples 2000" in content
    assert "--lambda-cost 0.05" in content
    assert "--seed 20260831" in content
    assert "select_screenqa_ranker_candidate.py" in content


def test_screenqa_ranker_fit_submission_binds_protocol_and_emails():
    content = (ROOT / "scripts/submit_screenqa_ranker_fit.sh").read_text()
    assert "c6118d8a013a171c3eecad374a3271e3bf00dfd199864d3efaab27c7b44e36b7" in content
    assert "d1b8dd10524d8610b19c19a91450cd2d5eac2127" in content
    assert "--mail-type=ALL" in content
    assert "tracked worktree must be clean" in content
    assert "low-capacity-oof-v1" in content
