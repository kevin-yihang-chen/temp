from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_screenqa_ranker_recovery_only_fits_registered_spatial_candidate():
    content = (ROOT / "scripts/slurm_screenqa_ranker_fit_recovery.sh").read_text()
    assert "#SBATCH --time=04:00:00" in content
    assert "#SBATCH --gres=gpu:rtx_4090:1" in content
    assert "#SBATCH --cpus-per-task=4" in content
    assert "#SBATCH --mail-user=yihangc@connect.hku.hk" in content
    assert "#SBATCH --mail-type=ALL" in content
    assert content.count("scripts/fit_multidomain_action_value.py") == 1
    assert "--feature-mode spatial-context-geometry" in content
    assert "--model-family factorized-oof" in content
    assert "--oof-folds 5" in content
    assert "--bootstrap-resamples 2000" in content
    assert "--lambda-cost 0.05" in content
    assert "--seed 20260831" in content
    assert "scripts/select_screenqa_ranker_candidate.py" in content
    assert "ranker-fit-recovery.audit.json" in content
    assert "ScreenQA preserved context model contract mismatch" in content


def test_screenqa_ranker_recovery_submission_binds_timeout_and_preserved_hashes():
    content = (ROOT / "scripts/submit_screenqa_ranker_fit_recovery.sh").read_text()
    assert "previous_job_id=196911" in content
    assert "previous_job_state=TIMEOUT" in content
    assert "1174023b6ff4e00046eceb3783299aec286691e4" in content
    assert "069e1e69ed6c74fe3d3ec95a201e9a13cc43150ae8e3792b595922f81b6493e5" in content
    assert "3f1c0edf36832304808a57bd6cc34a702b5283716bb92d51c7c27d949f08174e" in content
    assert "0651debaeb5e742f6823e7321e8bfe8184a398a468e42e36dd033f68af74563c" in content
    assert "JobState=${previous_job_state}" in content
    assert "--mail-type=ALL" in content
    assert "tracked worktree must be clean" in content
    assert "BE_SCREENQA_RECOVERY_WORKER_SHA256" in content
