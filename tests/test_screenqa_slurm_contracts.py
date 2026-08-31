from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_screenqa_smoke_slurm_contract_has_notifications_and_resume_audit():
    content = (ROOT / "scripts/slurm_screenqa_ranker_smoke.sh").read_text()
    assert "#SBATCH --array=0-1" in content
    assert "#SBATCH --gres=gpu:rtx_4090:1" in content
    assert "#SBATCH --mail-user=yihangc@connect.hku.hk" in content
    assert "#SBATCH --mail-type=ALL" in content
    assert "--scorer screenqa" in content
    assert "--shard-count" in content
    assert "--shard-index" in content
    assert '"${python_bin}" "${collect_args[@]}"' in content
    assert "rollouts_sha256_before_resume" in content
    assert "Tracked worktree must be clean" in content


def test_screenqa_smoke_merge_requires_all_resume_audits():
    content = (ROOT / "scripts/slurm_screenqa_ranker_smoke_merge.sh").read_text()
    assert "#SBATCH --gres=gpu:rtx_4090:1" in content
    assert "#SBATCH --mail-user=yihangc@connect.hku.hk" in content
    assert "#SBATCH --mail-type=ALL" in content
    assert "--require-resume-audit" in content
    assert "--expected-scorer screenqa" in content
    assert "sha256sum --check SHA256SUMS" in content


def test_screenqa_smoke_submission_chains_merge_and_validates_email():
    content = (ROOT / "scripts/submit_screenqa_ranker_smoke.sh").read_text()
    assert 'notify_email}" != "yihangc@connect.hku.hk"' in content
    assert '--dependency="afterok:${array_job_id}"' in content
    assert "--mail-type=ALL" in content
    assert "tracked worktree must be clean" in content
    assert "ranker-smoke-v1" in content


def test_screenqa_full_ranker_contract_uses_four_resumable_shards():
    content = (ROOT / "scripts/slurm_screenqa_ranker_full.sh").read_text()
    assert "#SBATCH --array=0-3" in content
    assert "#SBATCH --gres=gpu:rtx_4090:1" in content
    assert "#SBATCH --mail-user=yihangc@connect.hku.hk" in content
    assert "#SBATCH --mail-type=ALL" in content
    assert "--checkpoint-interval 32" in content
    assert "--scorer screenqa" in content
    assert "--shard-count" in content
    assert "--shard-index" in content
    assert "rollouts_sha256_before_resume" in content


def test_screenqa_full_ranker_submission_chains_strict_merge():
    submission = (ROOT / "scripts/submit_screenqa_ranker_full.sh").read_text()
    merge = (ROOT / "scripts/slurm_screenqa_ranker_full_merge.sh").read_text()
    assert "ranker-rollouts-v1" in submission
    assert '--dependency="afterok:${array_job_id}"' in submission
    assert "--mail-type=ALL" in submission
    assert "#SBATCH --gres=gpu:rtx_4090:1" in merge
    assert "--require-resume-audit" in merge
    assert "--bootstrap-resamples 5000" in merge
