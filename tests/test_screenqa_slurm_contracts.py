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
