from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def _script(name: str) -> str:
    return (REPO / "scripts" / name).read_text(encoding="utf-8")


def test_factorized_formal_jobs_satisfy_cluster_gpu_and_email_contracts():
    job_scripts = (
        "slurm_textvqa_factorized_v2_formal_export.sh",
        "slurm_textvqa_factorized_v2_formal_rollout.sh",
        "slurm_textvqa_factorized_v2_formal_features.sh",
        "slurm_textvqa_factorized_v2_formal_evaluate.sh",
    )
    for name in job_scripts:
        body = _script(name)
        assert "#SBATCH --partition=debug" in body
        assert "#SBATCH --gres=gpu:rtx_4090:1" in body
        assert "#SBATCH --mail-type=ALL" in body


def test_factorized_formal_submissions_use_private_state_change_email():
    export = _script("submit_textvqa_factorized_v2_formal_export.sh")
    chain = _script("submit_textvqa_factorized_v2_formal.sh")
    for body, expected_jobs in ((export, 1), (chain, 3)):
        assert 'mail_file="${repo_dir}/.slurm-notify-email"' in body
        assert body.count('--mail-user="${notify_email}"') == expected_jobs
        assert body.count("--mail-type=ALL") == expected_jobs
