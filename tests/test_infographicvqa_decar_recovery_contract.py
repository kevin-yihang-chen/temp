from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCIENTIFIC_REVISION = "5b1b0211372ccb96ec21fc55fa954d427a5504b5"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_recovery_worker_is_terminal_failure_only_and_scientifically_isolated() -> None:
    worker = (
        ROOT / "scripts/slurm_infographicvqa_decar_full_recovery_h800.sh"
    ).read_text()
    assert "#SBATCH --gres=gpu:h800:4" in worker
    assert "#SBATCH --time=08:15:00" in worker
    assert "#SBATCH --mail-user=yihangc@connect.hku.hk" in worker
    assert "#SBATCH --mail-type=ALL" in worker
    assert SCIENTIFIC_REVISION in worker
    assert "worktree add --detach" in worker
    assert 'ln -s "${live_manifest_dir}"' in worker
    assert 'ln -s "${live_root}"' in worker
    assert "BE_RECOVERY_SCIENTIFIC_REPO" in worker
    assert "scientific-worker-relocated.sh" in worker
    assert "8e2bb53dc067e0f81ee3372c3f468e3b56ef76ccbea0a6e4ffc6eda3a642f388" in worker
    assert 'bash "${runtime_worker}"' in worker
    assert '"${expected_generation_freeze_sha256}" 1 "${submit_epoch}"' in worker
    assert "isolated_scientific_worktree:true" in worker
    assert "launcher_revision" in worker
    assert "validation_or_test_inputs_used:false" in worker
    for state in (
        "TIMEOUT",
        "NODE_FAIL",
        "PREEMPTED",
        "FAILED",
        "OUT_OF_MEMORY",
        "CANCELLED",
    ):
        assert state in worker


def test_recovery_submitter_refuses_nonterminal_or_completed_predecessors() -> None:
    submitter = (
        ROOT / "scripts/submit_infographicvqa_decar_full_recovery_h800.sh"
    ).read_text()
    assert "--recover-from PRIOR_JOB_ID" in submitter
    assert "terminal unsuccessful prior job" in submitter
    assert "execution/job-${prior_job_id}.json" in submitter
    assert "/usr/local/bin/show-cpu-gpu-quota" in submitter
    assert "-lt 1980" in submitter
    assert "--test-only --export=NONE" in submitter
    assert "--parsable --export=NONE" in submitter
    assert "git push" not in submitter


def test_recovery_freeze_binds_exact_launcher_assets() -> None:
    worker = ROOT / "scripts/slurm_infographicvqa_decar_full_recovery_h800.sh"
    submitter = ROOT / "scripts/submit_infographicvqa_decar_full_recovery_h800.sh"
    freeze = (
        ROOT / "artifacts/docvqa-train-factorized-v2/ops/"
        "infographicvqa-decar-full-recovery-freeze-v1.md"
    ).read_text()
    assert f"{_sha256(worker)}  scripts/{worker.name}" in freeze
    assert f"{_sha256(submitter)}  scripts/{submitter.name}" in freeze
    assert SCIENTIFIC_REVISION in freeze
