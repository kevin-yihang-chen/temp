from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_attention_where_worker_binds_resources_and_email() -> None:
    content = (
        ROOT / "scripts/slurm_infographicvqa_attention_where_h800.sh"
    ).read_text()
    assert "#SBATCH --partition=q-hgpu-small" in content
    assert "#SBATCH --gres=gpu:h800:2" in content
    assert "#SBATCH --mail-user=yihangc@connect.hku.hk" in content
    assert "#SBATCH --mail-type=ALL" in content
    assert "--top-layers 4" in content
    assert "--checkpoint-interval 512" in content
    assert "for wave_start in 0 2" in content
    assert "unset HF_TOKEN HUGGINGFACE_HUB_TOKEN" in content
    assert "validation_or_test_inputs_used:false" in content
    assert "outcomes_included:false" in content


def test_attention_where_submitter_requires_quota_and_test_only() -> None:
    content = (
        ROOT / "scripts/submit_infographicvqa_attention_where_h800.sh"
    ).read_text()
    assert "/usr/local/bin/show-cpu-gpu-quota" in content
    assert "-lt 720" in content
    assert "sbatch --test-only --export=NONE" in content
    assert "sbatch --parsable --export=NONE" in content
    assert "git status --porcelain --untracked-files=no" in content


def test_attention_where_evaluator_is_train_only_and_reuses_formal_bootstrap() -> None:
    content = (ROOT / "scripts/evaluate_infographicvqa_attention_where.py").read_text()
    assert 'mmap_mode="r"' in content
    assert "evaluate_attention_where" in content
    assert '"formal_bootstrap_reused": True' in content
    assert '"validation_opened": False' in content
    assert '"test_opened": False' in content
    assert '"validation_or_test_inputs_used": False' in content
    assert "download" not in content.lower()


def test_attention_where_evaluation_worker_binds_feature_job_and_notifies() -> None:
    worker = (
        ROOT / "scripts/slurm_infographicvqa_attention_where_evaluation.sh"
    ).read_text()
    submitter = (
        ROOT / "scripts/submit_infographicvqa_attention_where_evaluation.sh"
    ).read_text()
    assert "#SBATCH --partition=debug" in worker
    assert "#SBATCH --gres=gpu:rtx_4090:1" in worker
    assert "#SBATCH --cpus-per-task=4" in worker
    assert "#SBATCH --mem=64G" in worker
    assert "#SBATCH --time=00:45:00" in worker
    assert "#SBATCH --mail-user=yihangc@connect.hku.hk" in worker
    assert "#SBATCH --mail-type=ALL" in worker
    assert 'feature_execution="${attention_root}/execution/job-203257.json"' in worker
    assert 'export CUDA_VISIBLE_DEVICES=""' in worker
    assert "unset HF_TOKEN HUGGINGFACE_HUB_TOKEN" in worker
    assert "validation_or_test_inputs_used:false" in worker
    assert "infographicvqa-attention-where-evaluation-float-recovery-v1.md" in worker
    assert "--expected-evaluation-recovery-sha256" in worker
    assert "recovery_sha256" in submitter
    assert "-lt 45" in submitter
    assert "-lt 180" in submitter
    assert "sbatch --test-only --export=NONE" in submitter
    assert "sbatch --parsable --export=NONE" in submitter
    assert "git push" not in submitter
