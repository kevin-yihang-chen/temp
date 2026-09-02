from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_signed_stop_runner_freezes_single_primary_and_smoke_boundary() -> None:
    runner = (
        ROOT / "scripts/fit_infographicvqa_attention_signed_stop_oof.py"
    ).read_text()
    module = (
        ROOT / "src/beyond_entropy/infographicvqa_attention_signed_stop.py"
    ).read_text()
    assert "--smoke-only" in runner
    assert '"fit_performed": False' in module
    assert '"policy_metrics_computed": False' in module
    assert "ATTENTION_SIGNED_STOP_PRIMARY_RATE = 0.02" in module
    assert "ATTENTION_SIGNED_STOP_PRIMARY_CALLS = 479" in module
    assert "ATTENTION_SIGNED_STOP_C = 0.01" in module
    assert "class_weight=None" in module
    assert "_source_utility_weights" in module
    assert '"validation_or_test_inputs_used": False' in module


def test_signed_stop_worker_binds_inputs_hides_gpu_and_notifies() -> None:
    worker = (
        ROOT / "scripts/slurm_infographicvqa_attention_signed_stop_oof.sh"
    ).read_text()
    submitter = (
        ROOT / "scripts/submit_infographicvqa_attention_signed_stop_oof.sh"
    ).read_text()
    assert "#SBATCH --partition=debug" in worker
    assert "#SBATCH --gres=gpu:rtx_4090:1" in worker
    assert "#SBATCH --mail-user=yihangc@connect.hku.hk" in worker
    assert "#SBATCH --mail-type=ALL" in worker
    assert 'export CUDA_VISIBLE_DEVICES=""' in worker
    assert "unset HF_TOKEN HUGGINGFACE_HUB_TOKEN" in worker
    assert "f07eddb658444cd11ab67a62b53143c90ebf81a07026f00c7bba1411a3ad8e1a" in worker
    assert "validation_or_test_inputs_used:false" in worker
    assert "/usr/local/bin/show-cpu-gpu-quota" in submitter
    assert "-lt 45" in submitter
    assert "-lt 180" in submitter
    assert "sbatch --test-only --export=NONE" in submitter
    assert "sbatch --parsable --export=NONE" in submitter
    assert "git push" not in submitter
