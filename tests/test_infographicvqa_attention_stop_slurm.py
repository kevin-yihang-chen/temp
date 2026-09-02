from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_attention_stop_runner_is_descriptive_and_train_only() -> None:
    content = (
        ROOT / "scripts/diagnose_infographicvqa_attention_stop_factorization.py"
    ).read_text()
    assert "evaluate_attention_stop_factorization" in content
    assert "_first_frozen_difference" in content
    assert '"raw_entropy_policy_reproduced"' in content
    assert 'mmap_mode="r"' in content
    assert '"raw_negative_decision_preserved": True' in content
    assert '"valid_for_formal_selection": False' in content
    assert '"validation_or_test_inputs_used": False' in content
    assert "download" not in content.lower()


def test_attention_stop_worker_binds_inputs_hides_gpu_and_notifies() -> None:
    worker = (
        ROOT / "scripts/slurm_infographicvqa_attention_stop_factorization.sh"
    ).read_text()
    submitter = (
        ROOT / "scripts/submit_infographicvqa_attention_stop_factorization.sh"
    ).read_text()
    assert "#SBATCH --partition=debug" in worker
    assert "#SBATCH --gres=gpu:rtx_4090:1" in worker
    assert "#SBATCH --mail-user=yihangc@connect.hku.hk" in worker
    assert "#SBATCH --mail-type=ALL" in worker
    assert 'export CUDA_VISIBLE_DEVICES=""' in worker
    assert "unset HF_TOKEN HUGGINGFACE_HUB_TOKEN" in worker
    assert "009cdbfa1761f9b53b791a50f70b7e98bdf275eec8743d8bcaf078a52ded8ce8" in worker
    assert "validation_or_test_inputs_used:false" in worker
    assert "/usr/local/bin/show-cpu-gpu-quota" in submitter
    assert "-lt 45" in submitter
    assert "-lt 180" in submitter
    assert "sbatch --test-only --export=NONE" in submitter
    assert "sbatch --parsable --export=NONE" in submitter
    assert "git push" not in submitter
