from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_phase_c_rollout_worker_uses_four_deterministic_gpu_shards():
    content = (ROOT / "scripts/slurm_factorized_phase_c_train_rollouts.sh").read_text()
    assert "#SBATCH --gres=gpu:rtx_4090:4" in content
    assert "#SBATCH --mail-type=ALL" in content
    assert "#SBATCH --no-requeue" in content
    assert "--shard-count \"${shard_count}\"" in content
    assert "--generation-seed 0" in content
    assert "--revision 66285546d2b821cf421d4f5eb2576359d3770cd3" in content
    assert "CUDA_VISIBLE_DEVICES=\"${allocated_gpus[${index}]}\"" in content
    assert "merge_sequential_rollout_shards.py" in content
    assert "--dataset-role train" in content
    assert "--features-output" not in content


def test_phase_c_rollout_submitter_covers_three_domains_and_notifications():
    content = (ROOT / "scripts/submit_factorized_phase_c_train_rollouts.sh").read_text()
    assert "requested=(chartqa docvqa hrbench)" in content
    assert "yihangc@connect.hku.hk" in content
    assert "--mail-type=ALL" in content
    assert "BE_PHASE_C_MANIFEST_SHA256" in content
    assert "BE_PHASE_C_ALLOCATION_REPORT_SHA256" in content
