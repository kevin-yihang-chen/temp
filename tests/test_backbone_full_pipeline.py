from __future__ import annotations

import subprocess
from pathlib import Path


def test_qwen7b_full_worker_is_frozen_to_four_h800s() -> None:
    root = Path(__file__).resolve().parents[1]
    worker_path = root / "scripts/slurm_screenqa_backbone_7b_full_h800.sh"
    worker = worker_path.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(worker_path)], check=True)

    assert "#SBATCH --gres=gpu:h800:4" in worker
    assert "#SBATCH --mail-type=ALL" in worker
    assert "cc594898137f460bfe9f0759e9844b3ce807cfb5" in worker
    assert "--shard-count 4" in worker
    assert "--bootstrap-resamples 5000" in worker
    assert "--bootstrap-seed 20260903" in worker
    assert "runtime_measurement.peak_allocated_bytes" in worker
    assert "runtime_measurement.peak_reserved_bytes" in worker
    assert "protected_role_inputs_used" in worker
    assert "calibration_or_formal_inputs_used" in worker
    assert "task_endpoints_computed" in worker


def test_qwen7b_full_submitter_binds_quota_cache_and_mail() -> None:
    root = Path(__file__).resolve().parents[1]
    submit_path = root / "scripts/submit_screenqa_backbone_7b_full_h800.sh"
    submit = submit_path.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(submit_path)], check=True)

    assert "show-cpu-gpu-quota" in submit
    assert "/usr/local/slurm/bin/sinfo" in submit
    assert "/usr/local/slurm/bin/squeue" in submit
    assert "gpu:h800:4" in submit
    assert "gpu_remaining" in submit
    assert "-lt 240" in submit
    assert "local_files_only=True" in submit
    assert "HF_HUB_CACHE=/userhome/cs3/yihangc/Data/hf_cache" in submit
    assert "--mail-type=ALL" in submit
    assert "yihangc@connect.hku.hk" in submit
    assert "BE_BB7_SUBMIT_EPOCH" in submit
    assert "BE_BB7_FULL_RESUME" in submit


def test_qwen7b_decision_runner_binds_frozen_implementation() -> None:
    root = Path(__file__).resolve().parents[1]
    runner_path = root / "scripts/run_screenqa_backbone_7b_decision.sh"
    runner = runner_path.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(runner_path)], check=True)

    assert "b03a830b6af1e6b4d8f2e7ca99a52cc0" in runner
    assert "a69f4b098a2e3a7879728085b5efd1e2" in runner
    assert "08111d528284bb18cc422d5f6113e11b" in runner
    assert "--expected-report-sha256" in runner
    assert "--expected-protocol-sha256" in runner
    assert "visual_action_backbone_replication_decision_v1" in runner
    assert "score_threshold_selected" in runner
    assert "call_rate_selected" in runner
    assert "protected_outcome_used" in runner
