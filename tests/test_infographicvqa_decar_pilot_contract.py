from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_decar_pilot_worker_freezes_model_hardware_and_leakage_contracts() -> None:
    worker = (ROOT / "scripts/slurm_infographicvqa_decar_pilot_h800.sh").read_text()
    assert "#SBATCH --gres=gpu:h800:4" in worker
    assert "#SBATCH --mail-user=yihangc@connect.hku.hk" in worker
    assert "#SBATCH --mail-type=ALL" in worker
    assert "Qwen/Qwen2.5-VL-7B-Instruct" in worker
    assert "cc594898137f460bfe9f0759e9844b3ce807cfb5" in worker
    assert "--scorer docvqa" in worker
    assert "--candidate-count 4 --proposer ug-grid" in worker
    assert "--visual-crop-ratio 2.0 --visual-cost 1.0" in worker
    assert "--question-feature-mode contextual_text_mean" in worker
    assert "--exclude-outcomes" in worker
    assert "raw_targets_written" in worker
    assert "pilot-qwen7b-v2" in worker
    assert "audit_infographicvqa_decar_inputs.py" in worker
    assert "generated_token_statistics_complete == true" in worker
    assert "scientific_endpoints_reported == false" in worker
    assert "decar_input_audit_sha256" in worker
    assert "infographicvqa-decar-feature-implementation-correction-v1.md" in worker
    assert "task_endpoints_used_for_selection:false" in worker
    assert "HF_HUB_OFFLINE=1" in worker
    assert "unset HF_TOKEN HUGGINGFACE_HUB_TOKEN" in worker
    assert "--allow-download" not in worker


def test_decar_pilot_submitter_is_quota_gated_and_does_not_export_environment() -> None:
    submitter = (ROOT / "scripts/submit_infographicvqa_decar_pilot_h800.sh").read_text()
    assert "/usr/local/bin/show-cpu-gpu-quota" in submitter
    assert "-lt 720" in submitter
    assert "--test-only --export=NONE" in submitter
    assert "--parsable --export=NONE" in submitter
    assert "--resume" in submitter
    assert "pilot-qwen7b-v2" in submitter
    assert "pilot-implementation-freeze-v2.md" in submitter
    assert "git status --porcelain --untracked-files=no" in submitter
    assert "git push" not in submitter
