from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


SCRIPTS = {
    "rollout": Path("scripts/slurm_docvqa_train_factorized_v2_rollout.sh"),
    "features": Path("scripts/slurm_docvqa_train_factorized_v2_features.sh"),
    "fit": Path("scripts/slurm_docvqa_train_factorized_v2_fit.sh"),
}
FORMAL_SCRIPTS = {
    "export": Path("scripts/slurm_docvqa_train_factorized_v2_formal_export.sh"),
    "rollout": Path("scripts/slurm_docvqa_train_factorized_v2_formal_rollout.sh"),
    "features": Path("scripts/slurm_docvqa_train_factorized_v2_formal_features.sh"),
    "evaluate": Path("scripts/slurm_docvqa_train_factorized_v2_formal_evaluate.sh"),
}
FORMAL_SUBMISSIONS = (
    Path("scripts/submit_docvqa_train_factorized_v2_formal_export.sh"),
    Path("scripts/submit_docvqa_train_factorized_v2_formal.sh"),
)
ALLOCATION_JOB = Path("scripts/slurm_docvqa_train_factorized_v2_allocation.sh")
ALLOCATION_SUBMISSION = Path(
    "scripts/submit_docvqa_train_factorized_v2_allocation.sh"
)


def test_docvqa_allocation_job_is_revision_locked_complete_and_emailed():
    subprocess.run(["bash", "-n", str(ALLOCATION_JOB)], check=True)
    content = ALLOCATION_JOB.read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:rtx_4090:1" in content
    assert "#SBATCH --mail-user=yihangc@connect.hku.hk" in content
    assert "#SBATCH --mail-type=ALL" in content
    assert "status --porcelain --untracked-files=no" in content
    assert "BE_DOCVQA_EXPECTED_CODE_REVISION" in content
    assert "for shard_index in {00..11}" in content
    assert content.count("--prior-manifest-root") == 2
    assert "verify_docvqa_train_factorized_v2_allocation.py" in content
    assert "--resume" in content


def test_docvqa_allocation_submission_uses_private_email_and_exact_revision():
    subprocess.run(["bash", "-n", str(ALLOCATION_SUBMISSION)], check=True)
    content = ALLOCATION_SUBMISSION.read_text(encoding="utf-8")
    assert ".slurm-notify-email" in content
    assert "yihangc@connect.hku.hk" in content
    assert '--mail-user="${notify_email}"' in content
    assert "--mail-type=ALL" in content
    assert "status --porcelain --untracked-files=no" in content
    assert "for shard_index in {00..11}" in content
    assert "BE_DOCVQA_EXPECTED_CODE_REVISION=${code_revision}" in content


@pytest.mark.parametrize("path", SCRIPTS.values())
def test_docvqa_slurm_wrapper_is_syntactically_valid_and_emails_every_state(path):
    subprocess.run(["bash", "-n", str(path)], check=True)
    content = path.read_text(encoding="utf-8")
    assert "#SBATCH --mail-user=yihangc@connect.hku.hk" in content
    assert "#SBATCH --mail-type=ALL" in content
    assert "status --porcelain --untracked-files=no" in content
    assert "BE_DOCVQA_EXPECTED_CODE_REVISION" in content


def test_docvqa_rollout_wrapper_freezes_model_actions_and_generation():
    content = SCRIPTS["rollout"].read_text(encoding="utf-8")
    for contract in (
        "--model Qwen/Qwen2.5-VL-3B-Instruct",
        "--model-revision 66285546d2b821cf421d4f5eb2576359d3770cd3",
        "--scorer docvqa",
        "--candidate-count 4",
        "--proposer ug-grid",
        "--visual-cost 1.0",
        "--generation-seeds 0",
        "--bootstrap-seed 20260829",
        "--max-new-tokens 32",
    ):
        assert contract in content
    assert "verify_docvqa_train_factorized_v2_manifest.py" in content
    assert "BE_DOCVQA_MANIFEST_AUDIT_SHA256" in content
    assert "BE_DOCVQA_FORMAL_OUTPUT_DIR" in content
    assert "formal_test" not in content


def test_docvqa_feature_wrapper_runs_exact_label_free_three_stage_pipeline():
    content = SCRIPTS["features"].read_text(encoding="utf-8")
    assert content.count("--exclude-outcomes") == 1
    assert "features-label-free.pt" in content
    assert "features-multimodal-label-free.pt" in content
    assert "features-question-region-attention-label-free.pt" in content
    assert "--mode multimodal-original" in content
    assert "--top-layers 4" in content
    assert "audit_docvqa_train_factorized_v2_rollouts.py" in content
    assert "audit_label_free_semantic_features.py" in content
    assert "verify_docvqa_train_factorized_v2_manifest.py" in content
    assert "BE_DOCVQA_MANIFEST_AUDIT_SHA256" in content


def test_docvqa_fit_wrapper_freezes_the_sole_preregistered_candidate():
    content = SCRIPTS["fit"].read_text(encoding="utf-8")
    for contract in (
        '--domain "docvqa=${BE_DOCVQA_RANKER_ROLLOUTS}"',
        '--features "docvqa=${BE_DOCVQA_RANKER_FEATURES}"',
        "--feature-mode hybrid-context-semantic",
        "--model-family factorized-oof",
        "--oof-folds 5",
        "--lambda-cost 0.05",
        "--alpha 1.0",
        "--seed 20260829",
    ):
        assert contract in content
    assert content.count("--alpha") == 1
    assert "BE_DOCVQA_RANKER_ROLLOUT_AUDIT_SHA256" in content
    assert "BE_DOCVQA_RANKER_LABEL_FREE_AUDIT_SHA256" in content


@pytest.mark.parametrize("path", FORMAL_SCRIPTS.values())
def test_docvqa_formal_slurm_wrappers_are_gpu_bound_hashed_and_emailed(path):
    subprocess.run(["bash", "-n", str(path)], check=True)
    content = path.read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:rtx_4090:1" in content
    assert "#SBATCH --mail-user=yihangc@connect.hku.hk" in content
    assert "#SBATCH --mail-type=ALL" in content
    assert "status --porcelain --untracked-files=no" in content
    assert "BE_DOCVQA_EXPECTED_CODE_REVISION" in content


def test_docvqa_formal_rollout_reverifies_gate_and_generation_contract():
    content = FORMAL_SCRIPTS["rollout"].read_text(encoding="utf-8")
    assert "verify_docvqa_train_factorized_v2_formal_gate.py" in content
    for contract in (
        "--model Qwen/Qwen2.5-VL-3B-Instruct",
        "--model-revision 66285546d2b821cf421d4f5eb2576359d3770cd3",
        "--scorer docvqa",
        "--candidate-count 4",
        "--proposer ug-grid",
        "--generation-seeds 0",
        "--bootstrap-seed 20260829",
        "--max-new-tokens 32",
        "--resume",
    ):
        assert contract in content


def test_docvqa_formal_features_reaudit_and_run_exact_label_free_pipeline():
    content = FORMAL_SCRIPTS["features"].read_text(encoding="utf-8")
    assert "verify_docvqa_train_factorized_v2_formal_gate.py" in content
    assert "audit_docvqa_train_factorized_v2_formal_rollouts.py" in content
    assert content.count("--exclude-outcomes") == 1
    assert "features-label-free.pt" in content
    assert "features-multimodal-label-free.pt" in content
    assert "features-question-region-attention-label-free.pt" in content
    assert "--mode multimodal-original" in content
    assert "--top-layers 4" in content
    assert "audit_label_free_semantic_features.py" in content


def test_docvqa_formal_evaluator_freezes_bootstrap_and_renders_once():
    content = FORMAL_SCRIPTS["evaluate"].read_text(encoding="utf-8")
    assert "evaluate_docvqa_train_factorized_v2_formal.py" in content
    assert "render_docvqa_train_factorized_v2_formal.py" in content
    assert "--bootstrap-resamples 20000" in content
    assert "--bootstrap-confidence 0.975" in content
    assert "--bootstrap-seed 20260829" in content
    assert "refusing to overwrite DocVQA one-shot formal result" in content


@pytest.mark.parametrize("path", FORMAL_SUBMISSIONS)
def test_docvqa_formal_submission_scripts_use_private_email_and_all_states(path):
    subprocess.run(["bash", "-n", str(path)], check=True)
    content = path.read_text(encoding="utf-8")
    assert '.slurm-notify-email' in content
    assert 'yihangc@connect.hku.hk' in content
    assert '--mail-user="${notify_email}"' in content
    assert "--mail-type=ALL" in content
    assert "status --porcelain --untracked-files=no" in content


def test_docvqa_formal_submission_is_one_afterok_chain():
    content = FORMAL_SUBMISSIONS[1].read_text(encoding="utf-8")
    assert content.count('--dependency="afterok:') == 2
    assert "slurm_docvqa_train_factorized_v2_formal_rollout.sh" in content
    assert "slurm_docvqa_train_factorized_v2_formal_features.sh" in content
    assert "slurm_docvqa_train_factorized_v2_formal_evaluate.sh" in content
    assert "verify_docvqa_train_factorized_v2_formal_gate.py" in content
    assert "refusing to reuse existing DocVQA formal outcomes" in content
