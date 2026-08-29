from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


SCRIPTS = {
    "rollout": Path("scripts/slurm_docvqa_train_factorized_v2_rollout.sh"),
    "features": Path("scripts/slurm_docvqa_train_factorized_v2_features.sh"),
    "fit": Path("scripts/slurm_docvqa_train_factorized_v2_fit.sh"),
}


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
