from pathlib import Path


def test_screenqa_semantic_feature_runner_is_frozen_label_free_and_resumable():
    root = Path(__file__).resolve().parents[1]
    runner = (root / "scripts/slurm_screenqa_semantic_features_4gpu.sh").read_text()
    assert "#SBATCH --gres=gpu:rtx_4090:4" in runner
    assert "#SBATCH --mail-user=yihangc@connect.hku.hk" in runner
    assert "#SBATCH --mail-type=ALL" in runner
    assert "verify_screenqa_semantic_activation.py" in runner
    assert "--exclude-outcomes" in runner
    assert "--checkpoint-interval 32" in runner
    assert "--batch-size 4" in runner
    assert "run_parallel_stage" in runner
    assert "merge_semantic_feature_shards.py" in runner
    assert "audit_label_free_semantic_features.py" in runner
    assert "outcomes_included_metadata" in runner
    assert "BE_SCREENQA_SEMANTIC_RESUME" in runner
    assert "calibration-manifest-v1" in runner
    assert "formal-manifest-v1" in runner


def test_screenqa_semantic_submitter_binds_activation_tools_and_notifications():
    root = Path(__file__).resolve().parents[1]
    submitter = (root / "scripts/submit_screenqa_semantic_features_4gpu.sh").read_text()
    assert "tracked worktree must be clean" in submitter
    assert "candidate.audit.json" in submitter
    assert "semantic_escalation" not in submitter or "verify_screenqa_semantic_activation.py" in submitter
    assert "925feba44324bf4e09aec5a7c162cc2f034bfb1e06cbae18fbaf1714d28d3a46" in submitter
    assert "28f3e3b06007cb9a14e7cdef0ec7a631a67581cb2a6618dd4249aec2d1da22f1" in submitter
    assert "3b1051ea28b07a5aefd70c4c347c43410c1023cc35eed739216dc0d0d1d3ff30" in submitter
    assert "--mail-user=\"${notify_email}\"" in submitter
    assert "--mail-type=ALL" in submitter
    assert "BE_SCREENQA_SEMANTIC_RUNNER_SHA256" in submitter


def test_promoted_semantic_sharding_tools_match_frozen_protocol_bytes():
    root = Path(__file__).resolve().parents[1]
    assert (
        root / "scripts/prepare_semantic_feature_batch_shards.py"
    ).read_bytes() == (
        root
        / "artifacts/docvqa-train-factorized-v2/ops/prepare_semantic_feature_batch_shards.py"
    ).read_bytes()
    assert (root / "scripts/merge_semantic_feature_shards.py").read_bytes() == (
        root
        / "artifacts/docvqa-train-factorized-v2/ops/merge_semantic_feature_shards.py"
    ).read_bytes()
