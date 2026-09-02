from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_literature_attention_worker_binds_h800_protocol_and_email() -> None:
    content = (
        ROOT / "scripts/slurm_infographicvqa_literature_attention_h800.sh"
    ).read_text()
    assert "#SBATCH --partition=q-hgpu-small" in content
    assert "#SBATCH --gres=gpu:h800:2" in content
    assert "#SBATCH --time=08:00:00" in content
    assert "#SBATCH --mail-user=yihangc@connect.hku.hk" in content
    assert "#SBATCH --mail-type=ALL" in content
    assert "--checkpoint-interval 256" in content
    assert "for wave_start in 0 2" in content
    for expected_feature_sha256 in (
        "2ef27cfe17b5d8d36bd4410850a24e982b23f7686a9fa064c48db73c1ba0f3da",
        "ed643ad1d4b82500db3dd3cec6f7d6d01412cef90e7f55690ad2e57b70cabdeb",
        "6cf284fec70ad2873ff05a1ef17ab0958ab57d92bc1b340c03a749fdb470a69b",
        "4eb20a4d9ca35b693889406eb82c74f1c635e93ab07ec74917fe0976773d948e",
    ):
        assert expected_feature_sha256 in content
    assert "unset HF_TOKEN HUGGINGFACE_HUB_TOKEN" in content
    assert "validation_or_test_inputs_used:false" in content
    assert "outcomes_included:false" in content
    assert "git push" not in content


def test_literature_attention_submitter_requires_quota_and_test_only() -> None:
    content = (
        ROOT / "scripts/submit_infographicvqa_literature_attention_h800.sh"
    ).read_text()
    assert "/usr/local/bin/show-cpu-gpu-quota" in content
    assert "-lt 960" in content
    assert "sbatch --test-only --export=NONE" in content
    assert "sbatch --parsable --export=NONE" in content
    assert "git status --porcelain --untracked-files=no" in content


def test_literature_attention_evaluator_is_train_only_and_multiplicity_corrected() -> (
    None
):
    runner = (
        ROOT / "scripts/evaluate_infographicvqa_literature_attention_where.py"
    ).read_text()
    module = (
        ROOT / "src/beyond_entropy/infographicvqa_literature_attention_evaluation.py"
    ).read_text()
    assert 'mmap_mode="r"' in runner
    assert "evaluate_literature_attention_where" in runner
    assert '"multiplicity_corrected": True' in runner
    assert '"validation_opened": False' in runner
    assert '"test_opened": False' in runner
    assert "LITERATURE_ATTENTION_CI_LOW = 0.0125" in module
    assert '"raw_attention_where"' in module
    assert "download" not in runner.lower()


def test_literature_attention_evaluation_worker_hides_reserved_gpu_and_notifies() -> (
    None
):
    worker = (
        ROOT / "scripts/slurm_infographicvqa_literature_attention_evaluation.sh"
    ).read_text()
    submitter = (
        ROOT / "scripts/submit_infographicvqa_literature_attention_evaluation.sh"
    ).read_text()
    assert "#SBATCH --partition=debug" in worker
    assert "#SBATCH --gres=gpu:rtx_4090:1" in worker
    assert "#SBATCH --mail-user=yihangc@connect.hku.hk" in worker
    assert "#SBATCH --mail-type=ALL" in worker
    assert 'export CUDA_VISIBLE_DEVICES=""' in worker
    assert "unset HF_TOKEN HUGGINGFACE_HUB_TOKEN" in worker
    assert "validation_or_test_inputs_used:false" in worker
    assert "feature_code_revision=$7" in worker
    assert '--expected-literature-code-revision "${feature_code_revision}"' in worker
    assert "feature_code_revision=$(jq -r" in submitter
    assert "-lt 60" in submitter
    assert "-lt 240" in submitter
    assert "sbatch --test-only --export=NONE" in submitter
    assert "sbatch --parsable --export=NONE" in submitter
    assert "git push" not in submitter
