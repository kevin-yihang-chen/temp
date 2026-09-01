from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from beyond_entropy.sharding import stable_shard_index


ROOT = Path(__file__).resolve().parents[1]


def test_decar_full_worker_freezes_source_aligned_h800_generation() -> None:
    worker = (ROOT / "scripts/slurm_infographicvqa_decar_full_h800.sh").read_text()
    assert "#SBATCH --gres=gpu:h800:4" in worker
    assert "#SBATCH --time=08:15:00" in worker
    assert "#SBATCH --mail-user=yihangc@connect.hku.hk" in worker
    assert "#SBATCH --mail-type=ALL" in worker
    assert "full-qwen7b-v1" in worker
    assert worker.count("--shard-key source_id") >= 3
    assert worker.count("--shard-namespace") >= 3
    assert "infovqa-decar-full-shard-v1-06817" in worker
    assert "6014 6036 5910 5986|538 597 547 522" in worker
    assert "\"$(jq -r '.shard_key'" in worker
    assert "\"$(jq -r '.shard_namespace'" in worker
    assert "source_shards_disjoint" in worker
    assert "--expected-decisions 23946" in worker
    assert "--expected-records 119730" in worker
    assert "--expected-sources 2204" in worker
    assert "generated_token_statistics_complete:true" in worker
    assert "scientific_endpoints_used_for_selection:false" in worker
    assert "validation_or_test_inputs_used:false" in worker
    assert "HF_HUB_OFFLINE=1" in worker
    assert "unset HF_TOKEN HUGGINGFACE_HUB_TOKEN" in worker
    assert "--allow-download" not in worker


def test_decar_full_frozen_namespace_balances_source_indivisible_population() -> None:
    manifest = (
        ROOT
        / "artifacts/infographicvqa-train-v1/decar-v1/full-manifest-v1/task-manifest.jsonl"
    )
    weights: Counter[str] = Counter()
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            weights[json.loads(line)["source_id"]] += 1
    question_counts = [0, 0, 0, 0]
    source_counts = [0, 0, 0, 0]
    for source_id, weight in weights.items():
        shard = stable_shard_index(
            source_id,
            4,
            namespace="infovqa-decar-full-shard-v1-06817",
        )
        question_counts[shard] += weight
        source_counts[shard] += 1
    assert question_counts == [6014, 6036, 5910, 5986]
    assert source_counts == [538, 597, 547, 522]
    assert sum(question_counts) == 23946
    assert sum(source_counts) == 2204


def test_decar_full_submitter_requires_projection_reserve_and_clean_revision() -> None:
    submitter = (ROOT / "scripts/submit_infographicvqa_decar_full_h800.sh").read_text()
    assert "/usr/local/bin/show-cpu-gpu-quota" in submitter
    assert "-lt 1980" in submitter
    assert "--test-only --export=NONE" in submitter
    assert "--parsable --export=NONE" in submitter
    assert "git status --porcelain --untracked-files=no" in submitter
    assert "full-qwen7b-v1" in submitter
    assert "git push" not in submitter
