import json
from copy import deepcopy
from pathlib import Path

import pytest

from beyond_entropy.phase_c_training import (
    BENCHMARKS,
    METHODS,
    materialize_phase_c_seed_configs,
    sha256_file,
    validate_phase_c_training_matrix,
)
from scripts.train_sequential_post_training import validate_config


ROOT = Path(__file__).resolve().parents[1]


def _synthetic_matrix(tmp_path):
    matrix = json.loads(
        (ROOT / "configs/factorized_phase_c_training_matrix_v1.json").read_text()
    )
    for benchmark in BENCHMARKS:
        for role in ("train", "validation"):
            directory = tmp_path / benchmark / role
            directory.mkdir(parents=True)
            manifest = directory / "manifest.jsonl"
            rollouts = directory / "rollouts.jsonl"
            manifest.write_text(json.dumps({"state_id": f"{benchmark}-{role}"}) + "\n")
            rollouts.write_text(json.dumps({"state_id": f"{benchmark}-{role}"}) + "\n")
            matrix["datasets"][benchmark][role] = {
                "states": 1,
                "manifest": {
                    "path": str(manifest.relative_to(tmp_path)),
                    "sha256": sha256_file(manifest),
                },
                "rollouts": {
                    "path": str(rollouts.relative_to(tmp_path)),
                    "sha256": sha256_file(rollouts),
                },
            }
    return matrix


def test_materialized_phase_c_seed_configs_are_matched_and_three_domain(tmp_path):
    matrix = _synthetic_matrix(tmp_path)
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix))
    result = materialize_phase_c_seed_configs(
        matrix_path=matrix_path, repository_root=tmp_path,
        seed=29, output_dir=tmp_path / "frozen",
    )
    configs = [json.loads(Path(result["configs"][method]).read_text()) for method in METHODS]
    assert {config["method"] for config in configs} == set(METHODS)
    assert all(config["seed"] == 29 for config in configs)
    assert all(config["formal_claim_eligible"] is False for config in configs)
    assert validate_config(configs[0]) == tuple(sorted(BENCHMARKS))
    matched = []
    for config in configs:
        value = deepcopy(config)
        value.pop("method")
        matched.append(value)
    assert matched[0] == matched[1] == matched[2]
    evaluation = json.loads(Path(result["evaluation"]).read_text())
    assert evaluation["formal_claim_eligible"] is False
    assert set(evaluation["validation_rollouts"]) == set(BENCHMARKS)


def test_phase_c_matrix_rejects_any_heldout_path(tmp_path):
    matrix = _synthetic_matrix(tmp_path)
    heldout = tmp_path / "heldout" / "manifest.jsonl"
    heldout.parent.mkdir()
    heldout.write_text(json.dumps({"state_id": "forbidden"}) + "\n")
    matrix["datasets"]["chartqa"]["train"]["manifest"] = {
        "path": str(heldout.relative_to(tmp_path)), "sha256": sha256_file(heldout),
    }
    with pytest.raises(ValueError, match="must not reference held-out"):
        validate_phase_c_training_matrix(matrix, tmp_path)


def test_phase_c_slurm_contract_uses_three_parallel_arms_and_all_mail():
    worker = (ROOT / "scripts/slurm_factorized_phase_c_training.sh").read_text()
    submitter = (ROOT / "scripts/submit_factorized_phase_c_training.sh").read_text()
    assert "#SBATCH --gres=gpu:rtx_4090:3" in worker
    assert "#SBATCH --time=04:00:00" in worker
    assert "#SBATCH --mail-type=ALL" in worker
    assert "#SBATCH --no-requeue" in worker
    assert "seeds=(17 29 47)" in submitter
    assert "--mail-type=ALL" in submitter

