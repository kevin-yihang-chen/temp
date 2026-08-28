#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-textvqa-train-scale-calibrate-%j.out
#SBATCH --mail-type=ALL

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
ranker_model="${repo_dir}/artifacts/textvqa-train-scale-v1/pairwise-attention-primary-v1/ranker-development/model.json"
rollouts="${repo_dir}/artifacts/textvqa-train-scale-v1/risk-calibration/qwen3b-c4-seed0/rollouts.jsonl"
feature_dir="${repo_dir}/artifacts/textvqa-train-scale-v1/risk-calibration/attention-semantic-v1"
features="${feature_dir}/features-question-region-attention-label-free.pt"
output_dir="${repo_dir}/artifacts/textvqa-train-scale-v1/pairwise-attention-primary-v1/risk-calibrated"

export PYTHONPATH="${repo_dir}/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${repo_dir}"
test -s "${feature_dir}/rollouts.audit.json"
test -s "${feature_dir}/label-free-audit.json"
ranker_model_sha256=$(sha256sum "${ranker_model}")
ranker_model_sha256=${ranker_model_sha256%% *}
rollouts_sha256=$(sha256sum "${rollouts}")
rollouts_sha256=${rollouts_sha256%% *}
features_sha256=$(sha256sum "${features}")
features_sha256=${features_sha256%% *}
echo "Frozen pre-calibration ranker SHA-256: ${ranker_model_sha256}"
"${python_bin}" scripts/calibrate_scaled_textvqa_action_value.py \
  --ranker-model "${ranker_model}" \
  --expected-ranker-model-sha256 "${ranker_model_sha256}" \
  --rollouts "${rollouts}" \
  --expected-rollouts-sha256 "${rollouts_sha256}" \
  --features "${features}" \
  --expected-features-sha256 "${features_sha256}" \
  --output-dir "${output_dir}"
