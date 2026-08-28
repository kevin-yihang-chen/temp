#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-textvqa-train-scale-learning-curves-%j.out
#SBATCH --mail-type=ALL

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
allocation="${repo_dir}/data/textvqa-train-scale-v1/allocation.json"
rollouts="${repo_dir}/artifacts/textvqa-train-scale-v1/ranker-training/qwen3b-c4-seed0/rollouts.jsonl"
features="${repo_dir}/artifacts/textvqa-train-scale-v1/ranker-training/attention-semantic-v1/features-question-region-attention-label-free.pt"
output="${repo_dir}/artifacts/textvqa-train-scale-v1/pairwise-attention-primary-v1/learning-curves.json"
expected_allocation_sha256=da6d41584bf4f3bfb91426fa9fa3bcb61a659846147c279eaab2aedb776e1657

export PYTHONPATH="${repo_dir}/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${repo_dir}"
if [[ -e "${output}" ]]; then
  echo "Learning-curve output already exists" >&2
  exit 2
fi
rollouts_sha256=$(sha256sum "${rollouts}")
rollouts_sha256=${rollouts_sha256%% *}
features_sha256=$(sha256sum "${features}")
features_sha256=${features_sha256%% *}
"${python_bin}" scripts/fit_scaled_textvqa_learning_curves.py \
  --allocation "${allocation}" \
  --expected-allocation-sha256 "${expected_allocation_sha256}" \
  --rollouts "${rollouts}" \
  --expected-rollouts-sha256 "${rollouts_sha256}" \
  --features "${features}" \
  --expected-features-sha256 "${features_sha256}" \
  --output "${output}" \
  --bootstrap-resamples 2000
