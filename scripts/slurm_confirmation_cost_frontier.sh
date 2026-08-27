#!/usr/bin/env bash
#SBATCH --job-name=be-confirm-cost
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-confirmation-cost-%j.out
#SBATCH --mail-type=ALL

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python

export PYTHONPATH="${repo_dir}/src"
cd "${repo_dir}"
export BE_CODE_REVISION
BE_CODE_REVISION=$(git rev-parse HEAD)
"${python_bin}" scripts/analyze_confirmation_cost_frontier.py \
  --rollouts "${repo_dir}/artifacts/confirmation-chartqa-val-1918/qwen3b-c4-concise-seed0/rollouts.jsonl" \
  --frozen-model "${repo_dir}/artifacts/gate2-transfer-chartqa-vstar/factorized-context-v2-quantile/model.json" \
  --output-dir "${repo_dir}/artifacts/confirmation-chartqa-val-1918/posthoc-cost-frontier-v1" \
  --lambda-costs 0 0.01 0.025 0.05 0.075 0.1 0.125 \
  --bootstrap-resamples 5000
