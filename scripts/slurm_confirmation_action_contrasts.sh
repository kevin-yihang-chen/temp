#!/usr/bin/env bash
#SBATCH --job-name=be-confirm-contrasts
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-confirmation-contrasts-%j.out
#SBATCH --mail-type=ALL

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python

export PYTHONPATH="${repo_dir}/src"
cd "${repo_dir}"
export BE_CODE_REVISION
BE_CODE_REVISION=$(git rev-parse HEAD)
"${python_bin}" scripts/analyze_confirmation_action_contrasts.py \
  --target-rollouts "${repo_dir}/artifacts/confirmation-chartqa-val-1918/qwen3b-c4-concise-seed0/rollouts.jsonl" \
  --frozen-model "${repo_dir}/artifacts/gate2-transfer-chartqa-vstar/factorized-context-v2-quantile/model.json" \
  --secondary-action-model "${repo_dir}/artifacts/gate2-transfer-chartqa-val/context-quadrant-secondary-v1/model.json" \
  --secondary-text-model "${repo_dir}/artifacts/gate2-transfer-chartqa-val/factorized-text-secondary-v1/model.json" \
  --output-dir "${repo_dir}/artifacts/confirmation-chartqa-val-1918/posthoc-action-contrasts-v1" \
  --bootstrap-resamples 5000
