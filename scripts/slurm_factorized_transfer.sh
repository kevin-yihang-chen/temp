#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-factorized-transfer-%j.out
#SBATCH --mail-type=ALL

set -euo pipefail

: "${BE_SOURCE_ROLLOUTS:?missing BE_SOURCE_ROLLOUTS}"
: "${BE_TARGET_ROLLOUTS:?missing BE_TARGET_ROLLOUTS}"
: "${BE_TARGET_MANIFEST:?missing BE_TARGET_MANIFEST}"
: "${BE_TRANSFER_OUTPUT_DIR:?missing BE_TRANSFER_OUTPUT_DIR}"

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python

export PYTHONPATH="${repo_dir}/src"

cd "${repo_dir}"
export BE_CODE_REVISION
BE_CODE_REVISION=$(git rev-parse HEAD)
"${python_bin}" scripts/analyze_factorized_transfer.py \
  --source-rollouts "${BE_SOURCE_ROLLOUTS}" \
  --target-rollouts "${BE_TARGET_ROLLOUTS}" \
  --target-manifest "${BE_TARGET_MANIFEST}" \
  --output-dir "${BE_TRANSFER_OUTPUT_DIR}" \
  --lambda-cost 0.05 \
  --bootstrap-resamples 5000 \
  --bootstrap-seed 0 \
  --seed 17
