#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-contextual-question-diag-%j.out
#SBATCH --mail-type=ALL

set -euo pipefail

: "${BE_RESCUE_ROLLOUTS:?missing BE_RESCUE_ROLLOUTS}"
: "${BE_RESCUE_FEATURES:?missing BE_RESCUE_FEATURES}"
: "${BE_RESCUE_OUTPUT_ROOT:?missing BE_RESCUE_OUTPUT_ROOT}"

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python

export PYTHONPATH="${repo_dir}/src"

cd "${repo_dir}"
export BE_CODE_REVISION
BE_CODE_REVISION=$(git rev-parse HEAD)
"${python_bin}" scripts/analyze_nested_oof_rescue_gate.py \
  --rollouts "${BE_RESCUE_ROLLOUTS}" \
  --features "${BE_RESCUE_FEATURES}" \
  --output-dir "${BE_RESCUE_OUTPUT_ROOT}/nested-oof" \
  --feature-modes semantic context semantic-context \
  --outer-folds 5 \
  --seed 17 \
  --lambda-cost 0.05 \
  --bootstrap-resamples 5000 \
  --bootstrap-seed 0
"${python_bin}" scripts/analyze_two_stage_rescue_gate.py \
  --rollouts "${BE_RESCUE_ROLLOUTS}" \
  --features "${BE_RESCUE_FEATURES}" \
  --output-dir "${BE_RESCUE_OUTPUT_ROOT}/two-stage" \
  --outer-folds 5 \
  --seed 17 \
  --lambda-cost 0.05 \
  --bootstrap-resamples 5000 \
  --bootstrap-seed 0
"${python_bin}" scripts/analyze_factorized_rescue_gate.py \
  --rollouts "${BE_RESCUE_ROLLOUTS}" \
  --features "${BE_RESCUE_FEATURES}" \
  --output-dir "${BE_RESCUE_OUTPUT_ROOT}/factorized" \
  --rescue-feature-modes semantic context semantic-context \
  --outer-folds 5 \
  --seed 17 \
  --lambda-cost 0.05 \
  --bootstrap-resamples 5000 \
  --bootstrap-seed 0
