#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"

if [[ ! -r "${mail_file}" ]]; then
  echo "Missing private Slurm email file: ${mail_file}" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != *@* ]]; then
  echo "Invalid notification email in ${mail_file}" >&2
  exit 2
fi

exec sbatch \
  --job-name=be-sem-vstar191 \
  --mail-user="${notify_email}" \
  --mail-type=ALL \
  --export=ALL,BE_SEMANTIC_ROLLOUTS="${repo_dir}/artifacts/gate1-vstar-191/qwen3b-c4-seed0/rollouts.jsonl",BE_SEMANTIC_ROLLOUTS_SHA256=e4d1a0b0f0811233de117ef50e45433cd8eebc5369fcf62fdf5d97f1ca3b2aec,BE_SEMANTIC_RUN_DIR="${repo_dir}/artifacts/gate2-vstar-191/qwen3b-roi-contextual-seed17",BE_SEMANTIC_MODEL_DIR="${repo_dir}/artifacts/gate2-vstar-191/qwen3b-roi-contextual-seed17/semantic-model-oof-v4",BE_SEMANTIC_QUESTION_FEATURE_MODE=contextual_text_mean \
  "${repo_dir}/scripts/slurm_semantic_experiment.sh"
