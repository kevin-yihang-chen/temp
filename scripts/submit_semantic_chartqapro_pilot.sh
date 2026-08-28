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
  --job-name=be-sem-chartqapro-pilot \
  --mail-user="${notify_email}" \
  --mail-type=ALL \
  --export=ALL,BE_SEMANTIC_ROLLOUTS="${repo_dir}/artifacts/gate3-chartqapro-pilot-309/qwen3b-c4-direct-seed0/rollouts.jsonl",BE_SEMANTIC_ROLLOUTS_SHA256=f4483dbd24733ab698abd1a402333cd830aeae5c7818b5b38be14c60656dbcda,BE_SEMANTIC_RUN_DIR="${repo_dir}/artifacts/gate3-chartqapro-pilot-309/qwen3b-roi-direct-seed17",BE_SEMANTIC_MODEL_DIR="${repo_dir}/artifacts/gate3-chartqapro-pilot-309/qwen3b-roi-direct-seed17/semantic-model-v1",BE_SEMANTIC_QUESTION_FEATURE_MODE=input_mean \
  "${repo_dir}/scripts/slurm_semantic_experiment.sh"
