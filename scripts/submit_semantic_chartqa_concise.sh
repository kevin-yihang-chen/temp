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
  --job-name=be-sem-chartqa-concise \
  --mail-user="${notify_email}" \
  --mail-type=ALL \
  --export=ALL,BE_SEMANTIC_ROLLOUTS="${repo_dir}/artifacts/gate1-chartqa-200/qwen3b-c4-concise-seed0/rollouts.jsonl",BE_SEMANTIC_ROLLOUTS_SHA256=b93a35e2bd586e253f27820421964c084b5b2f46ccabf537e4ed47fdb6a8118a,BE_SEMANTIC_RUN_DIR="${repo_dir}/artifacts/gate2-chartqa-200/qwen3b-roi-concise-seed17",BE_SEMANTIC_MODEL_DIR="${repo_dir}/artifacts/gate2-chartqa-200/qwen3b-roi-concise-seed17/semantic-model-oof-v5",BE_SEMANTIC_QUESTION_FEATURE_MODE=input_mean \
  "${repo_dir}/scripts/slurm_semantic_experiment.sh"
