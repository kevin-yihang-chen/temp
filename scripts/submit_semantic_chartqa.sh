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
  --job-name=be-sem-chartqa \
  --mail-user="${notify_email}" \
  --mail-type=ALL \
  --export=ALL,BE_SEMANTIC_ROLLOUTS="${repo_dir}/artifacts/gate1-chartqa-200/qwen3b-c4-seed0/rollouts.jsonl",BE_SEMANTIC_ROLLOUTS_SHA256=cdb23067943fc160cf793db0703ca4e2f0f5ffb6277f8d101248e68a85f69b31,BE_SEMANTIC_RUN_DIR="${repo_dir}/artifacts/gate2-chartqa-200/qwen3b-roi-seed17",BE_SEMANTIC_MODEL_DIR="${repo_dir}/artifacts/gate2-chartqa-200/qwen3b-roi-seed17/semantic-model-balanced-v2" \
  "${repo_dir}/scripts/slurm_semantic_experiment.sh"
