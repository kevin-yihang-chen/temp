#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
rollouts="${repo_dir}/artifacts/gate1-chartqa-2500/qwen3b-c4-concise-seed0/rollouts.jsonl"
source_features="${repo_dir}/artifacts/gate2-chartqa-2500/qwen3b-roi-concise-seed17/features.pt"
output_features="${repo_dir}/artifacts/gate2-chartqa-2500/qwen3b-roi-contextual-question-seed17/features.pt"

if [[ ! -r "${mail_file}" ]]; then
  echo "Missing private Slurm email file: ${mail_file}" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != *@* ]]; then
  echo "Invalid notification email in ${mail_file}" >&2
  exit 2
fi
if [[ ! -r "${source_features}" ]]; then
  echo "Missing source semantic features: ${source_features}" >&2
  exit 2
fi
actual_rows=$(wc -l < "${rollouts}")
if [[ "${actual_rows}" -ne 12500 ]]; then
  echo "Expected 12500 complete rollout rows, found ${actual_rows}" >&2
  exit 2
fi

exec sbatch \
  --job-name=be-question-reembed \
  --mail-user="${notify_email}" \
  --mail-type=ALL \
  --export=ALL,BE_SEMANTIC_SOURCE_FEATURES="${source_features}",BE_SEMANTIC_ROLLOUTS="${rollouts}",BE_SEMANTIC_OUTPUT_FEATURES="${output_features}" \
  "${repo_dir}/scripts/slurm_contextual_question_reembed.sh"
