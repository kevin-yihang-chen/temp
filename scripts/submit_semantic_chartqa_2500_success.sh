#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
rollouts="${repo_dir}/artifacts/gate1-chartqa-2500/qwen3b-c4-concise-seed0/rollouts.jsonl"
run_dir="${repo_dir}/artifacts/gate2-chartqa-2500/qwen3b-roi-concise-seed17"

if [[ ! -r "${mail_file}" ]]; then
  echo "Missing private Slurm email file: ${mail_file}" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != *@* ]]; then
  echo "Invalid notification email in ${mail_file}" >&2
  exit 2
fi
if [[ ! -r "${rollouts}" ]]; then
  echo "Missing rollout file: ${rollouts}" >&2
  exit 2
fi
actual_rows=$(wc -l < "${rollouts}")
if [[ "${actual_rows}" -ne 12500 ]]; then
  echo "Expected 12500 complete rollout rows, found ${actual_rows}" >&2
  exit 2
fi
rollouts_sha256=$(sha256sum "${rollouts}" | awk '{print $1}')

exec sbatch \
  --job-name=be-sem-success2500 \
  --mail-user="${notify_email}" \
  --mail-type=ALL \
  --export=ALL,BE_SEMANTIC_ROLLOUTS="${rollouts}",BE_SEMANTIC_ROLLOUTS_SHA256="${rollouts_sha256}",BE_SEMANTIC_RUN_DIR="${run_dir}",BE_SEMANTIC_MODEL_DIR="${run_dir}/semantic-model-oof-v1-success",BE_SEMANTIC_QUESTION_FEATURE_MODE=input_mean \
  "${repo_dir}/scripts/slurm_semantic_experiment.sh"
