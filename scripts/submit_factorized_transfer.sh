#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
source_rollouts="${repo_dir}/artifacts/gate1-chartqa-2500/qwen3b-c4-concise-seed0/rollouts.jsonl"
target_rollouts="${repo_dir}/artifacts/gate1-vstar-191/qwen3b-c4-seed0/rollouts.jsonl"
target_manifest="${repo_dir}/data/vstar-frozen-191/manifest.jsonl"
output_dir="${repo_dir}/artifacts/gate2-transfer-chartqa-vstar/factorized-context-v1"

if [[ ! -r "${mail_file}" ]]; then
  echo "Missing private Slurm email file: ${mail_file}" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != *@* ]]; then
  echo "Invalid notification email in ${mail_file}" >&2
  exit 2
fi
source_rows=$(wc -l < "${source_rollouts}")
target_rows=$(wc -l < "${target_rollouts}")
if [[ "${source_rows}" -ne 12500 || "${target_rows}" -ne 955 ]]; then
  echo "Unexpected source/target rollout row counts: ${source_rows}/${target_rows}" >&2
  exit 2
fi
if [[ ! -r "${target_manifest}" ]]; then
  echo "Missing target manifest: ${target_manifest}" >&2
  exit 2
fi

exec sbatch \
  --job-name=be-chartqa-vstar \
  --mail-user="${notify_email}" \
  --mail-type=ALL \
  --export=ALL,BE_SOURCE_ROLLOUTS="${source_rollouts}",BE_TARGET_ROLLOUTS="${target_rollouts}",BE_TARGET_MANIFEST="${target_manifest}",BE_TRANSFER_OUTPUT_DIR="${output_dir}" \
  "${repo_dir}/scripts/slurm_factorized_transfer.sh"
