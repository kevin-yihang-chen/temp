#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
target_rollouts="${repo_dir}/artifacts/confirmation-chartqa-val-1918/qwen3b-c4-concise-seed0/rollouts.jsonl"

if [[ ! -r "${mail_file}" ]]; then
  echo "Missing private Slurm email file: ${mail_file}" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != *@* ]]; then
  echo "Invalid notification email in ${mail_file}" >&2
  exit 2
fi
if [[ ! -r "${target_rollouts}" ]]; then
  echo "Missing completed confirmation rollouts" >&2
  exit 2
fi
if [[ "$(wc -l < "${target_rollouts}")" -ne 9590 ]]; then
  echo "Confirmation rollouts are incomplete" >&2
  exit 2
fi

exec sbatch \
  --mail-user="${notify_email}" \
  --mail-type=ALL \
  "${repo_dir}/scripts/slurm_confirmation_action_contrasts.sh"
