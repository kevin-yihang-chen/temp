#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
target_dir="${repo_dir}/artifacts/replication-chartqa-train-4500/qwen3b-c4-concise-seed0"
target_rollouts="${target_dir}/rollouts.jsonl"
target_provenance="${target_dir}/rollouts.provenance.json"

if [[ ! -r "${mail_file}" ]]; then
  echo "Missing private Slurm email file: ${mail_file}" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != *@* ]]; then
  echo "Invalid notification email in ${mail_file}" >&2
  exit 2
fi
if [[ "$(wc -l < "${target_rollouts}")" -ne 22500 ]]; then
  echo "Expected 22500 complete replication records" >&2
  exit 2
fi
if [[ ! -r "${target_provenance}" ]]; then
  echo "Missing completed replication provenance" >&2
  exit 2
fi

exec sbatch \
  --mail-user="${notify_email}" \
  --mail-type=ALL \
  "${repo_dir}/scripts/slurm_chartqa_train_replication_analysis.sh"
