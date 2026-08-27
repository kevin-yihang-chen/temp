#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
target_rollouts="${repo_dir}/artifacts/confirmation-chartqa-val-1918/qwen3b-c4-concise-seed0/rollouts.jsonl"
target_provenance="${repo_dir}/artifacts/confirmation-chartqa-val-1918/qwen3b-c4-concise-seed0/rollouts.provenance.json"
secondary_action_model="${repo_dir}/artifacts/gate2-transfer-chartqa-val/context-quadrant-secondary-v1/model.json"
secondary_source_report="${repo_dir}/artifacts/gate2-transfer-chartqa-val/context-quadrant-secondary-v1/report.json"

if [[ ! -r "${mail_file}" ]]; then
  echo "Missing private Slurm email file: ${mail_file}" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != *@* ]]; then
  echo "Invalid notification email in ${mail_file}" >&2
  exit 2
fi
target_rows=$(wc -l < "${target_rollouts}")
if [[ "${target_rows}" -ne 9590 ]]; then
  echo "Expected 9590 complete confirmation rows, found ${target_rows}" >&2
  exit 2
fi
if [[ ! -r "${target_provenance}" ]]; then
  echo "Missing completed rollout provenance: ${target_provenance}" >&2
  exit 2
fi
if [[ ! -r "${secondary_action_model}" || ! -r "${secondary_source_report}" ]]; then
  echo "Missing frozen secondary-policy inputs" >&2
  exit 2
fi

exec sbatch \
  --job-name=be-frozen-confirm \
  --mail-user="${notify_email}" \
  --mail-type=ALL \
  "${repo_dir}/scripts/slurm_frozen_confirmation_analysis.sh"
