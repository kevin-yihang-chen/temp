#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
treatment_dir="${repo_dir}/artifacts/confirmation-chart-layout-2137/qwen3b-chart-layout-c4-concise-seed0"
treatment_rollouts="${treatment_dir}/rollouts.jsonl"
treatment_provenance="${treatment_dir}/rollouts.provenance.json"

if [[ ! -r "${mail_file}" ]]; then
  echo "Missing private Slurm email file: ${mail_file}" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != *@* ]]; then
  echo "Invalid notification email in ${mail_file}" >&2
  exit 2
fi
if [[ ! -r "${treatment_provenance}" ]]; then
  echo "Missing completed treatment provenance" >&2
  exit 2
fi
actual_rows=$(wc -l < "${treatment_rollouts}")
if [[ "${actual_rows}" -ne 10685 ]]; then
  echo "Expected 10685 treatment records, found ${actual_rows}" >&2
  exit 2
fi

exec sbatch \
  --mail-user="${notify_email}" \
  --mail-type=ALL \
  "${repo_dir}/scripts/slurm_chart_layout_confirmation_analysis.sh"
