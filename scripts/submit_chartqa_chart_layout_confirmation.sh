#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
manifest="${repo_dir}/data/chartqa-frozen-2500/manifest.chart-layout-confirm.jsonl"

if [[ ! -r "${mail_file}" ]]; then
  echo "Missing private Slurm email file: ${mail_file}" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != *@* ]]; then
  echo "Invalid notification email in ${mail_file}" >&2
  exit 2
fi
actual_rows=$(wc -l < "${manifest}")
actual_sha256=$(sha256sum "${manifest}" | awk '{print $1}')
if [[ "${actual_rows}" -ne 2137 ]]; then
  echo "Expected 2137 chart-layout confirmation states, found ${actual_rows}" >&2
  exit 2
fi
if [[ "${actual_sha256}" != d7c96df369259c8c3645bf64c27c220936636c92e359171a50e420344c5ff0bd ]]; then
  echo "Chart-layout confirmation manifest SHA-256 mismatch" >&2
  exit 2
fi

exec sbatch \
  --mail-user="${notify_email}" \
  --mail-type=ALL \
  "${repo_dir}/scripts/slurm_chartqa_chart_layout_confirmation.sh"
