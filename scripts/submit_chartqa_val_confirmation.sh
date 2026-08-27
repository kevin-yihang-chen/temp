#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
manifest="${repo_dir}/data/chartqa-val-confirmation-1918/manifest.jsonl"

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
if [[ "${actual_rows}" -ne 1918 ]]; then
  echo "Expected 1918 confirmation states, found ${actual_rows}" >&2
  exit 2
fi
if [[ "${actual_sha256}" != d3178218853b10447228963e839716f0eac768b51bdc0f5b4a83268d3819b58b ]]; then
  echo "Confirmation manifest SHA-256 mismatch" >&2
  exit 2
fi

exec sbatch \
  --mail-user="${notify_email}" \
  --mail-type=ALL \
  "${repo_dir}/scripts/slurm_chartqa_val_confirmation.sh"
