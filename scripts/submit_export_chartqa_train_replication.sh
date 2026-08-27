#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
external_dir="${repo_dir}/data/external/chartqa-hfm4-b605b6e-train"
parquets=(
  "${external_dir}/train-00000-of-00003-49492f364babfa44.parquet"
  "${external_dir}/train-00001-of-00003-7302bae5e425bbc7.parquet"
  "${external_dir}/train-00002-of-00003-194c9400785577a2.parquet"
)

if [[ ! -r "${mail_file}" ]]; then
  echo "Missing private Slurm email file: ${mail_file}" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != *@* ]]; then
  echo "Invalid notification email in ${mail_file}" >&2
  exit 2
fi
for parquet in "${parquets[@]}"; do
  if [[ ! -r "${parquet}" ]]; then
    echo "Missing ChartQA train parquet: ${parquet}" >&2
    exit 2
  fi
done

exec sbatch \
  --mail-user="${notify_email}" \
  --mail-type=ALL \
  "${repo_dir}/scripts/slurm_export_chartqa_train_replication.sh"
