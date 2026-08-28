#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
: "${BE_SCALE_FIT_JOB_ID:?missing BE_SCALE_FIT_JOB_ID}"
fit_job_id=${BE_SCALE_FIT_JOB_ID}
mail_file="${repo_dir}/.slurm-notify-email"
if [[ ! -r "${mail_file}" ]]; then
  echo "Missing private Slurm email file: ${mail_file}" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != *@* ]]; then
  echo "Invalid notification email in ${mail_file}" >&2
  exit 2
fi

sbatch \
  --dependency="afterok:${fit_job_id}" \
  --job-name=be-tvqa-scale-curves \
  --mail-user="${notify_email}" \
  --mail-type=ALL \
  "${repo_dir}/scripts/slurm_textvqa_train_scale_learning_curves.sh"
