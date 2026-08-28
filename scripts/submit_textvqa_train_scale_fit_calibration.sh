#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
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

fit_job_id=$(
  sbatch \
    --parsable \
    --dependency=afterok:190831 \
    --job-name=be-tvqa-scale-fit \
    --mail-user="${notify_email}" \
    --mail-type=ALL \
    "${repo_dir}/scripts/slurm_textvqa_train_scale_fit.sh"
)
fit_job_id=${fit_job_id%%;*}
echo "Submitted ranker fit job ${fit_job_id}"

calibration_job_id=$(
  sbatch \
    --parsable \
    --dependency="afterok:${fit_job_id}:190832" \
    --job-name=be-tvqa-scale-calibrate \
    --mail-user="${notify_email}" \
    --mail-type=ALL \
    "${repo_dir}/scripts/slurm_textvqa_train_scale_calibrate.sh"
)
calibration_job_id=${calibration_job_id%%;*}
echo "Submitted risk calibration job ${calibration_job_id}"
