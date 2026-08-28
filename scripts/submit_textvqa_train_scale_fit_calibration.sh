#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
ranker_feature_job_id=${BE_SCALE_RANKER_FEATURE_JOB_ID:-190831}
risk_calibration_feature_job_id=${BE_SCALE_RISK_CALIBRATION_FEATURE_JOB_ID:-190832}
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
    --dependency="afterok:${ranker_feature_job_id}" \
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
    --dependency="afterok:${fit_job_id}:${risk_calibration_feature_job_id}" \
    --job-name=be-tvqa-scale-calibrate \
    --mail-user="${notify_email}" \
    --mail-type=ALL \
    "${repo_dir}/scripts/slurm_textvqa_train_scale_calibrate.sh"
)
calibration_job_id=${calibration_job_id%%;*}
echo "Submitted risk calibration job ${calibration_job_id}"
echo "BE_SCALE_FIT_JOB_ID=${fit_job_id}"
echo "BE_SCALE_CALIBRATION_JOB_ID=${calibration_job_id}"
