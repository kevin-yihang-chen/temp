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

submit_role() {
  local role=$1
  local rollout_job_id=$2
  local expected_states=$3
  local manifest_sha256=$4
  local scientific_status=$5
  local manifest="${repo_dir}/data/textvqa-train-scale-v1/${role}/manifest.jsonl"
  local run_dir="${repo_dir}/artifacts/textvqa-train-scale-v1/${role}/qwen3b-c4-seed0"
  local feature_dir="${repo_dir}/artifacts/textvqa-train-scale-v1/${role}/attention-semantic-v1"

  sbatch \
    --dependency="afterok:${rollout_job_id}" \
    --job-name="be-tvqa-feat-${role}" \
    --mail-user="${notify_email}" \
    --mail-type=ALL \
    --export=ALL,BE_SCALE_ROLE="${role}",BE_SCALE_MANIFEST="${manifest}",BE_SCALE_MANIFEST_SHA256="${manifest_sha256}",BE_SCALE_EXPECTED_STATES="${expected_states}",BE_SCALE_ROLLOUTS="${run_dir}/rollouts.jsonl",BE_SCALE_SCIENTIFIC_STATUS="${scientific_status}",BE_SCALE_FEATURE_DIR="${feature_dir}" \
    "${repo_dir}/scripts/slurm_textvqa_train_scale_features.sh"
}

submit_role \
  ranker-training \
  190829 \
  7912 \
  5a93e5279036db874076f0a5109ace91261f2416a48c3d397bc592d7d03c4468 \
  "scaled TextVQA train ranker-development sibling bank; outcomes may tune architectures"

submit_role \
  risk-calibration \
  190830 \
  4712 \
  423621b83ec3e4103be3ca8782fa659526612a231cc0e911c6231e4a2da747c8 \
  "scaled TextVQA train risk-calibration sibling bank; outcomes may calibrate frozen thresholds only"
