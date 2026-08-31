#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
manifest_dir="${repo_dir}/artifacts/screenqa-train-factorized-v1/calibration-manifest-v1"
candidate_dir="${repo_dir}/artifacts/screenqa-train-factorized-v1/ranker-training/low-capacity-oof-v1/candidate-v1"
run_root="${repo_dir}/artifacts/screenqa-train-factorized-v1/calibration-rollouts-v1"
expected_states=9951
shard_count=4
python_bin=/userhome/cs3/yihangc/anaconda3/bin/python

if [[ ! -r "${mail_file}" ]]; then
  echo "missing private Slurm email file" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != "yihangc@connect.hku.hk" ]]; then
  echo "invalid ScreenQA notification email" >&2
  exit 2
fi
for required_file in manifest.jsonl manifest.audit.json SHA256SUMS; do
  if [[ ! -s "${manifest_dir}/${required_file}" ]]; then
    echo "ScreenQA calibration manifest is incomplete: ${required_file}" >&2
    exit 2
  fi
done
if [[ -e "${run_root}" ]]; then
  echo "refusing to reuse ScreenQA calibration run root: ${run_root}" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before ScreenQA calibration-bank submission" >&2
  exit 2
fi

manifest_sha256=$(sha256sum "${manifest_dir}/manifest.jsonl")
manifest_sha256=${manifest_sha256%% *}
manifest_audit_sha256=$(sha256sum "${manifest_dir}/manifest.audit.json")
manifest_audit_sha256=${manifest_audit_sha256%% *}
candidate_bundle_sha256=$(sha256sum "${candidate_dir}/SHA256SUMS")
candidate_bundle_sha256=${candidate_bundle_sha256%% *}
code_revision=$(git -C "${repo_dir}" rev-parse HEAD)

export_args="ALL,BE_SCREENQA_CALIBRATION_MANIFEST_DIR=${manifest_dir},BE_SCREENQA_CALIBRATION_MANIFEST_SHA256=${manifest_sha256},BE_SCREENQA_CALIBRATION_MANIFEST_AUDIT_SHA256=${manifest_audit_sha256},BE_SCREENQA_CANDIDATE_DIR=${candidate_dir},BE_SCREENQA_CANDIDATE_BUNDLE_SHA256=${candidate_bundle_sha256},BE_SCREENQA_CALIBRATION_EXPECTED_STATES=${expected_states},BE_SCREENQA_CALIBRATION_RUN_ROOT=${run_root},BE_SCREENQA_EXPECTED_CODE_REVISION=${code_revision},BE_SCREENQA_SHARD_COUNT=${shard_count}"

cd "${repo_dir}"
PYTHONPATH="${repo_dir}/src" "${python_bin}" -m \
  scripts.verify_screenqa_calibration_manifest \
  --manifest-dir "${manifest_dir}" \
  --candidate-dir "${candidate_dir}" \
  --expected-candidate-bundle-sha256 "${candidate_bundle_sha256}" \
  --expected-manifest-sha256 "${manifest_sha256}" \
  --expected-audit-sha256 "${manifest_audit_sha256}"

array_submission=$(
  sbatch \
    --mail-user="${notify_email}" \
    --mail-type=ALL \
    --export="${export_args}" \
    "${repo_dir}/scripts/slurm_screenqa_calibration_bank.sh"
)
array_job_id=${array_submission##* }
if [[ ! "${array_job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse ScreenQA calibration array job ID: ${array_submission}" >&2
  exit 2
fi
printf '%s\n' "${array_submission}"
printf 'screenqa_calibration_array_job_id=%s code_revision=%s\n' \
  "${array_job_id}" "${code_revision}"

merge_submission=$(
  sbatch \
    --dependency="afterok:${array_job_id}" \
    --mail-user="${notify_email}" \
    --mail-type=ALL \
    --export="${export_args}" \
    "${repo_dir}/scripts/slurm_screenqa_calibration_bank_merge.sh"
)
merge_job_id=${merge_submission##* }
if [[ ! "${merge_job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse ScreenQA calibration merge job ID: ${merge_submission}" >&2
  exit 2
fi
printf '%s\n' "${merge_submission}"
printf 'screenqa_calibration_merge_job_id=%s code_revision=%s candidate_bundle_sha256=%s\n' \
  "${merge_job_id}" "${code_revision}" "${candidate_bundle_sha256}"
