#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
policy_freeze="${repo_dir}/artifacts/textvqa-train-factorized-v2/fixed-sequence-calibrated/policy-freeze.json"
output_dir="${repo_dir}/data/textvqa-train-factorized-v2/formal-test"
audit_output="${repo_dir}/data/textvqa-train-factorized-v2/formal-test.audit.json"

if [[ ! -r "${mail_file}" ]]; then
  echo "missing private Slurm email file" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != *@* ]]; then
  echo "invalid notification email" >&2
  exit 2
fi
if [[ ! -f "${policy_freeze}" ]]; then
  echo "successful calibration policy freeze does not exist" >&2
  exit 2
fi
if [[ -e "${output_dir}" || -e "${audit_output}" ]]; then
  echo "refusing to overwrite factorized-v2 formal export" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before formal export submission" >&2
  exit 2
fi

policy_freeze_sha256=$(sha256sum "${policy_freeze}")
policy_freeze_sha256=${policy_freeze_sha256%% *}
code_revision=$(git -C "${repo_dir}" rev-parse HEAD)
sbatch \
  --job-name=be-tvqa-fv2-formal-export \
  --mail-user="${notify_email}" \
  --mail-type=ALL \
  --export="ALL,BE_FV2_POLICY_FREEZE=${policy_freeze},BE_FV2_POLICY_FREEZE_SHA256=${policy_freeze_sha256},BE_CODE_REVISION=${code_revision}" \
  "${repo_dir}/scripts/slurm_textvqa_factorized_v2_formal_export.sh"
