#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 POLICY_FREEZE_SHA256" >&2
  exit 2
fi

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
policy_freeze="${repo_dir}/artifacts/textvqa-train-scale-v1/pairwise-attention-primary-v1/policy-freeze.json"
output_dir="${repo_dir}/data/textvqa-train-scale-v1/formal-test"
audit_output="${repo_dir}/data/textvqa-train-scale-v1/formal-test.audit.json"
expected_freeze_sha256=$1

if [[ ! "${expected_freeze_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "Policy-freeze SHA-256 must contain 64 lowercase hex characters" >&2
  exit 2
fi
actual_freeze_sha256=$(sha256sum "${policy_freeze}")
actual_freeze_sha256=${actual_freeze_sha256%% *}
if [[ "${actual_freeze_sha256}" != "${expected_freeze_sha256}" ]]; then
  echo "Policy-freeze SHA-256 mismatch" >&2
  exit 2
fi
if [[ -e "${output_dir}" || -e "${audit_output}" ]]; then
  echo "Formal manifest or audit destination already exists" >&2
  exit 2
fi
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
  --job-name=be-tvqa-scale-formal-export \
  --mail-user="${notify_email}" \
  --mail-type=ALL \
  --export=ALL,BE_FORMAL_POLICY_FREEZE_SHA256="${expected_freeze_sha256}" \
  "${repo_dir}/scripts/slurm_textvqa_train_scale_formal_export.sh"
