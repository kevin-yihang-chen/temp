#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
manifest="${repo_dir}/data/cross-benchmark-v1/textvqa-formal/manifest.jsonl"
manifest_sha256=847899f91147633186b61a802004c49cfe8ef3258427cb92ea390c891ec5ef2c
model="${repo_dir}/artifacts/textvqa-oof-factorized-action-value-context-v13/model.json"
model_sha256=ca224964aeb429478aeffaa3f084750cab05daf2c56be0b3f70fda68dceadc33
run_dir="${repo_dir}/artifacts/textvqa-formal-v1/qwen3b-c4-seed0"

if [[ ! -r "${mail_file}" ]]; then
  echo "Missing private Slurm email file: ${mail_file}" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != *@* ]]; then
  echo "Invalid notification email in ${mail_file}" >&2
  exit 2
fi
if [[ "$(wc -l < "${manifest}")" -ne 633 ]]; then
  echo "Frozen TextVQA formal manifest count mismatch" >&2
  exit 2
fi
actual_manifest_sha256=$(sha256sum "${manifest}")
actual_manifest_sha256=${actual_manifest_sha256%% *}
if [[ "${actual_manifest_sha256}" != "${manifest_sha256}" ]]; then
  echo "Frozen TextVQA formal manifest SHA-256 mismatch" >&2
  exit 2
fi
actual_model_sha256=$(sha256sum "${model}")
actual_model_sha256=${actual_model_sha256%% *}
if [[ "${actual_model_sha256}" != "${model_sha256}" ]]; then
  echo "Frozen TextVQA action-value model SHA-256 mismatch" >&2
  exit 2
fi
code_revision=$(git -C "${repo_dir}" rev-parse HEAD)

exec sbatch \
  --job-name=be-textvqa-formal \
  --mail-user="${notify_email}" \
  --mail-type=ALL \
  --export=ALL,BE_FORMAL_MANIFEST="${manifest}",BE_FORMAL_MANIFEST_SHA256="${manifest_sha256}",BE_FORMAL_RUN_DIR="${run_dir}",BE_CODE_REVISION="${code_revision}" \
  "${repo_dir}/scripts/slurm_textvqa_formal.sh"
