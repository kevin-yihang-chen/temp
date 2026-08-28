#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
manifest="${repo_dir}/data/cross-benchmark-v1/docvqa-formal-v2/manifest.jsonl"
manifest_sha256=9ceb28d05df5feecedf6cf61fbbb27ce281b94dd027e5d6d6da43ddc091081ac
model="${repo_dir}/artifacts/docvqa-oof-factorized-action-value-context-v1/model.json"
model_sha256=33f2e0b1fd29e52c878bbbf2cd9819cd3c7e65e12afbabbdc5fa1f6687c8496b
run_dir="${repo_dir}/artifacts/docvqa-formal-v1/qwen3b-c4-seed0"

if [[ ! -r "${mail_file}" ]]; then
  echo "Missing private Slurm email file: ${mail_file}" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != *@* ]]; then
  echo "Invalid notification email in ${mail_file}" >&2
  exit 2
fi
if [[ "$(wc -l < "${manifest}")" -ne 1608 ]]; then
  echo "Frozen DocVQA formal manifest count mismatch" >&2
  exit 2
fi
actual_manifest_sha256=$(sha256sum "${manifest}")
actual_manifest_sha256=${actual_manifest_sha256%% *}
if [[ "${actual_manifest_sha256}" != "${manifest_sha256}" ]]; then
  echo "Frozen DocVQA formal manifest SHA-256 mismatch" >&2
  exit 2
fi
actual_model_sha256=$(sha256sum "${model}")
actual_model_sha256=${actual_model_sha256%% *}
if [[ "${actual_model_sha256}" != "${model_sha256}" ]]; then
  echo "Frozen DocVQA action-value model SHA-256 mismatch" >&2
  exit 2
fi
code_revision=$(git -C "${repo_dir}" rev-parse HEAD)

exec sbatch \
  --job-name=be-docvqa-formal \
  --mail-user="${notify_email}" \
  --mail-type=ALL \
  --export=ALL,BE_FORMAL_MANIFEST="${manifest}",BE_FORMAL_MANIFEST_SHA256="${manifest_sha256}",BE_FORMAL_RUN_DIR="${run_dir}",BE_CODE_REVISION="${code_revision}" \
  "${repo_dir}/scripts/slurm_docvqa_formal.sh"
