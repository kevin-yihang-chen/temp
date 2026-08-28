#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
target_dir="${repo_dir}/data/cross-benchmark-v2/textvqa-fresh-formal"
manifest="${target_dir}/manifest.jsonl"
manifest_sha256=56973583dcd1aa8367d8a4e72f1c84f130864536dac128741a3914ae69ed901d
provenance_sha256=0acb0182e23d71ad07fdb12b761f22bf629b9e673aea16dae78270acbe5ab55a
development_audit_sha256=7302b97e9631b9488e4d4dff777a0de8c8a903d564eb0d51401f29e25953b5ab
prior_formal_audit_sha256=97882f42b2daf580f5be0c863e3a6834cd0347ba399bc73e771849ee0f4b7696
model="${repo_dir}/artifacts/textvqa-oof-factorized-action-value-attention-semantic-postfailure-v1/model.json"
model_sha256=f9b5dc897c5e8499ea5a245b0c512684579a5c6756da9196b628148ccf2c9a76
run_dir="${repo_dir}/artifacts/textvqa-attention-fresh-formal-v1/qwen3b-c4-seed0"

check_sha256() {
  local path=$1
  local expected=$2
  local actual
  actual=$(sha256sum "${path}")
  actual=${actual%% *}
  if [[ "${actual}" != "${expected}" ]]; then
    echo "SHA-256 mismatch: ${path}" >&2
    exit 2
  fi
}

if [[ ! -r "${mail_file}" ]]; then
  echo "Missing private Slurm email file: ${mail_file}" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != *@* ]]; then
  echo "Invalid notification email in ${mail_file}" >&2
  exit 2
fi
if [[ "$(wc -l < "${manifest}")" -ne 3166 ]]; then
  echo "Frozen fresh-split TextVQA manifest count mismatch" >&2
  exit 2
fi
check_sha256 "${manifest}" "${manifest_sha256}"
check_sha256 "${target_dir}/manifest.provenance.json" "${provenance_sha256}"
check_sha256 "${target_dir}/audit-vs-development.json" "${development_audit_sha256}"
check_sha256 "${target_dir}/audit-vs-prior-formal.json" "${prior_formal_audit_sha256}"
check_sha256 "${model}" "${model_sha256}"
code_revision=$(git -C "${repo_dir}" rev-parse HEAD)

exec sbatch \
  --job-name=be-textvqa-attn-fresh-formal \
  --mail-user="${notify_email}" \
  --mail-type=ALL \
  --export=ALL,BE_FORMAL_MANIFEST="${manifest}",BE_FORMAL_MANIFEST_SHA256="${manifest_sha256}",BE_FORMAL_RUN_DIR="${run_dir}",BE_CODE_REVISION="${code_revision}" \
  "${repo_dir}/scripts/slurm_textvqa_attention_fresh_formal.sh"

