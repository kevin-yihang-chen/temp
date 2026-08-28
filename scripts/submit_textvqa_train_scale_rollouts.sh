#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
allocation="${repo_dir}/data/textvqa-train-scale-v1/allocation.json"
allocation_audit="${repo_dir}/data/textvqa-train-scale-v1/allocation.audit.json"
allocation_sha256=da6d41584bf4f3bfb91426fa9fa3bcb61a659846147c279eaab2aedb776e1657
allocation_audit_sha256=303258b8e79d36e551dfd5b3d8632929b4c2cf192cdcff77c35de8d71b6f6186

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
check_sha256 "${allocation}" "${allocation_sha256}"
check_sha256 "${allocation_audit}" "${allocation_audit_sha256}"
code_revision=$(git -C "${repo_dir}" rev-parse HEAD)

submit_role() {
  local role=$1
  local expected_states=$2
  local manifest_sha256=$3
  local manifest_provenance_sha256=$4
  local scientific_status=$5
  local role_dir="${repo_dir}/data/textvqa-train-scale-v1/${role}"
  local manifest="${role_dir}/manifest.jsonl"
  local run_dir="${repo_dir}/artifacts/textvqa-train-scale-v1/${role}/qwen3b-c4-seed0"

  check_sha256 "${manifest}" "${manifest_sha256}"
  check_sha256 "${role_dir}/manifest.provenance.json" "${manifest_provenance_sha256}"
  if [[ "$(wc -l < "${manifest}")" -ne "${expected_states}" ]]; then
    echo "Frozen ${role} manifest count mismatch" >&2
    exit 2
  fi
  sbatch \
    --job-name="be-tvqa-scale-${role}" \
    --mail-user="${notify_email}" \
    --mail-type=ALL \
    --export=ALL,BE_SCALE_ROLE="${role}",BE_SCALE_MANIFEST="${manifest}",BE_SCALE_MANIFEST_SHA256="${manifest_sha256}",BE_SCALE_EXPECTED_STATES="${expected_states}",BE_SCALE_RUN_DIR="${run_dir}",BE_SCALE_SCIENTIFIC_STATUS="${scientific_status}",BE_CODE_REVISION="${code_revision}" \
    "${repo_dir}/scripts/slurm_textvqa_train_scale_rollout.sh"
}

submit_role \
  ranker-training \
  7912 \
  5a93e5279036db874076f0a5109ace91261f2416a48c3d397bc592d7d03c4468 \
  4369f318631d72091c3ca8894b11cb024396b08d13c80c5daebcb321bdcf701c \
  "scaled TextVQA train ranker-development sibling bank; outcomes may tune architectures"

submit_role \
  risk-calibration \
  4712 \
  423621b83ec3e4103be3ca8782fa659526612a231cc0e911c6231e4a2da747c8 \
  f8953dbfa386f383450133bb3e9ff37487f06a599bb34986e5cae2a18b183d79 \
  "scaled TextVQA train risk-calibration sibling bank; outcomes may calibrate frozen thresholds only"
