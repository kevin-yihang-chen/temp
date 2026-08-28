#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 5 ]]; then
  echo "Usage: $0 POLICY_FREEZE_SHA256 MANIFEST_SHA256 PROVENANCE_SHA256 AUDIT_SHA256 MODEL_SHA256" >&2
  exit 2
fi

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
policy_freeze="${repo_dir}/artifacts/textvqa-train-scale-v1/pairwise-attention-primary-v1/policy-freeze.json"
formal_dir="${repo_dir}/data/textvqa-train-scale-v1/formal-test"
manifest="${formal_dir}/manifest.jsonl"
provenance="${formal_dir}/manifest.provenance.json"
audit="${repo_dir}/data/textvqa-train-scale-v1/formal-test.audit.json"
model="${repo_dir}/artifacts/textvqa-train-scale-v1/pairwise-attention-primary-v1/risk-calibrated/model.json"
run_dir="${repo_dir}/artifacts/textvqa-train-scale-v1/formal-test/qwen3b-c4-seed0"
expected_freeze_sha256=$1
expected_manifest_sha256=$2
expected_provenance_sha256=$3
expected_audit_sha256=$4
expected_model_sha256=$5
scientific_status="scaled TextVQA one-shot formal sibling bank; frozen risk-controlled policy; no target-derived tuning"

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
check_sha256 "${policy_freeze}" "${expected_freeze_sha256}"
check_sha256 "${manifest}" "${expected_manifest_sha256}"
check_sha256 "${provenance}" "${expected_provenance_sha256}"
check_sha256 "${audit}" "${expected_audit_sha256}"
check_sha256 "${model}" "${expected_model_sha256}"
PYTHONPATH="${repo_dir}/src" /userhome/cs3/yihangc/anaconda3/bin/python \
  "${repo_dir}/scripts/verify_scaled_textvqa_formal_gate.py" \
  --policy-freeze "${policy_freeze}" \
  --expected-policy-freeze-sha256 "${expected_freeze_sha256}" \
  --model "${model}" \
  --expected-model-sha256 "${expected_model_sha256}" \
  --manifest "${manifest}" \
  --expected-manifest-sha256 "${expected_manifest_sha256}" \
  --audit "${audit}" \
  --expected-audit-sha256 "${expected_audit_sha256}"
expected_states=$(wc -l < "${manifest}")
if [[ "${expected_states}" -lt 5000 ]]; then
  echo "Formal manifest must contain at least one state per allocated source" >&2
  exit 2
fi
if [[ -e "${run_dir}/rollouts.provenance.json" ]]; then
  echo "Formal rollout is already complete; refusing a second submission" >&2
  exit 2
fi
code_revision=$(git -C "${repo_dir}" rev-parse HEAD)

sbatch \
  --job-name=be-tvqa-scale-formal \
  --mail-user="${notify_email}" \
  --mail-type=ALL \
  --export=ALL,BE_SCALE_ROLE=formal-test,BE_SCALE_MANIFEST="${manifest}",BE_SCALE_MANIFEST_SHA256="${expected_manifest_sha256}",BE_SCALE_EXPECTED_STATES="${expected_states}",BE_SCALE_RUN_DIR="${run_dir}",BE_SCALE_SCIENTIFIC_STATUS="${scientific_status}",BE_CODE_REVISION="${code_revision}",BE_SCALE_POLICY_FREEZE="${policy_freeze}",BE_SCALE_POLICY_FREEZE_SHA256="${expected_freeze_sha256}",BE_SCALE_FROZEN_MODEL="${model}",BE_SCALE_FROZEN_MODEL_SHA256="${expected_model_sha256}",BE_SCALE_FORMAL_AUDIT="${audit}",BE_SCALE_FORMAL_AUDIT_SHA256="${expected_audit_sha256}" \
  "${repo_dir}/scripts/slurm_textvqa_train_scale_rollout.sh"
