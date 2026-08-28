#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 9 ]]; then
  echo "Usage: $0 POLICY_FREEZE_SHA256 MODEL_SHA256 MANIFEST_SHA256 AUDIT_SHA256 ROLLOUTS_SHA256 FEATURES_SHA256 PROTOCOL_SHA256 EVALUATOR_MODULE_SHA256 EVALUATOR_SCRIPT_SHA256" >&2
  exit 2
fi

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
policy_freeze="${repo_dir}/artifacts/textvqa-train-scale-v1/pairwise-attention-primary-v1/policy-freeze.json"
model="${repo_dir}/artifacts/textvqa-train-scale-v1/pairwise-attention-primary-v1/risk-calibrated/model.json"
manifest="${repo_dir}/data/textvqa-train-scale-v1/formal-test/manifest.jsonl"
audit="${repo_dir}/data/textvqa-train-scale-v1/formal-test.audit.json"
rollouts="${repo_dir}/artifacts/textvqa-train-scale-v1/formal-test/qwen3b-c4-seed0/rollouts.jsonl"
features="${repo_dir}/artifacts/textvqa-train-scale-v1/formal-test/attention-semantic-v1/features-question-region-attention-label-free.pt"
protocol="${repo_dir}/docs/scaled_textvqa_risk_control_preregistration.md"
evaluator_module="${repo_dir}/src/beyond_entropy/scaled_evaluation.py"
evaluator_script="${repo_dir}/scripts/evaluate_scaled_textvqa_action_value.py"
expected_freeze_sha256=$1
expected_model_sha256=$2
expected_manifest_sha256=$3
expected_audit_sha256=$4
expected_rollouts_sha256=$5
expected_features_sha256=$6
expected_protocol_sha256=$7
expected_evaluator_module_sha256=$8
expected_evaluator_script_sha256=$9

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
check_sha256 "${model}" "${expected_model_sha256}"
check_sha256 "${manifest}" "${expected_manifest_sha256}"
check_sha256 "${audit}" "${expected_audit_sha256}"
check_sha256 "${rollouts}" "${expected_rollouts_sha256}"
check_sha256 "${features}" "${expected_features_sha256}"
check_sha256 "${protocol}" "${expected_protocol_sha256}"
check_sha256 "${evaluator_module}" "${expected_evaluator_module_sha256}"
check_sha256 "${evaluator_script}" "${expected_evaluator_script_sha256}"
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

sbatch \
  --job-name=be-tvqa-scale-formal-eval \
  --mail-user="${notify_email}" \
  --mail-type=ALL \
  --export=ALL,BE_FORMAL_MODEL_SHA256="${expected_model_sha256}",BE_FORMAL_MANIFEST_SHA256="${expected_manifest_sha256}",BE_FORMAL_ROLLOUTS_SHA256="${expected_rollouts_sha256}",BE_FORMAL_FEATURES_SHA256="${expected_features_sha256}",BE_FORMAL_PROTOCOL_SHA256="${expected_protocol_sha256}",BE_FORMAL_EVALUATOR_MODULE_SHA256="${expected_evaluator_module_sha256}",BE_FORMAL_EVALUATOR_SCRIPT_SHA256="${expected_evaluator_script_sha256}",BE_FORMAL_EXPECTED_STATES="${expected_states}",BE_FORMAL_POLICY_FREEZE_SHA256="${expected_freeze_sha256}",BE_FORMAL_AUDIT_SHA256="${expected_audit_sha256}" \
  "${repo_dir}/scripts/slurm_textvqa_train_scale_formal_evaluate.sh"
