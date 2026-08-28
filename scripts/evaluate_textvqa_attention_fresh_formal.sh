#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 EXPECTED_ROLLOUTS_SHA256 EXPECTED_FEATURES_SHA256" >&2
  exit 2
fi

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
manifest="${repo_dir}/data/cross-benchmark-v2/textvqa-fresh-formal/manifest.jsonl"
rollouts="${repo_dir}/artifacts/textvqa-attention-fresh-formal-v1/qwen3b-c4-seed0/rollouts.jsonl"
model="${repo_dir}/artifacts/textvqa-oof-factorized-action-value-attention-semantic-postfailure-v1/model.json"
feature_dir="${repo_dir}/artifacts/textvqa-attention-fresh-formal-v1/attention-semantic-v1"
features="${feature_dir}/features-question-region-attention-label-free.pt"
output="${feature_dir}/evaluation.json"
expected_rollouts_sha256=$1
expected_features_sha256=$2
expected_manifest_sha256=56973583dcd1aa8367d8a4e72f1c84f130864536dac128741a3914ae69ed901d
expected_model_sha256=f9b5dc897c5e8499ea5a245b0c512684579a5c6756da9196b628148ccf2c9a76

for digest in "${expected_rollouts_sha256}" "${expected_features_sha256}"; do
  if [[ ! "${digest}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Expected SHA-256 values must contain 64 lowercase hex characters" >&2
    exit 2
  fi
done
if [[ -e "${output}" ]]; then
  echo "Frozen evaluation output already exists" >&2
  exit 2
fi
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
check_sha256 "${manifest}" "${expected_manifest_sha256}"
check_sha256 "${rollouts}" "${expected_rollouts_sha256}"
check_sha256 "${model}" "${expected_model_sha256}"
check_sha256 "${features}" "${expected_features_sha256}"

export PYTHONPATH="${repo_dir}/src"
cd "${repo_dir}"
"${python_bin}" scripts/audit_label_free_semantic_features.py \
  --features "${features}" \
  --rollouts "${rollouts}" \
  > "${feature_dir}/label-free-audit-before-evaluation.json"
"${python_bin}" scripts/evaluate_frozen_action_value.py \
  --model "${model}" \
  --expected-model-sha256 "${expected_model_sha256}" \
  --rollouts "${rollouts}" \
  --expected-rollouts-sha256 "${expected_rollouts_sha256}" \
  --features "${features}" \
  --expected-features-sha256 "${expected_features_sha256}" \
  --require-label-free-features \
  --output "${output}" \
  --bootstrap-resamples 20000 \
  --bootstrap-confidence 0.975 \
  --bootstrap-seed 20260828 \
  --cluster-by source_id

