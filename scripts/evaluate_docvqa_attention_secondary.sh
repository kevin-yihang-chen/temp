#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 EXPECTED_FEATURE_SHA256" >&2
  exit 2
fi

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
rollouts="${repo_dir}/artifacts/docvqa-formal-v1/qwen3b-c4-seed0/rollouts.jsonl"
model="${repo_dir}/artifacts/docvqa-oof-factorized-action-value-attention-semantic-postfailure-v1/model.json"
feature_dir="${repo_dir}/artifacts/docvqa-formal-v1/attention-semantic-v1"
features="${feature_dir}/features-question-region-attention-label-free.pt"
audit="${feature_dir}/label-free-audit.json"
output="${feature_dir}/evaluation.json"
primary="${repo_dir}/artifacts/docvqa-formal-v1/frozen-context-v1/evaluation.json"
expected_features_sha256=$1
expected_model_sha256=1f8b6cf5d026bcd9921434c1c6ef0c753259d36504dedc040b8145c76bd06ff3
expected_rollouts_sha256=a7f44c267b11c12f6cbf8f1e714350174c4dfd7e4ab3866fde0dbd84fe0b5aa3
expected_primary_sha256=9f7428b661ea213ac5fa6bd9e58b5a22ac3dd505848064c47a94fb4a4310efc9

actual_primary_sha256=$(sha256sum "${primary}")
actual_primary_sha256=${actual_primary_sha256%% *}
if [[ "${actual_primary_sha256}" != "${expected_primary_sha256}" ]]; then
  echo "Primary evaluation is missing or changed" >&2
  exit 2
fi
if [[ ! -r "${audit}" ]]; then
  echo "Label-free audit is missing" >&2
  exit 2
fi
actual_features_sha256=$(sha256sum "${features}")
actual_features_sha256=${actual_features_sha256%% *}
if [[ "${actual_features_sha256}" != "${expected_features_sha256}" ]]; then
  echo "Formal attention feature SHA-256 mismatch" >&2
  exit 2
fi

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
  --bootstrap-resamples 10000 \
  --bootstrap-confidence 0.975 \
  --bootstrap-seed 20260828 \
  --cluster-by source_id
