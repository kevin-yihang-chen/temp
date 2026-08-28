#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
rollouts="${repo_dir}/artifacts/docvqa-formal-v1/qwen3b-c4-seed0/rollouts.jsonl"
feature_dir="${repo_dir}/artifacts/docvqa-formal-v1/attention-semantic-v1"
model="${repo_dir}/artifacts/docvqa-oof-factorized-action-value-attention-semantic-postfailure-v1/model.json"
expected_model_sha256=1f8b6cf5d026bcd9921434c1c6ef0c753259d36504dedc040b8145c76bd06ff3

if [[ ! -r "${mail_file}" ]]; then
  echo "Missing private Slurm email file: ${mail_file}" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != *@* ]]; then
  echo "Invalid notification email in ${mail_file}" >&2
  exit 2
fi
actual_model_sha256=$(sha256sum "${model}")
actual_model_sha256=${actual_model_sha256%% *}
if [[ "${actual_model_sha256}" != "${expected_model_sha256}" ]]; then
  echo "Frozen attention model SHA-256 mismatch" >&2
  exit 2
fi

dependency_args=()
if [[ "$#" -eq 1 ]]; then
  dependency_args=(--dependency="afterok:$1")
elif [[ "$#" -ne 0 ]]; then
  echo "Usage: $0 [rollout-job-id]" >&2
  exit 2
fi

exec sbatch \
  "${dependency_args[@]}" \
  --job-name=be-docvqa-attn-formal-feat \
  --mail-user="${notify_email}" \
  --mail-type=ALL \
  --export=ALL,BE_FORMAL_ROLLOUTS="${rollouts}",BE_FORMAL_FEATURE_DIR="${feature_dir}",BE_FROZEN_MODEL="${model}",BE_EXPECTED_MODEL_SHA256="${expected_model_sha256}" \
  "${repo_dir}/scripts/slurm_docvqa_attention_formal_features.sh"
