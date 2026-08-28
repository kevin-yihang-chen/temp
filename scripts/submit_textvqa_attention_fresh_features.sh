#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 EXPECTED_ROLLOUTS_SHA256" >&2
  exit 2
fi

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
rollouts="${repo_dir}/artifacts/textvqa-attention-fresh-formal-v1/qwen3b-c4-seed0/rollouts.jsonl"
feature_dir="${repo_dir}/artifacts/textvqa-attention-fresh-formal-v1/attention-semantic-v1"
model="${repo_dir}/artifacts/textvqa-oof-factorized-action-value-attention-semantic-postfailure-v1/model.json"
expected_model_sha256=f9b5dc897c5e8499ea5a245b0c512684579a5c6756da9196b628148ccf2c9a76
expected_rollouts_sha256=$1

if [[ ! "${expected_rollouts_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "Expected rollout SHA-256 must contain 64 lowercase hex characters" >&2
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
if [[ "$(wc -l < "${rollouts}")" -ne 15830 ]]; then
  echo "Fresh-split TextVQA rollout row-count mismatch" >&2
  exit 2
fi
actual_rollouts_sha256=$(sha256sum "${rollouts}")
actual_rollouts_sha256=${actual_rollouts_sha256%% *}
if [[ "${actual_rollouts_sha256}" != "${expected_rollouts_sha256}" ]]; then
  echo "Fresh-split TextVQA rollout SHA-256 mismatch" >&2
  exit 2
fi
actual_model_sha256=$(sha256sum "${model}")
actual_model_sha256=${actual_model_sha256%% *}
if [[ "${actual_model_sha256}" != "${expected_model_sha256}" ]]; then
  echo "Frozen attention model SHA-256 mismatch" >&2
  exit 2
fi

exec sbatch \
  --job-name=be-textvqa-attn-fresh-features \
  --mail-user="${notify_email}" \
  --mail-type=ALL \
  --export=ALL,BE_FORMAL_ROLLOUTS="${rollouts}",BE_EXPECTED_ROLLOUTS_SHA256="${expected_rollouts_sha256}",BE_FORMAL_FEATURE_DIR="${feature_dir}",BE_FROZEN_MODEL="${model}",BE_EXPECTED_MODEL_SHA256="${expected_model_sha256}" \
  "${repo_dir}/scripts/slurm_textvqa_attention_fresh_features.sh"

