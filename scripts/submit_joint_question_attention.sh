#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
base="${repo_dir}/artifacts/cross-benchmark-dev-v1/docvqa/qwen3b-c4-seed0"
rollouts="${base}/rollouts.jsonl"
source_features="${base}/features.pt"
output_features="${base}/features-joint-question-attention.pt"
expected_rollouts_sha256=4d3d3a33f644d1f5122aabecd47a8168d2dce2db5014692b508ba76ae4ddbe52
expected_features_sha256=6114b2f9365a3028263ac159bd5b8677a0117a1951d3e8443d17cf1b5959dd0e

if [[ ! -r "${mail_file}" ]]; then
  echo "Missing private Slurm email file: ${mail_file}" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != *@* ]]; then
  echo "Invalid notification email in ${mail_file}" >&2
  exit 2
fi
actual_rollouts_sha256=$(sha256sum "${rollouts}")
actual_rollouts_sha256=${actual_rollouts_sha256%% *}
actual_features_sha256=$(sha256sum "${source_features}")
actual_features_sha256=${actual_features_sha256%% *}
if [[ "${actual_rollouts_sha256}" != "${expected_rollouts_sha256}" ]]; then
  echo "DocVQA rollout SHA-256 mismatch" >&2
  exit 2
fi
if [[ "${actual_features_sha256}" != "${expected_features_sha256}" ]]; then
  echo "DocVQA source feature SHA-256 mismatch" >&2
  exit 2
fi

exec sbatch \
  --job-name=be-docvqa-joint-attn \
  --mail-user="${notify_email}" \
  --mail-type=ALL \
  --export=ALL,BE_SEMANTIC_SOURCE_FEATURES="${source_features}",BE_SEMANTIC_ROLLOUTS="${rollouts}",BE_SEMANTIC_OUTPUT_FEATURES="${output_features}" \
  "${repo_dir}/scripts/slurm_joint_question_attention.sh"
