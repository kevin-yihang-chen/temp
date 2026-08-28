#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 {docvqa|textvqa}" >&2
  exit 2
fi

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
task=$1
base="${repo_dir}/artifacts/cross-benchmark-dev-v1/${task}/qwen3b-c4-seed0"
rollouts="${base}/rollouts.jsonl"
source_features="${base}/features-multimodal-context.pt"
output_features="${base}/features-question-region-attention.pt"

case "${task}" in
  docvqa)
    expected_rows=4120
    expected_rollouts_sha256=4d3d3a33f644d1f5122aabecd47a8168d2dce2db5014692b508ba76ae4ddbe52
    expected_features_sha256=27b9095651830451e1841058957dc88325790e66b82dda7d81e719b2647774a5
    ;;
  textvqa)
    expected_rows=1590
    expected_rollouts_sha256=a94c72b1977e86436c6187248f64826a34b791151c52a7c7b73ca89f92b97ddb
    expected_features_sha256=b292a33a425f86d0319f4d9561dd07d2458779e51a169519d9d80a587a7627e8
    ;;
  *)
    echo "Unsupported task: ${task}" >&2
    exit 2
    ;;
esac

if [[ ! -r "${mail_file}" ]]; then
  echo "Missing private Slurm email file: ${mail_file}" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != *@* ]]; then
  echo "Invalid notification email in ${mail_file}" >&2
  exit 2
fi
if [[ "$(wc -l < "${rollouts}")" -ne "${expected_rows}" ]]; then
  echo "${task} rollout row-count mismatch" >&2
  exit 2
fi
actual_rollouts_sha256=$(sha256sum "${rollouts}")
actual_rollouts_sha256=${actual_rollouts_sha256%% *}
actual_features_sha256=$(sha256sum "${source_features}")
actual_features_sha256=${actual_features_sha256%% *}
if [[ "${actual_rollouts_sha256}" != "${expected_rollouts_sha256}" ]]; then
  echo "${task} rollout SHA-256 mismatch" >&2
  exit 2
fi
if [[ "${actual_features_sha256}" != "${expected_features_sha256}" ]]; then
  echo "${task} semantic feature SHA-256 mismatch" >&2
  exit 2
fi

exec sbatch \
  --job-name="be-${task}-region-attn" \
  --mail-user="${notify_email}" \
  --mail-type=ALL \
  --export=ALL,BE_SEMANTIC_SOURCE_FEATURES="${source_features}",BE_SEMANTIC_ROLLOUTS="${rollouts}",BE_SEMANTIC_OUTPUT_FEATURES="${output_features}" \
  "${repo_dir}/scripts/slurm_question_region_attention.sh"
