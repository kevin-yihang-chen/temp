#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
rollouts="${repo_dir}/artifacts/gate1-chartqa-2500/qwen3b-c4-concise-seed0/rollouts.jsonl"
features="${repo_dir}/artifacts/gate2-chartqa-2500/qwen3b-roi-concise-seed17/features.pt"
output_dir="${repo_dir}/artifacts/gate2-chartqa-2500/compact-factorized-nested-oof-v8"

if [[ ! -r "${mail_file}" ]]; then
  echo "Missing private Slurm email file: ${mail_file}" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != *@* ]]; then
  echo "Invalid notification email in ${mail_file}" >&2
  exit 2
fi
if [[ ! -r "${features}" ]]; then
  echo "Missing complete semantic feature file: ${features}" >&2
  exit 2
fi
actual_rows=$(wc -l < "${rollouts}")
if [[ "${actual_rows}" -ne 12500 ]]; then
  echo "Expected 12500 complete rollout rows, found ${actual_rows}" >&2
  exit 2
fi

exec sbatch \
  --job-name=be-factorized2500 \
  --mail-user="${notify_email}" \
  --mail-type=ALL \
  --export=ALL,BE_RESCUE_FEATURES="${features}",BE_RESCUE_ROLLOUTS="${rollouts}",BE_RESCUE_OUTPUT_DIR="${output_dir}" \
  "${repo_dir}/scripts/slurm_factorized_rescue_gate.sh"
