#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"

if [[ ! -r "${mail_file}" ]]; then
  echo "Missing private Slurm email file: ${mail_file}" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != *@* ]]; then
  echo "Invalid notification email in ${mail_file}" >&2
  exit 2
fi

exec sbatch \
  --job-name=be-sem-vstar-c9 \
  --mail-user="${notify_email}" \
  --mail-type=ALL \
  --export=ALL,BE_SEMANTIC_ROLLOUTS="${repo_dir}/artifacts/gate1-vstar-64/qwen3b-c9-seed0/rollouts.jsonl",BE_SEMANTIC_ROLLOUTS_SHA256=def744acf04ffba6736fe4b34f29c378ad2a257cf2ea733bba00d0d7041ecc7f,BE_SEMANTIC_RUN_DIR="${repo_dir}/artifacts/gate2-vstar-64-c9/qwen3b-roi-seed17" \
  "${repo_dir}/scripts/slurm_semantic_experiment.sh"
