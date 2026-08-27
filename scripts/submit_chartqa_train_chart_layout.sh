#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
manifest="${repo_dir}/data/chartqa-train-replication-4500/manifest.jsonl"
expected_manifest_sha256=72db6feaa4bc042e98741a48dd55421c5246c1b48c84b1fd75740d1d072ca621

if [[ ! -r "${mail_file}" ]]; then
  echo "Missing private Slurm email file: ${mail_file}" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != *@* ]]; then
  echo "Invalid notification email in ${mail_file}" >&2
  exit 2
fi
if [[ "$(wc -l < "${manifest}")" -ne 4500 ]]; then
  echo "Expected 4500 frozen chart-layout target states" >&2
  exit 2
fi
actual_manifest_sha256=$(sha256sum "${manifest}")
actual_manifest_sha256=${actual_manifest_sha256%% *}
if [[ "${actual_manifest_sha256}" != "${expected_manifest_sha256}" ]]; then
  echo "Chart-layout target manifest SHA-256 mismatch" >&2
  exit 2
fi
code_revision=$(git -C "${repo_dir}" rev-parse HEAD)

exec sbatch \
  --mail-user="${notify_email}" \
  --mail-type=ALL \
  --export=ALL,BE_CODE_REVISION="${code_revision}" \
  "${repo_dir}/scripts/slurm_chartqa_train_chart_layout.sh"
