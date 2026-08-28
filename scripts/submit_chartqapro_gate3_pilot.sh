#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
manifest="${repo_dir}/data/chartqapro-gate3-e27c287-v1/pilot/manifest.jsonl"
expected_manifest_sha256=e02f62ae794125c5e4565493e54b72855b1db96542257c57a15049de65f6a722

if [[ ! -r "${mail_file}" ]]; then
  echo "Missing private Slurm email file: ${mail_file}" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != *@* ]]; then
  echo "Invalid notification email in ${mail_file}" >&2
  exit 2
fi
if [[ "$(wc -l < "${manifest}")" -ne 309 ]]; then
  echo "Expected 309 frozen ChartQAPro pilot states" >&2
  exit 2
fi
actual_manifest_sha256=$(sha256sum "${manifest}")
actual_manifest_sha256=${actual_manifest_sha256%% *}
if [[ "${actual_manifest_sha256}" != "${expected_manifest_sha256}" ]]; then
  echo "ChartQAPro pilot manifest SHA-256 mismatch" >&2
  exit 2
fi
code_revision=$(git -C "${repo_dir}" rev-parse HEAD)

exec sbatch \
  --mail-user="${notify_email}" \
  --mail-type=ALL \
  --export=ALL,BE_CODE_REVISION="${code_revision}" \
  "${repo_dir}/scripts/slurm_chartqapro_gate3_pilot.sh"
