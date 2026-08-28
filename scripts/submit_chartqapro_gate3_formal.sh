#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
manifest="${repo_dir}/data/chartqapro-gate3-e27c287-v2/formal/manifest.jsonl"
pilot_report="${repo_dir}/artifacts/gate3-chartqapro-pilot-309/analysis-v2-final/report.json"
replay_audit="${repo_dir}/artifacts/gate3-chartqapro-pilot-309/replay-audit-v1-v2.json"
expected_manifest_sha256=5a3ddca2e6476196aac8ad4fa7bc00033f2ac9c39d2011fe21fa070e965b97d4
expected_pilot_report_sha256=93e6f04989fa00c247406baaad2815a486b8d145bf8fa932b83648cf5995fe99
expected_replay_audit_sha256=173ff249f1fb8c25b73abdc28f32d705bd3d25737dea6d3bd58b8ce042106480

if [[ ! -r "${mail_file}" ]]; then
  echo "Missing private Slurm email file: ${mail_file}" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != *@* ]]; then
  echo "Invalid notification email in ${mail_file}" >&2
  exit 2
fi
if [[ "$(wc -l < "${manifest}")" -ne 1625 ]]; then
  echo "Expected 1,625 frozen ChartQAPro formal states" >&2
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
check_sha256 "${pilot_report}" "${expected_pilot_report_sha256}"
check_sha256 "${replay_audit}" "${expected_replay_audit_sha256}"
if ! jq -e '.compatibility_acceptance.passed == true' "${pilot_report}" >/dev/null; then
  echo "ChartQAPro v2 compatibility pilot did not pass" >&2
  exit 2
fi
if ! jq -e '.passed == true' "${replay_audit}" >/dev/null; then
  echo "ChartQAPro v1/v2 replay audit did not pass" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "Tracked worktree changes must be committed before formal submission" >&2
  exit 2
fi
code_revision=$(git -C "${repo_dir}" rev-parse HEAD)

exec sbatch \
  --mail-user="${notify_email}" \
  --mail-type=ALL \
  --export=ALL,BE_CODE_REVISION="${code_revision}" \
  "${repo_dir}/scripts/slurm_chartqapro_gate3_formal.sh"
