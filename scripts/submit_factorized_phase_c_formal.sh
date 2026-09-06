#!/usr/bin/env bash
set -euo pipefail

repo=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo}/.slurm-notify-email"
[[ "$#" -eq 1 ]] || { echo "usage: $0 FORMAL_PLAN" >&2; exit 2; }
plan=$(realpath "$1")
IFS= read -r notify_email < "${mail_file}"
[[ "${notify_email}" == "yihangc@connect.hku.hk" ]] || {
  echo "invalid notification email" >&2; exit 2;
}
[[ -z "$(git -C "${repo}" status --porcelain --untracked-files=no)" ]] || {
  echo "tracked worktree must be clean before formal submission" >&2; exit 2;
}

digest() { sha256sum "$1" | cut -d ' ' -f 1; }
revision=$(git -C "${repo}" rev-parse HEAD)
plan_sha256=$(digest "${plan}")
worker="${repo}/scripts/slurm_factorized_phase_c_formal.sh"
plan_revision=$(jq -er '.code_revision' "${plan}")
[[ "${plan_revision}" == "${revision}" ]] || {
  echo "formal plan revision differs from current code" >&2; exit 2;
}
ledger=$(jq -er '.access_ledger' "${plan}")
transaction_root=$(jq -er '.transaction_root' "${plan}")
[[ ! -e "${ledger}" && ! -e "${transaction_root}" ]] || {
  echo "formal transaction was already opened" >&2; exit 2;
}

exports="ALL,BE_PHASE_C_FORMAL_PLAN=${plan},BE_PHASE_C_FORMAL_PLAN_SHA256=${plan_sha256},BE_PHASE_C_FORMAL_CODE_REVISION=${revision},BE_PHASE_C_FORMAL_WORKER_SHA256=$(digest "${worker}")"
/usr/local/slurm/bin/sbatch \
  --mail-user="${notify_email}" --mail-type=ALL --no-requeue \
  --export="${exports}" "${worker}"
