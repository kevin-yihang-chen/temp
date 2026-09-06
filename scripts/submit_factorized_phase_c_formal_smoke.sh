#!/usr/bin/env bash
set -euo pipefail
repo=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo}/.slurm-notify-email"
[[ "$#" -eq 2 ]] || { echo "usage: $0 TRAIN_JOB SEED" >&2; exit 2; }
train_job=$1
seed=$2
[[ "${train_job}" =~ ^[0-9]+$ ]] || { echo "invalid train job" >&2; exit 2; }
case "${seed}" in 17|29|47) ;; *) echo "invalid seed" >&2; exit 2 ;; esac
IFS= read -r notify_email < "${mail_file}"
[[ "${notify_email}" == "yihangc@connect.hku.hk" ]] || exit 2
[[ -z "$(git -C "${repo}" status --porcelain --untracked-files=no)" ]] || exit 2
digest() { sha256sum "$1" | cut -d ' ' -f 1; }
revision=$(git -C "${repo}" rev-parse HEAD)
worker="${repo}/scripts/slurm_factorized_phase_c_formal_smoke.sh"
script="${repo}/scripts/smoke_factorized_phase_c_formal_runtime.py"
exports="ALL,BE_PHASE_C_SMOKE_CODE_REVISION=${revision},BE_PHASE_C_SMOKE_WORKER_SHA256=$(digest "${worker}"),BE_PHASE_C_SMOKE_SCRIPT_SHA256=$(digest "${script}"),BE_PHASE_C_SMOKE_TRAIN_JOB=${train_job},BE_PHASE_C_SMOKE_SEED=${seed}"
/usr/local/slurm/bin/sbatch \
  --mail-user="${notify_email}" --mail-type=ALL --no-requeue \
  --export="${exports}" "${worker}"
