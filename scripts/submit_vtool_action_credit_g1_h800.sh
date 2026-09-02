#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 0 ]]; then
  echo "usage: $0" >&2
  exit 2
fi

export PATH=/usr/local/slurm/bin:/usr/local/bin:/usr/bin:/bin
repo=/userhome/cs3/yihangc/Documents/beyond-entropy
worker="${repo}/scripts/slurm_vtool_action_credit_g1_h800.sh"
launcher="${repo}/scripts/run_vtool_action_credit_g1.py"
config="${repo}/configs/vtool_action_credit_g1_v1.json"
cd "${repo}"
runtime_audit_relative=$(jq -er '.preflight.full_train_runtime_audit_report' "${config}")
runtime_audit="${repo}/${runtime_audit_relative}"

revision=$(git rev-parse HEAD)
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "repository worktree must be clean before G1 submission" >&2
  exit 2
fi
for path in "${worker}" "${launcher}" "${config}" "${runtime_audit}"; do
  if [[ ! -f "${path}" ]]; then
    echo "required G1 input is absent: ${path}" >&2
    exit 2
  fi
done
if /usr/local/slurm/bin/squeue -h -u yihangc -n be-vtool-g1-signed | grep -q .; then
  echo "a paired-signed G1 job is already queued or running" >&2
  exit 2
fi

quota=$(/usr/local/bin/show-cpu-gpu-quota)
gpu_limit=$(sed -n 's/^GPU Quota in Minutes: \([0-9][0-9]*\), Used: \([0-9][0-9]*\)$/\1/p' <<< "${quota}")
gpu_used=$(sed -n 's/^GPU Quota in Minutes: \([0-9][0-9]*\), Used: \([0-9][0-9]*\)$/\2/p' <<< "${quota}")
if [[ -z "${gpu_limit}" || -z "${gpu_used}" || $((gpu_limit - gpu_used)) -lt 480 ]]; then
  echo "G1 needs a 480 GPU-minute reserve" >&2
  exit 2
fi

worker_sha256=$(sha256sum "${worker}" | awk '{print $1}')
launcher_sha256=$(sha256sum "${launcher}" | awk '{print $1}')
config_sha256=$(sha256sum "${config}" | awk '{print $1}')
audit_sha256=$(sha256sum "${runtime_audit}" | awk '{print $1}')
submit_epoch=$(date +%s)
args=(
  "${revision}"
  "${worker_sha256}"
  "${launcher_sha256}"
  "${config_sha256}"
  "${audit_sha256}"
  "${runtime_audit}"
  "${submit_epoch}"
)
/usr/local/slurm/bin/sbatch --test-only --export=NONE "${worker}" "${args[@]}" >/dev/null
submission=$(/usr/local/slurm/bin/sbatch --parsable --export=NONE "${worker}" "${args[@]}")
job_id=${submission%%;*}
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse paired-signed G1 Slurm job ID: ${submission}" >&2
  exit 2
fi
printf 'vtool_action_credit_g1_job_id=%s code_revision=%s gpu_type=h800 gpu_count=4 max_steps=2\n' \
  "${job_id}" "${revision}"
