#!/usr/bin/env bash
set -euo pipefail

export PATH=/usr/local/slurm/bin:/usr/local/bin:/usr/bin:/bin
repo=/userhome/cs3/yihangc/Documents/beyond-entropy
worker="${repo}/scripts/slurm_infographicvqa_decar_oof_h800.sh"
freeze="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-decar-oof-evaluation-freeze-v1.md"
root="${repo}/artifacts/infographicvqa-train-v1/decar-v1/full-qwen7b-v1"
rollouts="${root}/merged-rollouts/rollouts.jsonl"
answer_nll="${root}/merged-nll/answer-nll.jsonl"
features="${root}/merged-features/features-label-free.pt"
input_audit="${root}/merged-features/decar-input-audit.json"
fit_dir="${root}/nested-oof-v1"
evaluation_dir="${root}/evaluation-v1"
cd "${repo}"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before DECAR OOF submission" >&2
  exit 2
fi
if [[ ! -f "${worker}" || ! -f "${freeze}" ]]; then
  echo "DECAR OOF evaluation freeze is incomplete" >&2
  exit 2
fi
if [[ -e "${fit_dir}" || -e "${evaluation_dir}" ]]; then
  echo "DECAR OOF fit/evaluation output already exists" >&2
  exit 2
fi
for path in "${rollouts}" "${answer_nll}" "${features}" "${input_audit}"; do
  if [[ ! -f "${path}" ]]; then
    echo "DECAR OOF generation input is incomplete: ${path}" >&2
    exit 2
  fi
done
mapfile -t generation_executions < <(find "${root}/execution" -maxdepth 1 -type f -name 'job-*.json' -print | sort)
if [[ "${#generation_executions[@]}" -ne 1 ]]; then
  echo "DECAR OOF requires exactly one generation execution record" >&2
  exit 2
fi
generation_execution=${generation_executions[0]}

quota=$(/usr/local/bin/show-cpu-gpu-quota)
gpu_limit=$(sed -n 's/^GPU Quota in Minutes: \([0-9][0-9]*\), Used: \([0-9][0-9]*\)$/\1/p' <<< "${quota}")
gpu_used=$(sed -n 's/^GPU Quota in Minutes: \([0-9][0-9]*\), Used: \([0-9][0-9]*\)$/\2/p' <<< "${quota}")
if [[ -z "${gpu_limit}" || -z "${gpu_used}" || $((gpu_limit - gpu_used)) -lt 240 ]]; then
  echo "DECAR OOF needs a 240 GPU-minute reserve" >&2
  exit 2
fi

revision=$(git rev-parse HEAD)
worker_sha256=$(sha256sum "${worker}" | awk '{print $1}')
freeze_sha256=$(sha256sum "${freeze}" | awk '{print $1}')
rollouts_sha256=$(sha256sum "${rollouts}" | awk '{print $1}')
nll_sha256=$(sha256sum "${answer_nll}" | awk '{print $1}')
features_sha256=$(sha256sum "${features}" | awk '{print $1}')
input_audit_sha256=$(sha256sum "${input_audit}" | awk '{print $1}')
generation_execution_sha256=$(sha256sum "${generation_execution}" | awk '{print $1}')
submit_epoch=$(date +%s)
args=("${revision}" "${worker_sha256}" "${freeze_sha256}" "${rollouts_sha256}" \
  "${nll_sha256}" "${features_sha256}" "${input_audit_sha256}" \
  "${generation_execution_sha256}" "${submit_epoch}")
/usr/local/slurm/bin/sbatch --test-only --export=NONE "${worker}" "${args[@]}" >/dev/null
submission=$(/usr/local/slurm/bin/sbatch --parsable --export=NONE "${worker}" "${args[@]}")
job_id=${submission%%;*}
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse DECAR OOF Slurm job ID: ${submission}" >&2
  exit 2
fi
printf 'infographicvqa_decar_oof_job_id=%s code_revision=%s gpu_type=h800 gpu_count=1\n' \
  "${job_id}" "${revision}"
