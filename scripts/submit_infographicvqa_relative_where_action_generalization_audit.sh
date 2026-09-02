#!/usr/bin/env bash
set -euo pipefail

export PATH=/usr/local/slurm/bin:/usr/local/bin:/usr/bin:/bin
repo=/userhome/cs3/yihangc/Documents/beyond-entropy
root="${repo}/artifacts/infographicvqa-train-v1/decar-v1/full-qwen7b-v1"
worker="${repo}/scripts/slurm_infographicvqa_relative_where_action_generalization_audit.sh"
runner="${repo}/scripts/audit_infographicvqa_relative_where_action_generalization.py"
module="${repo}/src/beyond_entropy/infographicvqa_relative_where_diagnostics.py"
protocol="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-relative-where-action-generalization-audit-protocol-v1.md"
parent_result="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-relative-where-oof-result-job-203237-v1.md"
output_dir="${root}/relative-where-action-generalization-audit-v1"
cd "${repo}"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before relative-where action audit submission" >&2
  exit 2
fi
for path in "${worker}" "${runner}" "${module}" "${protocol}" \
  "${parent_result}" "${root}/relative-where-oof-v1/predictions.jsonl" \
  "${root}/merged-nll/answer-nll.jsonl" \
  "${root}/merged-rollouts/rollouts.jsonl"; do
  if [[ ! -f "${path}" ]]; then
    echo "relative-where action audit input is incomplete: ${path}" >&2
    exit 2
  fi
done
if [[ -e "${output_dir}" ]]; then
  echo "relative-where action audit output already exists" >&2
  exit 2
fi

quota=$(/usr/local/bin/show-cpu-gpu-quota)
gpu_limit=$(sed -n 's/^GPU Quota in Minutes: \([0-9][0-9]*\), Used: \([0-9][0-9]*\)$/\1/p' <<< "${quota}")
gpu_used=$(sed -n 's/^GPU Quota in Minutes: \([0-9][0-9]*\), Used: \([0-9][0-9]*\)$/\2/p' <<< "${quota}")
cpu_limit=$(sed -n 's/^CPU Quota in Minutes: \([0-9][0-9]*\), Used: \([0-9][0-9]*\)$/\1/p' <<< "${quota}")
cpu_used=$(sed -n 's/^CPU Quota in Minutes: \([0-9][0-9]*\), Used: \([0-9][0-9]*\)$/\2/p' <<< "${quota}")
if [[ -z "${gpu_limit}" || -z "${gpu_used}" || -z "${cpu_limit}" \
  || -z "${cpu_used}" || $((gpu_limit - gpu_used)) -lt 30 \
  || $((cpu_limit - cpu_used)) -lt 120 ]]; then
  echo "relative-where action audit needs 30 GPU-minutes and 120 CPU-minutes in reserve" >&2
  exit 2
fi

revision=$(git rev-parse HEAD)
worker_sha256=$(sha256sum "${worker}" | awk '{print $1}')
runner_sha256=$(sha256sum "${runner}" | awk '{print $1}')
module_sha256=$(sha256sum "${module}" | awk '{print $1}')
protocol_sha256=$(sha256sum "${protocol}" | awk '{print $1}')
result_sha256=$(sha256sum "${parent_result}" | awk '{print $1}')
submit_epoch=$(date +%s)
args=("${revision}" "${worker_sha256}" "${runner_sha256}" "${module_sha256}" \
  "${protocol_sha256}" "${result_sha256}" "${submit_epoch}")
/usr/local/slurm/bin/sbatch --test-only --export=NONE "${worker}" "${args[@]}" >/dev/null
submission=$(/usr/local/slurm/bin/sbatch --parsable --export=NONE "${worker}" "${args[@]}")
job_id=${submission%%;*}
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse relative-where action audit Slurm job ID: ${submission}" >&2
  exit 2
fi
printf 'infographicvqa_relative_where_action_audit_job_id=%s code_revision=%s gpu_type=rtx_4090 gpu_count=1 cpus=4 diagnostic_device=cpu\n' \
  "${job_id}" "${revision}"
