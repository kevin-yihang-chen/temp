#!/usr/bin/env bash
set -euo pipefail

export PATH=/usr/local/slurm/bin:/usr/local/bin:/usr/bin:/bin
repo=/userhome/cs3/yihangc/Documents/beyond-entropy
worker="${repo}/scripts/slurm_infographicvqa_decar_entropy_where_hybrid.sh"
runner="${repo}/scripts/evaluate_infographicvqa_decar_entropy_where_hybrid.py"
module="${repo}/src/beyond_entropy/infographicvqa_decar_evaluation.py"
freeze="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-decar-entropy-where-hybrid-freeze-v1.md"
resource_amendment="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-decar-entropy-where-hybrid-resource-amendment-v1.md"
root="${repo}/artifacts/infographicvqa-train-v1/decar-v1/full-qwen7b-v1"
output_dir="${root}/entropy-where-hybrid-v1"
cd "${repo}"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before DECAR hybrid submission" >&2
  exit 2
fi
for path in "${worker}" "${runner}" "${module}" "${freeze}" "${resource_amendment}" \
  "${root}/merged-rollouts/rollouts.jsonl" \
  "${root}/nested-oof-v1/predictions.jsonl" \
  "${root}/evaluation-v1/evaluation.json" \
  "${root}/evaluation-v1/bootstrap-indices.npy"; do
  if [[ ! -f "${path}" ]]; then
    echo "DECAR hybrid input is incomplete: ${path}" >&2
    exit 2
  fi
done
if [[ -e "${output_dir}" ]]; then
  echo "DECAR hybrid output already exists" >&2
  exit 2
fi

quota=$(/usr/local/bin/show-cpu-gpu-quota)
gpu_limit=$(sed -n 's/^GPU Quota in Minutes: \([0-9][0-9]*\), Used: \([0-9][0-9]*\)$/\1/p' <<< "${quota}")
gpu_used=$(sed -n 's/^GPU Quota in Minutes: \([0-9][0-9]*\), Used: \([0-9][0-9]*\)$/\2/p' <<< "${quota}")
cpu_limit=$(sed -n 's/^CPU Quota in Minutes: \([0-9][0-9]*\), Used: \([0-9][0-9]*\)$/\1/p' <<< "${quota}")
cpu_used=$(sed -n 's/^CPU Quota in Minutes: \([0-9][0-9]*\), Used: \([0-9][0-9]*\)$/\2/p' <<< "${quota}")
if [[ -z "${gpu_limit}" || -z "${gpu_used}" || -z "${cpu_limit}" \
  || -z "${cpu_used}" || $((gpu_limit - gpu_used)) -lt 45 \
  || $((cpu_limit - cpu_used)) -lt 180 ]]; then
  echo "DECAR hybrid needs 45 GPU-minutes and 180 CPU-minutes in reserve" >&2
  exit 2
fi

revision=$(git rev-parse HEAD)
worker_sha256=$(sha256sum "${worker}" | awk '{print $1}')
runner_sha256=$(sha256sum "${runner}" | awk '{print $1}')
module_sha256=$(sha256sum "${module}" | awk '{print $1}')
freeze_sha256=$(sha256sum "${freeze}" | awk '{print $1}')
resource_amendment_sha256=$(sha256sum "${resource_amendment}" | awk '{print $1}')
submit_epoch=$(date +%s)
args=("${revision}" "${worker_sha256}" "${runner_sha256}" "${module_sha256}" \
  "${freeze_sha256}" "${resource_amendment_sha256}" "${submit_epoch}")
/usr/local/slurm/bin/sbatch --test-only --export=NONE "${worker}" "${args[@]}" >/dev/null
submission=$(/usr/local/slurm/bin/sbatch --parsable --export=NONE "${worker}" "${args[@]}")
job_id=${submission%%;*}
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse DECAR hybrid Slurm job ID: ${submission}" >&2
  exit 2
fi
printf 'infographicvqa_decar_hybrid_job_id=%s code_revision=%s gpu_type=rtx_4090 gpu_count=1 cpus=4 evaluator_device=cpu\n' \
  "${job_id}" "${revision}"
