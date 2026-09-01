#!/usr/bin/env bash
set -euo pipefail

export PATH=/usr/local/slurm/bin:/usr/local/bin:/usr/bin:/bin
repo=/userhome/cs3/yihangc/Documents/beyond-entropy
runner="${repo}/scripts/benchmark_infographicvqa_decar_fit_runtime.py"
worker="${repo}/scripts/slurm_infographicvqa_decar_fit_benchmark_h800.sh"
output="${repo}/artifacts/infographicvqa-train-v1/decar-v1/full-qwen7b-v1/ops/oof-fit-runtime-benchmark-v1/report.json"
cd "${repo}"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before DECAR fit benchmark" >&2
  exit 2
fi
if [[ ! -f "${runner}" || ! -f "${worker}" || -e "${output}" ]]; then
  echo "DECAR fit benchmark inputs are missing or output already exists" >&2
  exit 2
fi
quota=$(/usr/local/bin/show-cpu-gpu-quota)
gpu_limit=$(sed -n 's/^GPU Quota in Minutes: \([0-9][0-9]*\), Used: \([0-9][0-9]*\)$/\1/p' <<< "${quota}")
gpu_used=$(sed -n 's/^GPU Quota in Minutes: \([0-9][0-9]*\), Used: \([0-9][0-9]*\)$/\2/p' <<< "${quota}")
if [[ -z "${gpu_limit}" || -z "${gpu_used}" || $((gpu_limit - gpu_used)) -lt 45 ]]; then
  echo "DECAR fit benchmark needs a 45 GPU-minute reserve" >&2
  exit 2
fi

revision=$(git rev-parse HEAD)
runner_sha256=$(sha256sum "${runner}" | awk '{print $1}')
submit_epoch=$(date +%s)
args=("${revision}" "${runner_sha256}" "${submit_epoch}")
/usr/local/slurm/bin/sbatch --test-only --export=NONE "${worker}" "${args[@]}" >/dev/null
submission=$(/usr/local/slurm/bin/sbatch --parsable --export=NONE "${worker}" "${args[@]}")
job_id=${submission%%;*}
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse DECAR fit benchmark Slurm job ID: ${submission}" >&2
  exit 2
fi
printf 'infographicvqa_decar_fit_benchmark_job_id=%s code_revision=%s gpu_type=h800 gpu_count=1\n' \
  "${job_id}" "${revision}"
