#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 0 ]]; then
  echo "usage: $0" >&2
  exit 2
fi

export PATH=/usr/local/slurm/bin:/usr/local/bin:/usr/bin:/bin
repo=/userhome/cs3/yihangc/Documents/beyond-entropy
worker="${repo}/scripts/slurm_vtool_vllm_model_load_h800.sh"
smoke="${repo}/scripts/smoke_vtool_vllm_model_load.py"
config="${repo}/configs/vtool_action_credit_g1_v1.json"
dataset="${repo}/artifacts/docvqa-train-factorized-v2/dataset-converter/refocus-g1-one-row-v1/g1_smoke.parquet"
cd "${repo}"

revision=$(git rev-parse HEAD)
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before model-load submission" >&2
  exit 2
fi
for path in "${worker}" "${smoke}" "${config}" "${dataset}"; do
  if [[ ! -f "${path}" ]]; then
    echo "required model-load input is absent: ${path}" >&2
    exit 2
  fi
done

quota=$(/usr/local/bin/show-cpu-gpu-quota)
gpu_limit=$(sed -n 's/^GPU Quota in Minutes: \([0-9][0-9]*\), Used: \([0-9][0-9]*\)$/\1/p' <<< "${quota}")
gpu_used=$(sed -n 's/^GPU Quota in Minutes: \([0-9][0-9]*\), Used: \([0-9][0-9]*\)$/\2/p' <<< "${quota}")
if [[ -z "${gpu_limit}" || -z "${gpu_used}" || $((gpu_limit - gpu_used)) -lt 120 ]]; then
  echo "model-load smoke needs a 120 GPU-minute reserve" >&2
  exit 2
fi

worker_sha256=$(sha256sum "${worker}" | awk '{print $1}')
smoke_sha256=$(sha256sum "${smoke}" | awk '{print $1}')
dataset_sha256=$(sha256sum "${dataset}" | awk '{print $1}')
config_sha256=$(sha256sum "${config}" | awk '{print $1}')
submit_epoch=$(date +%s)
args=(
  "${revision}"
  "${worker_sha256}"
  "${smoke_sha256}"
  "${dataset_sha256}"
  "${config_sha256}"
  "${dataset}"
  "${submit_epoch}"
)
/usr/local/slurm/bin/sbatch --test-only --export=NONE "${worker}" "${args[@]}" >/dev/null
submission=$(/usr/local/slurm/bin/sbatch --parsable --export=NONE "${worker}" "${args[@]}")
job_id=${submission%%;*}
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse model-load Slurm job ID: ${submission}" >&2
  exit 2
fi
printf 'vtool_model_load_job_id=%s code_revision=%s gpu_type=h800 gpu_count=1\n' \
  "${job_id}" "${revision}"
