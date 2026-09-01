#!/usr/bin/env bash

set -euo pipefail

repo=/userhome/cs3/yihangc/Documents/beyond-entropy
worker="${repo}/scripts/slurm_infographicvqa_decar_full_materialize.sh"
freeze="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-decar-full-materialization-freeze-v1.md"
output_dir="${repo}/artifacts/infographicvqa-train-v1/decar-v1/full-manifest-v1"
notify_email=yihangc@connect.hku.hk

cd "${repo}"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before DECAR full materialization" >&2
  exit 2
fi
if [[ -e "${output_dir}" ]]; then
  echo "DECAR full materialization output already exists" >&2
  exit 2
fi
for path in "${worker}" "${freeze}"; do
  if [[ ! -f "${path}" ]]; then
    echo "missing DECAR full materialization input: ${path}" >&2
    exit 2
  fi
done
active_jobs=$(/usr/local/slurm/bin/squeue -h -u "${USER}" -t PENDING,RUNNING,CONFIGURING,COMPLETING | wc -l)
if [[ "${active_jobs}" -ne 0 ]]; then
  echo "DECAR full materialization requires the sole job slot to be free" >&2
  exit 2
fi
quota=$(/usr/local/bin/show-cpu-gpu-quota)
gpu_limit=$(sed -n 's/^GPU Quota in Minutes: \([0-9][0-9]*\), Used: \([0-9][0-9]*\)$/\1/p' <<< "${quota}")
gpu_used=$(sed -n 's/^GPU Quota in Minutes: \([0-9][0-9]*\), Used: \([0-9][0-9]*\)$/\2/p' <<< "${quota}")
if [[ -z "${gpu_limit}" || -z "${gpu_used}" || $((gpu_limit - gpu_used)) -lt 30 ]]; then
  echo "DECAR full materialization needs a 30 GPU-minute reserve" >&2
  exit 2
fi

revision=$(git rev-parse HEAD)
worker_sha256=$(sha256sum "${worker}" | awk '{print $1}')
freeze_sha256=$(sha256sum "${freeze}" | awk '{print $1}')
submit_epoch=$(date +%s)
submit_args=(
  --partition=q-h800
  --gres=gpu:h800:1
  --cpus-per-task=12
  --mem=96G
  --time=00:30:00
  --mail-user="${notify_email}"
  --mail-type=ALL
  --export=NONE
)
/usr/local/slurm/bin/sbatch --test-only "${submit_args[@]}" "${worker}" \
  "${revision}" "${worker_sha256}" "${freeze_sha256}" "${submit_epoch}" >/dev/null
submission=$(
  /usr/local/slurm/bin/sbatch "${submit_args[@]}" "${worker}" \
    "${revision}" "${worker_sha256}" "${freeze_sha256}" "${submit_epoch}"
)
job_id=${submission##* }
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse DECAR full materialization job ID: ${submission}" >&2
  exit 2
fi
printf '%s\n' "${submission}"
printf 'infographicvqa_decar_full_materialization_job_id=%s code_revision=%s gpu_type=h800 gpu_count=1\n' \
  "${job_id}" "${revision}"
