#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 || "$1" != "--recover-from" || ! "$2" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 --recover-from PRIOR_JOB_ID" >&2
  exit 2
fi
prior_job_id=$2

export PATH=/usr/local/slurm/bin:/usr/local/bin:/usr/bin:/bin
repo=/userhome/cs3/yihangc/Documents/beyond-entropy
wrapper="${repo}/scripts/slurm_infographicvqa_decar_full_recovery_h800.sh"
recovery_freeze="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-decar-full-recovery-freeze-v1.md"
original_worker="${repo}/scripts/slurm_infographicvqa_decar_full_h800.sh"
generation_freeze="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-decar-full-generation-freeze-v1.md"
root="${repo}/artifacts/infographicvqa-train-v1/decar-v1/full-qwen7b-v1"
scientific_revision=5b1b0211372ccb96ec21fc55fa954d427a5504b5
cd "${repo}"

launcher_revision=$(git rev-parse HEAD)
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before DECAR recovery submission" >&2
  exit 2
fi
if [[ ! -f "${wrapper}" || ! -f "${recovery_freeze}" \
  || ! -f "${original_worker}" || ! -f "${generation_freeze}" ]]; then
  echo "DECAR recovery freeze is incomplete" >&2
  exit 2
fi
if [[ ! -d "${root}" || -z "$(find "${root}" -mindepth 1 -print -quit)" ]]; then
  echo "DECAR recovery found no checkpointed full output" >&2
  exit 2
fi
if [[ -e "${root}/execution/job-${prior_job_id}.json" ]]; then
  echo "DECAR recovery refuses a prior job with a completed execution record" >&2
  exit 2
fi

prior_record=$(/usr/local/slurm/bin/scontrol show job -o "${prior_job_id}")
prior_state=$(sed -n 's/.*JobState=\([^ ]*\).*/\1/p' <<< "${prior_record}" | head -n 1)
case "${prior_state}" in
  TIMEOUT|NODE_FAIL|PREEMPTED|FAILED|OUT_OF_MEMORY|BOOT_FAIL|DEADLINE|REVOKED|CANCELLED*) ;;
  *)
    echo "DECAR recovery requires a terminal unsuccessful prior job, got ${prior_state:-unknown}" >&2
    exit 2
    ;;
esac

quota=$(/usr/local/bin/show-cpu-gpu-quota)
gpu_limit=$(sed -n 's/^GPU Quota in Minutes: \([0-9][0-9]*\), Used: \([0-9][0-9]*\)$/\1/p' <<< "${quota}")
gpu_used=$(sed -n 's/^GPU Quota in Minutes: \([0-9][0-9]*\), Used: \([0-9][0-9]*\)$/\2/p' <<< "${quota}")
if [[ -z "${gpu_limit}" || -z "${gpu_used}" || $((gpu_limit - gpu_used)) -lt 1980 ]]; then
  echo "DECAR recovery needs a 1980 GPU-minute reserve" >&2
  exit 2
fi

wrapper_sha256=$(sha256sum "${wrapper}" | awk '{print $1}')
recovery_freeze_sha256=$(sha256sum "${recovery_freeze}" | awk '{print $1}')
original_worker_sha256=$(sha256sum "${original_worker}" | awk '{print $1}')
generation_freeze_sha256=$(sha256sum "${generation_freeze}" | awk '{print $1}')
submit_epoch=$(date +%s)
args=(
  "${scientific_revision}" "${launcher_revision}" "${wrapper_sha256}"
  "${recovery_freeze_sha256}" "${original_worker_sha256}"
  "${generation_freeze_sha256}" "${prior_job_id}" "${submit_epoch}"
)
/usr/local/slurm/bin/sbatch --test-only --export=NONE "${wrapper}" "${args[@]}" >/dev/null
submission=$(/usr/local/slurm/bin/sbatch --parsable --export=NONE "${wrapper}" "${args[@]}")
job_id=${submission%%;*}
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse DECAR recovery Slurm job ID: ${submission}" >&2
  exit 2
fi
printf 'infographicvqa_decar_recovery_job_id=%s prior_job_id=%s scientific_revision=%s launcher_revision=%s gpu_type=h800 gpu_count=4\n' \
  "${job_id}" "${prior_job_id}" "${scientific_revision}" "${launcher_revision}"
