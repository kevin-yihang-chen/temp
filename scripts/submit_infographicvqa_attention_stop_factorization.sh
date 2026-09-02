#!/usr/bin/env bash
set -euo pipefail

export PATH=/usr/local/slurm/bin:/usr/local/bin:/usr/bin:/bin
repo=/userhome/cs3/yihangc/Documents/beyond-entropy
root="${repo}/artifacts/infographicvqa-train-v1/decar-v1/full-qwen7b-v1"
attention_root="${root}/attention-where-v1"
worker="${repo}/scripts/slurm_infographicvqa_attention_stop_factorization.sh"
runner="${repo}/scripts/diagnose_infographicvqa_attention_stop_factorization.py"
module="${repo}/src/beyond_entropy/infographicvqa_attention_stop_diagnostic.py"
protocol="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-attention-stop-factorization-diagnostic-protocol-v1.md"
output_dir="${attention_root}/attention-stop-factorization-v1"
cd "${repo}"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before attention-stop diagnostic" >&2
  exit 2
fi
for path in "${worker}" "${runner}" "${module}" "${protocol}" \
  "${attention_root}/merged-features/features-question-region-attention-label-free.pt" \
  "${attention_root}/evaluation-v1/complete.json"; do
  if [[ ! -f "${path}" ]]; then
    echo "attention-stop diagnostic input is incomplete: ${path}" >&2
    exit 2
  fi
done
if [[ -e "${output_dir}" ]]; then
  echo "attention-stop diagnostic output already exists" >&2
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
  echo "attention-stop diagnostic needs 45 GPU-minutes and 180 CPU-minutes" >&2
  exit 2
fi
revision=$(git rev-parse HEAD)
worker_sha256=$(sha256sum "${worker}" | awk '{print $1}')
runner_sha256=$(sha256sum "${runner}" | awk '{print $1}')
module_sha256=$(sha256sum "${module}" | awk '{print $1}')
protocol_sha256=$(sha256sum "${protocol}" | awk '{print $1}')
submit_epoch=$(date +%s)
args=("${revision}" "${worker_sha256}" "${runner_sha256}" "${module_sha256}" \
  "${protocol_sha256}" "${submit_epoch}")
/usr/local/slurm/bin/sbatch --test-only --export=NONE "${worker}" "${args[@]}" >/dev/null
submission=$(/usr/local/slurm/bin/sbatch --parsable --export=NONE "${worker}" "${args[@]}")
job_id=${submission%%;*}
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse attention-stop diagnostic Slurm job ID: ${submission}" >&2
  exit 2
fi
printf 'attention_stop_diagnostic_job_id=%s code_revision=%s evaluator_device=cpu reserved_gpu=rtx_4090\n' \
  "${job_id}" "${revision}"
