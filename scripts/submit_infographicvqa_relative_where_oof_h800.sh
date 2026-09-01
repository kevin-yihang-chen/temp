#!/usr/bin/env bash
set -euo pipefail

export PATH=/usr/local/slurm/bin:/usr/local/bin:/usr/bin:/bin
repo=/userhome/cs3/yihangc/Documents/beyond-entropy
root="${repo}/artifacts/infographicvqa-train-v1/decar-v1/full-qwen7b-v1"
worker="${repo}/scripts/slurm_infographicvqa_relative_where_oof_h800.sh"
fit_runner="${repo}/scripts/fit_infographicvqa_relative_where_oof.py"
eval_runner="${repo}/scripts/evaluate_infographicvqa_relative_where_oof.py"
train_module="${repo}/src/beyond_entropy/infographicvqa_relative_where.py"
eval_module="${repo}/src/beyond_entropy/infographicvqa_relative_where_evaluation.py"
decar_eval_module="${repo}/src/beyond_entropy/infographicvqa_decar_evaluation.py"
protocol="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-relative-where-oof-protocol-v1.md"
resource_amendment="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-relative-where-resource-amendment-v1.md"
output_dir="${root}/relative-where-oof-v1"
cd "${repo}"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before relative-where submission" >&2
  exit 2
fi
for path in "${worker}" "${fit_runner}" "${eval_runner}" "${train_module}" \
  "${eval_module}" "${decar_eval_module}" "${protocol}" \
  "${resource_amendment}" "${root}/merged-rollouts/rollouts.jsonl" \
  "${root}/merged-nll/answer-nll.jsonl" \
  "${root}/merged-features/features-label-free.pt" \
  "${root}/evaluation-v1/bootstrap-indices.npy"; do
  if [[ ! -f "${path}" ]]; then
    echo "relative-where input is incomplete: ${path}" >&2
    exit 2
  fi
done
if [[ -e "${output_dir}" ]]; then
  echo "relative-where output already exists" >&2
  exit 2
fi

quota=$(/usr/local/bin/show-cpu-gpu-quota)
gpu_limit=$(sed -n 's/^GPU Quota in Minutes: \([0-9][0-9]*\), Used: \([0-9][0-9]*\)$/\1/p' <<< "${quota}")
gpu_used=$(sed -n 's/^GPU Quota in Minutes: \([0-9][0-9]*\), Used: \([0-9][0-9]*\)$/\2/p' <<< "${quota}")
cpu_limit=$(sed -n 's/^CPU Quota in Minutes: \([0-9][0-9]*\), Used: \([0-9][0-9]*\)$/\1/p' <<< "${quota}")
cpu_used=$(sed -n 's/^CPU Quota in Minutes: \([0-9][0-9]*\), Used: \([0-9][0-9]*\)$/\2/p' <<< "${quota}")
if [[ -z "${gpu_limit}" || -z "${gpu_used}" || -z "${cpu_limit}" \
  || -z "${cpu_used}" || $((gpu_limit - gpu_used)) -lt 60 \
  || $((cpu_limit - cpu_used)) -lt 720 ]]; then
  echo "relative-where needs 60 GPU-minutes and 720 CPU-minutes in reserve" >&2
  exit 2
fi

revision=$(git rev-parse HEAD)
worker_sha256=$(sha256sum "${worker}" | awk '{print $1}')
fit_runner_sha256=$(sha256sum "${fit_runner}" | awk '{print $1}')
eval_runner_sha256=$(sha256sum "${eval_runner}" | awk '{print $1}')
train_module_sha256=$(sha256sum "${train_module}" | awk '{print $1}')
eval_module_sha256=$(sha256sum "${eval_module}" | awk '{print $1}')
decar_eval_sha256=$(sha256sum "${decar_eval_module}" | awk '{print $1}')
protocol_sha256=$(sha256sum "${protocol}" | awk '{print $1}')
resource_amendment_sha256=$(sha256sum "${resource_amendment}" | awk '{print $1}')
submit_epoch=$(date +%s)
args=("${revision}" "${worker_sha256}" "${fit_runner_sha256}" \
  "${eval_runner_sha256}" "${train_module_sha256}" "${eval_module_sha256}" \
  "${decar_eval_sha256}" "${protocol_sha256}" \
  "${resource_amendment_sha256}" "${submit_epoch}")
/usr/local/slurm/bin/sbatch --test-only --export=NONE "${worker}" "${args[@]}" >/dev/null
submission=$(/usr/local/slurm/bin/sbatch --parsable --export=NONE "${worker}" "${args[@]}")
job_id=${submission%%;*}
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse relative-where Slurm job ID: ${submission}" >&2
  exit 2
fi
printf 'infographicvqa_relative_where_job_id=%s code_revision=%s gpu_type=h800 gpu_count=1 cpus=12\n' \
  "${job_id}" "${revision}"
