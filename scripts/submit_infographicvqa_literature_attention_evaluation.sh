#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 FEATURE_JOB_ID" >&2
  exit 2
fi
feature_job_id=$1
export PATH=/usr/local/slurm/bin:/usr/local/bin:/usr/bin:/bin
repo=/userhome/cs3/yihangc/Documents/beyond-entropy
root="${repo}/artifacts/infographicvqa-train-v1/decar-v1/full-qwen7b-v1"
literature_root="${root}/literature-attention-where-v1"
worker="${repo}/scripts/slurm_infographicvqa_literature_attention_evaluation.sh"
runner="${repo}/scripts/evaluate_infographicvqa_literature_attention_where.py"
eval_module="${repo}/src/beyond_entropy/infographicvqa_literature_attention_evaluation.py"
protocol="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-literature-attention-where-protocol-20260902-pending.md"
feature_complete="${literature_root}/complete.json"
feature_execution="${literature_root}/execution/job-${feature_job_id}.json"
output_dir="${literature_root}/evaluation-v1"
cd "${repo}"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before literature-attention evaluation" >&2
  exit 2
fi
for path in "${worker}" "${runner}" "${eval_module}" "${protocol}" \
  "${feature_complete}" "${feature_execution}" \
  "${literature_root}/merged-features/features-literature-attention-label-free.pt" \
  "${root}/attention-where-v1/evaluation-v1/complete.json"; do
  if [[ ! -f "${path}" ]]; then
    echo "literature-attention evaluation input is incomplete: ${path}" >&2
    exit 2
  fi
done
if [[ -e "${output_dir}" ]]; then
  echo "literature-attention evaluation output already exists" >&2
  exit 2
fi
quota=$(/usr/local/bin/show-cpu-gpu-quota)
gpu_limit=$(sed -n 's/^GPU Quota in Minutes: \([0-9][0-9]*\), Used: \([0-9][0-9]*\)$/\1/p' <<< "${quota}")
gpu_used=$(sed -n 's/^GPU Quota in Minutes: \([0-9][0-9]*\), Used: \([0-9][0-9]*\)$/\2/p' <<< "${quota}")
cpu_limit=$(sed -n 's/^CPU Quota in Minutes: \([0-9][0-9]*\), Used: \([0-9][0-9]*\)$/\1/p' <<< "${quota}")
cpu_used=$(sed -n 's/^CPU Quota in Minutes: \([0-9][0-9]*\), Used: \([0-9][0-9]*\)$/\2/p' <<< "${quota}")
if [[ -z "${gpu_limit}" || -z "${gpu_used}" || -z "${cpu_limit}" \
  || -z "${cpu_used}" || $((gpu_limit - gpu_used)) -lt 60 \
  || $((cpu_limit - cpu_used)) -lt 240 ]]; then
  echo "literature-attention evaluation needs 60 GPU-minutes and 240 CPU-minutes" >&2
  exit 2
fi
revision=$(git rev-parse HEAD)
worker_sha256=$(sha256sum "${worker}" | awk '{print $1}')
runner_sha256=$(sha256sum "${runner}" | awk '{print $1}')
eval_sha256=$(sha256sum "${eval_module}" | awk '{print $1}')
protocol_sha256=$(sha256sum "${protocol}" | awk '{print $1}')
submit_epoch=$(date +%s)
args=("${revision}" "${worker_sha256}" "${runner_sha256}" "${eval_sha256}" \
  "${protocol_sha256}" "${feature_job_id}" "${submit_epoch}")
/usr/local/slurm/bin/sbatch --test-only --export=NONE "${worker}" "${args[@]}" >/dev/null
submission=$(/usr/local/slurm/bin/sbatch --parsable --export=NONE "${worker}" "${args[@]}")
job_id=${submission%%;*}
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse literature-attention evaluation job ID: ${submission}" >&2
  exit 2
fi
printf 'literature_attention_evaluation_job_id=%s feature_job_id=%s code_revision=%s evaluator_device=cpu\n' \
  "${job_id}" "${feature_job_id}" "${revision}"
