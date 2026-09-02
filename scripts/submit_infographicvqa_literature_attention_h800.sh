#!/usr/bin/env bash
set -euo pipefail

resume=0
if [[ "${1:-}" == "--resume" ]]; then resume=1; shift; fi
if [[ "$#" -ne 0 ]]; then
  echo "usage: $0 [--resume]" >&2
  exit 2
fi

export PATH=/usr/local/slurm/bin:/usr/local/bin:/usr/bin:/bin
repo=/userhome/cs3/yihangc/Documents/beyond-entropy
worker="${repo}/scripts/slurm_infographicvqa_literature_attention_h800.sh"
protocol="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-literature-attention-where-protocol-20260902-pending.md"
blind_audit="${repo}/artifacts/docvqa-train-factorized-v2/ops/attention-crop-literature-blind-audit-20260902-pending.md"
output_root="${repo}/artifacts/infographicvqa-train-v1/decar-v1/full-qwen7b-v1/literature-attention-where-v1"
cd "${repo}"
revision=$(git rev-parse HEAD)
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before literature-attention submission" >&2
  exit 2
fi
for path in "${worker}" "${protocol}" "${blind_audit}"; do
  if [[ ! -f "${path}" ]]; then
    echo "literature-attention required binding is absent: ${path}" >&2
    exit 2
  fi
done
if [[ -d "${output_root}" && -n "$(find "${output_root}" -mindepth 1 -print -quit)" \
  && "${resume}" != 1 ]]; then
  echo "existing literature-attention outputs require --resume" >&2
  exit 2
fi
quota=$(/usr/local/bin/show-cpu-gpu-quota)
gpu_limit=$(sed -n 's/^GPU Quota in Minutes: \([0-9][0-9]*\), Used: \([0-9][0-9]*\)$/\1/p' <<< "${quota}")
gpu_used=$(sed -n 's/^GPU Quota in Minutes: \([0-9][0-9]*\), Used: \([0-9][0-9]*\)$/\2/p' <<< "${quota}")
if [[ -z "${gpu_limit}" || -z "${gpu_used}" || $((gpu_limit - gpu_used)) -lt 960 ]]; then
  echo "literature-attention needs a 960 GPU-minute reserve" >&2
  exit 2
fi
worker_sha256=$(sha256sum "${worker}" | awk '{print $1}')
protocol_sha256=$(sha256sum "${protocol}" | awk '{print $1}')
blind_audit_sha256=$(sha256sum "${blind_audit}" | awk '{print $1}')
submit_epoch=$(date +%s)
args=("${revision}" "${worker_sha256}" "${protocol_sha256}" "${blind_audit_sha256}" "${resume}" "${submit_epoch}")
/usr/local/slurm/bin/sbatch --test-only --export=NONE "${worker}" "${args[@]}" >/dev/null
submission=$(/usr/local/slurm/bin/sbatch --parsable --export=NONE "${worker}" "${args[@]}")
job_id=${submission%%;*}
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse literature-attention Slurm job ID: ${submission}" >&2
  exit 2
fi
printf 'infographicvqa_literature_attention_job_id=%s code_revision=%s gpu_type=h800 gpu_count=2 resume=%s\n' \
  "${job_id}" "${revision}" "${resume}"
