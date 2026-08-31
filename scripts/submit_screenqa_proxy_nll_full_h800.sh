#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
root="${repo_dir}/artifacts/screenqa-train-factorized-v1/proxy-to-outcome-audit-v1/full-v1"
worker="${repo_dir}/scripts/slurm_screenqa_proxy_nll_full_h800.sh"
scorer="${repo_dir}/scripts/score_visual_action_answer_nll.py"
score_module="${repo_dir}/src/beyond_entropy/answer_likelihood.py"
merger="${repo_dir}/scripts/merge_visual_action_answer_nll.py"
analyzer="${repo_dir}/scripts/analyze_visual_action_proxy_outcomes.py"
audit_module="${repo_dir}/src/beyond_entropy/proxy_outcome_audit.py"
protocol="${repo_dir}/artifacts/screenqa-train-factorized-v1/ops/proxy-to-outcome-audit-protocol-v1.md"
implementation_contract="${repo_dir}/artifacts/screenqa-train-factorized-v1/ops/proxy-to-outcome-analysis-implementation-v1.md"
frozen_protocol_sha256=42d952c35ac16f5584ae4fa0f6849920ba0bdc86a3b7cdfd92fb8fdc79c3f129
frozen_implementation_sha256=c497ec89317cbfa6cc7fa2097b8be064c21a29132f95e446b649994ac65c117e
resume_mode=${BE_PROXY_FULL_RESUME:-0}

if [[ ! -r "${mail_file}" ]]; then
  echo "missing private Slurm email file" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != "yihangc@connect.hku.hk" ]]; then
  echo "invalid ScreenQA notification email" >&2
  exit 2
fi
if [[ "${resume_mode}" != 0 && "${resume_mode}" != 1 ]]; then
  echo "BE_PROXY_FULL_RESUME must be 0 or 1" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before full H800 proxy-NLL submission" >&2
  exit 2
fi
for path in "${worker}" "${scorer}" "${score_module}" "${merger}" "${analyzer}" "${audit_module}" "${protocol}" "${implementation_contract}"; do
  if [[ ! -s "${path}" ]]; then
    echo "missing full ScreenQA proxy-NLL implementation: ${path}" >&2
    exit 2
  fi
done
if [[ "$(sha256sum "${protocol}" | cut -d ' ' -f 1)" != "${frozen_protocol_sha256}" ]]; then
  echo "frozen ScreenQA proxy protocol hash mismatch" >&2
  exit 2
fi
if [[ "$(sha256sum "${implementation_contract}" | cut -d ' ' -f 1)" != "${frozen_implementation_sha256}" ]]; then
  echo "frozen ScreenQA proxy implementation contract hash mismatch" >&2
  exit 2
fi
if [[ -d "${root}" && -n "$(find "${root}" -mindepth 1 -print -quit)" && "${resume_mode}" != 1 ]]; then
  echo "existing full ScreenQA proxy-NLL outputs require BE_PROXY_FULL_RESUME=1" >&2
  exit 2
fi

code_revision=$(git -C "${repo_dir}" rev-parse HEAD)
scorer_sha256=$(sha256sum "${scorer}" | cut -d ' ' -f 1)
score_module_sha256=$(sha256sum "${score_module}" | cut -d ' ' -f 1)
merger_sha256=$(sha256sum "${merger}" | cut -d ' ' -f 1)
analyzer_sha256=$(sha256sum "${analyzer}" | cut -d ' ' -f 1)
audit_module_sha256=$(sha256sum "${audit_module}" | cut -d ' ' -f 1)
worker_sha256=$(sha256sum "${worker}" | cut -d ' ' -f 1)
export_args="ALL,BE_PROXY_EXPECTED_CODE_REVISION=${code_revision},BE_PROXY_SCORER_SHA256=${scorer_sha256},BE_PROXY_SCORE_MODULE_SHA256=${score_module_sha256},BE_PROXY_MERGER_SHA256=${merger_sha256},BE_PROXY_ANALYZER_SHA256=${analyzer_sha256},BE_PROXY_AUDIT_MODULE_SHA256=${audit_module_sha256},BE_PROXY_FULL_WORKER_SHA256=${worker_sha256},BE_PROXY_PROTOCOL_SHA256=${frozen_protocol_sha256},BE_PROXY_IMPLEMENTATION_CONTRACT_SHA256=${frozen_implementation_sha256},BE_PROXY_FULL_RESUME=${resume_mode}"
submission=$(
  /usr/local/slurm/bin/sbatch \
    --mail-user="${notify_email}" \
    --mail-type=ALL \
    --export="${export_args}" \
    "${worker}"
)
job_id=${submission##* }
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse full ScreenQA H800 proxy-NLL job ID: ${submission}" >&2
  exit 2
fi
printf '%s\n' "${submission}"
printf 'screenqa_proxy_nll_full_job_id=%s code_revision=%s gpu_type=h800 gpu_count=4 resume=%s\n' \
  "${job_id}" "${code_revision}" "${resume_mode}"
