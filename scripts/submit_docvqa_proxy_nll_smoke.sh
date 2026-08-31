#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
worker="${repo_dir}/scripts/slurm_docvqa_proxy_nll_smoke.sh"
scorer="${repo_dir}/scripts/score_visual_action_answer_nll.py"
module="${repo_dir}/src/beyond_entropy/answer_likelihood.py"
protocol="${repo_dir}/artifacts/docvqa-train-factorized-v2/ops/proxy-to-outcome-cross-domain-protocol-v1.md"
implementation="${repo_dir}/artifacts/docvqa-train-factorized-v2/ops/proxy-to-outcome-cross-domain-implementation-v1.md"

if [[ ! -r "${mail_file}" ]]; then
  echo "missing private Slurm email file" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != "yihangc@connect.hku.hk" ]]; then
  echo "invalid DocVQA notification email" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before DocVQA proxy smoke submission" >&2
  exit 2
fi
for path in "${worker}" "${scorer}" "${module}" "${protocol}" "${implementation}"; do
  if [[ ! -s "${path}" ]]; then
    echo "missing DocVQA proxy smoke input: ${path}" >&2
    exit 2
  fi
done

code_revision=$(git -C "${repo_dir}" rev-parse HEAD)
scorer_sha256=$(sha256sum "${scorer}" | cut -d ' ' -f 1)
module_sha256=$(sha256sum "${module}" | cut -d ' ' -f 1)
worker_sha256=$(sha256sum "${worker}" | cut -d ' ' -f 1)
protocol_sha256=$(sha256sum "${protocol}" | cut -d ' ' -f 1)
implementation_sha256=$(sha256sum "${implementation}" | cut -d ' ' -f 1)
export_args="ALL,BE_DOCVQA_PROXY_EXPECTED_CODE_REVISION=${code_revision},BE_DOCVQA_PROXY_SCORER_SHA256=${scorer_sha256},BE_DOCVQA_PROXY_MODULE_SHA256=${module_sha256},BE_DOCVQA_PROXY_WORKER_SHA256=${worker_sha256},BE_DOCVQA_PROXY_PROTOCOL_SHA256=${protocol_sha256},BE_DOCVQA_PROXY_IMPLEMENTATION_SHA256=${implementation_sha256}"
submission=$(
  /usr/local/slurm/bin/sbatch \
    --mail-user="${notify_email}" \
    --mail-type=ALL \
    --export="${export_args}" \
    "${worker}"
)
job_id=${submission##* }
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse DocVQA proxy smoke job ID: ${submission}" >&2
  exit 2
fi
printf '%s\n' "${submission}"
printf 'docvqa_proxy_nll_smoke_job_id=%s code_revision=%s gpu_type=rtx_4090 gpu_count=1\n' \
  "${job_id}" "${code_revision}"
