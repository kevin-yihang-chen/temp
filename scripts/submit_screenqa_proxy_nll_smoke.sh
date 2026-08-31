#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
worker="${repo_dir}/scripts/slurm_screenqa_proxy_nll_smoke.sh"
scorer="${repo_dir}/scripts/score_visual_action_answer_nll.py"
module="${repo_dir}/src/beyond_entropy/answer_likelihood.py"

if [[ ! -r "${mail_file}" ]]; then
  echo "missing private Slurm email file" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != "yihangc@connect.hku.hk" ]]; then
  echo "invalid ScreenQA notification email" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before ScreenQA proxy-NLL smoke submission" >&2
  exit 2
fi
for path in "${worker}" "${scorer}" "${module}"; do
  if [[ ! -s "${path}" ]]; then
    echo "missing ScreenQA proxy-NLL smoke implementation: ${path}" >&2
    exit 2
  fi
done

code_revision=$(git -C "${repo_dir}" rev-parse HEAD)
scorer_sha256=$(sha256sum "${scorer}" | cut -d ' ' -f 1)
module_sha256=$(sha256sum "${module}" | cut -d ' ' -f 1)
export_args="ALL,BE_PROXY_EXPECTED_CODE_REVISION=${code_revision},BE_PROXY_SCORER_SHA256=${scorer_sha256},BE_PROXY_MODULE_SHA256=${module_sha256}"
submission=$(
  /usr/local/slurm/bin/sbatch \
    --mail-user="${notify_email}" \
    --mail-type=ALL \
    --export="${export_args}" \
    "${worker}"
)
job_id=${submission##* }
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse ScreenQA proxy-NLL smoke job ID: ${submission}" >&2
  exit 2
fi
printf '%s\n' "${submission}"
printf 'screenqa_proxy_nll_smoke_job_id=%s code_revision=%s gpu_type=rtx_4090 gpu_count=1\n' \
  "${job_id}" "${code_revision}"
