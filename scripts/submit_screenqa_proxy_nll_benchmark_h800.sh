#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
worker="${repo_dir}/scripts/slurm_screenqa_proxy_nll_benchmark_h800.sh"
scorer="${repo_dir}/scripts/score_visual_action_answer_nll.py"
score_module="${repo_dir}/src/beyond_entropy/answer_likelihood.py"

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
  echo "tracked worktree must be clean before H800 benchmark submission" >&2
  exit 2
fi
for path in "${worker}" "${scorer}" "${score_module}"; do
  if [[ ! -s "${path}" ]]; then
    echo "missing ScreenQA H800 benchmark implementation: ${path}" >&2
    exit 2
  fi
done

code_revision=$(git -C "${repo_dir}" rev-parse HEAD)
scorer_sha256=$(sha256sum "${scorer}" | cut -d ' ' -f 1)
score_module_sha256=$(sha256sum "${score_module}" | cut -d ' ' -f 1)
worker_sha256=$(sha256sum "${worker}" | cut -d ' ' -f 1)
export_args="ALL,BE_PROXY_EXPECTED_CODE_REVISION=${code_revision},BE_PROXY_SCORER_SHA256=${scorer_sha256},BE_PROXY_SCORE_MODULE_SHA256=${score_module_sha256},BE_PROXY_BENCHMARK_WORKER_SHA256=${worker_sha256}"
submission=$(
  /usr/local/slurm/bin/sbatch \
    --mail-user="${notify_email}" \
    --mail-type=ALL \
    --export="${export_args}" \
    "${worker}"
)
job_id=${submission##* }
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse ScreenQA H800 benchmark job ID: ${submission}" >&2
  exit 2
fi
printf '%s\n' "${submission}"
printf 'screenqa_proxy_nll_h800_benchmark_job_id=%s code_revision=%s gpu_type=h800 gpu_count=1 decisions=64\n' \
  "${job_id}" "${code_revision}"
