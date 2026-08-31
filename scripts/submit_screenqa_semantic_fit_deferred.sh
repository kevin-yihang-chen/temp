#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
deferred="${repo_dir}/scripts/slurm_deferred_screenqa_semantic_fit_submit.sh"
submitter="${repo_dir}/scripts/submit_screenqa_semantic_fit.sh"
feature_job_id=197065

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
  echo "tracked worktree must be clean before deferred ScreenQA semantic fit submission" >&2
  exit 2
fi
for path in "${deferred}" "${submitter}"; do
  if [[ ! -x "${path}" ]]; then
    echo "missing executable ScreenQA semantic fit component: ${path}" >&2
    exit 2
  fi
done
deferred_sha256=$(sha256sum "${deferred}")
deferred_sha256=${deferred_sha256%% *}
submitter_sha256=$(sha256sum "${submitter}")
submitter_sha256=${submitter_sha256%% *}
export_args="ALL,BE_SCREENQA_SEMANTIC_DEFERRED_SHA256=${deferred_sha256},BE_SCREENQA_SEMANTIC_FIT_SUBMITTER_SHA256=${submitter_sha256},BE_SCREENQA_FEATURE_JOB_ID=${feature_job_id}"
submission=$(
  /usr/local/slurm/bin/sbatch \
    --dependency="afterok:${feature_job_id}" \
    --kill-on-invalid-dep=yes \
    --mail-user="${notify_email}" \
    --mail-type=ALL \
    --export="${export_args}" \
    "${deferred}"
)
job_id=${submission##* }
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse deferred ScreenQA semantic fit job ID: ${submission}" >&2
  exit 2
fi
printf '%s\n' "${submission}"
printf 'screenqa_semantic_fit_deferred_job_id=%s dependency=afterok:%s gpu_count=0\n' \
  "${job_id}" "${feature_job_id}"
