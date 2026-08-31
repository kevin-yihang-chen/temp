#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
manifest_dir="${repo_dir}/artifacts/screenqa-train-factorized-v1/ranker-manifest-v1"
manifest_sha256=a2b6941e2a073b24571d2ccb50960f7c1cd70cb0ce53dc8339c7ec44a47f67ec
manifest_audit_sha256=4c40f9c80eecabcf0a9cce38e64ca2df7603fde67f05c37a8cbd019a3811a3ef
expected_states=14511
smoke_limit=200
shard_count=2
run_root="${repo_dir}/artifacts/screenqa-train-factorized-v1/ranker-smoke-v1"

if [[ ! -r "${mail_file}" ]]; then
  echo "missing private Slurm email file" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != "yihangc@connect.hku.hk" ]]; then
  echo "invalid ScreenQA notification email" >&2
  exit 2
fi
if [[ -e "${run_root}" ]]; then
  echo "refusing to reuse ScreenQA smoke run root: ${run_root}" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before ScreenQA smoke submission" >&2
  exit 2
fi
code_revision=$(git -C "${repo_dir}" rev-parse HEAD)

export_args="ALL,BE_SCREENQA_MANIFEST_DIR=${manifest_dir},BE_SCREENQA_MANIFEST_SHA256=${manifest_sha256},BE_SCREENQA_MANIFEST_AUDIT_SHA256=${manifest_audit_sha256},BE_SCREENQA_EXPECTED_STATES=${expected_states},BE_SCREENQA_RUN_ROOT=${run_root},BE_SCREENQA_EXPECTED_CODE_REVISION=${code_revision},BE_SCREENQA_SMOKE_LIMIT=${smoke_limit},BE_SCREENQA_SHARD_COUNT=${shard_count}"

PYTHONPATH="${repo_dir}/src" /userhome/cs3/yihangc/anaconda3/bin/python \
  "${repo_dir}/scripts/verify_screenqa_ranker_manifest.py" \
  --manifest-dir "${manifest_dir}" \
  --expected-manifest-sha256 "${manifest_sha256}" \
  --expected-audit-sha256 "${manifest_audit_sha256}" \
  --expected-states "${expected_states}"

array_submission=$(
  sbatch \
    --mail-user="${notify_email}" \
    --mail-type=ALL \
    --export="${export_args}" \
    "${repo_dir}/scripts/slurm_screenqa_ranker_smoke.sh"
)
array_job_id=${array_submission##* }
if [[ ! "${array_job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse ScreenQA smoke array job ID: ${array_submission}" >&2
  exit 2
fi

merge_submission=$(
  sbatch \
    --dependency="afterok:${array_job_id}" \
    --mail-user="${notify_email}" \
    --mail-type=ALL \
    --export="${export_args}" \
    "${repo_dir}/scripts/slurm_screenqa_ranker_smoke_merge.sh"
)
merge_job_id=${merge_submission##* }
if [[ ! "${merge_job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse ScreenQA smoke merge job ID: ${merge_submission}" >&2
  exit 2
fi

printf '%s\n' "${array_submission}" "${merge_submission}"
printf 'screenqa_smoke_array_job_id=%s merge_job_id=%s code_revision=%s\n' \
  "${array_job_id}" "${merge_job_id}" "${code_revision}"
