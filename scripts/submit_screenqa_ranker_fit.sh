#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
manifest_dir="${repo_dir}/artifacts/screenqa-train-factorized-v1/ranker-manifest-v1"
manifest_sha256=a2b6941e2a073b24571d2ccb50960f7c1cd70cb0ce53dc8339c7ec44a47f67ec
manifest_audit_sha256=4c40f9c80eecabcf0a9cce38e64ca2df7603fde67f05c37a8cbd019a3811a3ef
bank_code_revision=d1b8dd10524d8610b19c19a91450cd2d5eac2127
bank_root="${repo_dir}/artifacts/screenqa-train-factorized-v1/ranker-rollouts-v1/merged"
rollouts="${bank_root}/rollouts.jsonl"
merge_audit="${bank_root}/rollouts.merge.json"
protocol="${repo_dir}/artifacts/screenqa-train-factorized-v1/ops/ranker-development-protocol-v1.md"
protocol_sha256=c6118d8a013a171c3eecad374a3271e3bf00dfd199864d3efaab27c7b44e36b7
fit_root="${repo_dir}/artifacts/screenqa-train-factorized-v1/ranker-training/low-capacity-oof-v1"

if [[ ! -r "${mail_file}" ]]; then
  echo "missing private Slurm email file" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != "yihangc@connect.hku.hk" ]]; then
  echo "invalid ScreenQA notification email" >&2
  exit 2
fi
if [[ ! -s "${rollouts}" || ! -s "${merge_audit}" ]]; then
  echo "complete ScreenQA ranker merge is not available" >&2
  exit 2
fi
if [[ -e "${fit_root}" ]]; then
  echo "refusing to reuse ScreenQA fit root: ${fit_root}" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before ScreenQA fit submission" >&2
  exit 2
fi
actual_protocol_sha256=$(sha256sum "${protocol}")
actual_protocol_sha256=${actual_protocol_sha256%% *}
if [[ "${actual_protocol_sha256}" != "${protocol_sha256}" ]]; then
  echo "ScreenQA ranker-development protocol SHA-256 mismatch" >&2
  exit 2
fi
rollouts_sha256=$(sha256sum "${rollouts}")
rollouts_sha256=${rollouts_sha256%% *}
merge_audit_sha256=$(sha256sum "${merge_audit}")
merge_audit_sha256=${merge_audit_sha256%% *}
code_revision=$(git -C "${repo_dir}" rev-parse HEAD)

export_args="ALL,BE_SCREENQA_MANIFEST_DIR=${manifest_dir},BE_SCREENQA_MANIFEST_SHA256=${manifest_sha256},BE_SCREENQA_MANIFEST_AUDIT_SHA256=${manifest_audit_sha256},BE_SCREENQA_RANKER_ROLLOUTS=${rollouts},BE_SCREENQA_RANKER_ROLLOUTS_SHA256=${rollouts_sha256},BE_SCREENQA_MERGE_AUDIT=${merge_audit},BE_SCREENQA_MERGE_AUDIT_SHA256=${merge_audit_sha256},BE_SCREENQA_BANK_CODE_REVISION=${bank_code_revision},BE_SCREENQA_PROTOCOL=${protocol},BE_SCREENQA_PROTOCOL_SHA256=${protocol_sha256},BE_SCREENQA_FIT_ROOT=${fit_root},BE_SCREENQA_EXPECTED_CODE_REVISION=${code_revision}"

submission=$(
  sbatch \
    --mail-user="${notify_email}" \
    --mail-type=ALL \
    --export="${export_args}" \
    "${repo_dir}/scripts/slurm_screenqa_ranker_fit.sh"
)
job_id=${submission##* }
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse ScreenQA ranker-fit job ID: ${submission}" >&2
  exit 2
fi
printf '%s\n' "${submission}"
printf 'screenqa_ranker_fit_job_id=%s code_revision=%s rollouts_sha256=%s\n' \
  "${job_id}" "${code_revision}" "${rollouts_sha256}"
