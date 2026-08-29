#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
source_dir=/userhome/cs3/yihangc/Data/beyond-entropy-docvqa-train-factorized-v2/source
protocol="${repo_dir}/docs/docvqa_train_factorized_v2_preregistration.md"
allocation="${repo_dir}/artifacts/docvqa-train-factorized-v2/allocation/allocation.json"
allocation_audit="${repo_dir}/artifacts/docvqa-train-factorized-v2/allocation/allocation.audit.json"

if [[ ! -r "${mail_file}" ]]; then
  echo "missing private Slurm email file" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != "yihangc@connect.hku.hk" ]]; then
  echo "invalid DocVQA notification email" >&2
  exit 2
fi
if [[ ! -f "${protocol}" ]]; then
  echo "missing DocVQA preregistration" >&2
  exit 2
fi
for shard_index in {00..11}; do
  shard="${source_dir}/DocVQA/train-000${shard_index}-of-00012.parquet"
  if [[ ! -s "${shard}" ]]; then
    echo "missing DocVQA train shard: ${shard}" >&2
    exit 2
  fi
done
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before DocVQA allocation submission" >&2
  exit 2
fi
code_revision=$(git -C "${repo_dir}" rev-parse HEAD)

submission=$(
  sbatch \
    --job-name=be-docvqa-fv2-allocation \
    --mail-user="${notify_email}" \
    --mail-type=ALL \
    --export="ALL,BE_DOCVQA_SOURCE_DIR=${source_dir},BE_DOCVQA_ALLOCATION=${allocation},BE_DOCVQA_ALLOCATION_AUDIT=${allocation_audit},BE_DOCVQA_PROTOCOL=${protocol},BE_DOCVQA_EXPECTED_CODE_REVISION=${code_revision}" \
    "${repo_dir}/scripts/slurm_docvqa_train_factorized_v2_allocation.sh"
)
job_id=${submission##* }
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse DocVQA allocation job ID: ${submission}" >&2
  exit 2
fi
printf '%s\n' "${submission}"
printf 'allocation_job_id=%s code_revision=%s\n' "${job_id}" "${code_revision}"
