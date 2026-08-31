#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -gt 1 || ( "$#" -eq 1 && "$1" != "--resume" ) ]]; then
  echo "usage: $0 [--resume]" >&2
  exit 2
fi
resume_mode=0
if [[ "$#" -eq 1 ]]; then resume_mode=1; fi

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
freeze="${repo_dir}/artifacts/docvqa-train-factorized-v2/reserve-comparator/toolgate-proxy-v1/execution-freeze-v1/freeze.json"
manifest_dir="${repo_dir}/data/docvqa-train-factorized-v2/reserve-toolgate-v1"
manifest="${manifest_dir}/manifest.jsonl"
manifest_audit="${repo_dir}/data/docvqa-train-factorized-v2/reserve-toolgate-v1.audit.json"
evaluation_root="${repo_dir}/artifacts/docvqa-train-factorized-v2/reserve-comparator/toolgate-proxy-v1/reserve-evaluation-v1"
run_root="${evaluation_root}/qwen3b-c4-seed0"
feature_dir="${evaluation_root}/attention-semantic-v1"
score_dir="${evaluation_root}/policy-scores"
result="${evaluation_root}/evaluation/report.json"
export_worker="${repo_dir}/scripts/slurm_docvqa_reserve_toolgate_export.sh"
pipeline_worker="${repo_dir}/scripts/slurm_docvqa_reserve_toolgate_pipeline.sh"

if [[ ! -r "${mail_file}" ]]; then echo "missing private Slurm email file" >&2; exit 2; fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != "yihangc@connect.hku.hk" ]]; then
  echo "invalid reserve notification email" >&2
  exit 2
fi
for path in "${freeze}" "${export_worker}" "${pipeline_worker}"; do
  if [[ ! -s "${path}" ]]; then echo "missing reserve submission input: ${path}" >&2; exit 2; fi
done
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before reserve submission" >&2
  exit 2
fi
if [[ -e "${result}" ]]; then echo "reserve one-shot result already exists" >&2; exit 2; fi
if [[ "${resume_mode}" == 0 ]]; then
  if [[ -e "${manifest_dir}" || -e "${manifest_audit}" || -e "${evaluation_root}" ]]; then
    echo "fresh reserve submission refuses existing outputs" >&2
    exit 2
  fi
else
  if [[ ! -s "${manifest}" || ! -s "${manifest_audit}" ]]; then
    echo "reserve resume requires a completed manifest export" >&2
    exit 2
  fi
fi

active_jobs=$(/usr/local/slurm/bin/squeue -h -u "${USER}" -t PENDING,RUNNING,CONFIGURING,COMPLETING | wc -l)
if [[ "${active_jobs}" -ne 0 ]]; then
  echo "reserve submission requires the account's sole job slot to be free" >&2
  exit 2
fi
quota_output=$(/usr/local/bin/show-cpu-gpu-quota)
gpu_limit=$(awk '/GPU Quota in Minutes:/ {print $5}' <<< "${quota_output}")
gpu_used=$(awk '/GPU Quota in Minutes:/ {print $7}' <<< "${quota_output}")
gpu_limit=${gpu_limit%,}
if [[ ! "${gpu_limit}" =~ ^[0-9]+$ || ! "${gpu_used}" =~ ^[0-9]+$ ]]; then
  echo "could not parse live GPU quota" >&2
  exit 2
fi
gpu_remaining=$((gpu_limit - gpu_used))
if [[ "${gpu_remaining}" -lt 360 ]]; then
  echo "reserve pipeline requires at least 360 remaining GPU-minutes; remaining=${gpu_remaining}" >&2
  exit 2
fi

code_revision=$(git -C "${repo_dir}" rev-parse HEAD)
freeze_sha256=$(sha256sum "${freeze}" | cut -d ' ' -f 1)
common_export="PATH=/usr/local/bin:/usr/bin:/bin,LANG=C.UTF-8,BE_RESERVE_FREEZE=${freeze},BE_RESERVE_FREEZE_SHA256=${freeze_sha256},BE_RESERVE_EXPECTED_CODE_REVISION=${code_revision},BE_RESERVE_MANIFEST_DIR=${manifest_dir},BE_RESERVE_MANIFEST_AUDIT=${manifest_audit}"
dependency_args=()
export_job_id=""
if [[ "${resume_mode}" == 0 ]]; then
  export_submission=$(
    /usr/local/slurm/bin/sbatch \
      --partition=debug \
      --cpus-per-task=8 \
      --mem=64G \
      --time=02:00:00 \
      --mail-user="${notify_email}" \
      --mail-type=ALL \
      --export="${common_export}" \
      "${export_worker}"
  )
  export_job_id=${export_submission##* }
  if [[ ! "${export_job_id}" =~ ^[0-9]+$ ]]; then
    echo "could not parse reserve export job ID: ${export_submission}" >&2
    exit 2
  fi
  printf '%s\n' "${export_submission}"
  dependency_args=(--dependency="afterok:${export_job_id}")
fi

pipeline_export="${common_export},BE_RESERVE_MANIFEST=${manifest},BE_RESERVE_RUN_ROOT=${run_root},BE_RESERVE_FEATURE_DIR=${feature_dir},BE_RESERVE_SCORE_DIR=${score_dir},BE_RESERVE_RESULT=${result},BE_RESERVE_RESUME=${resume_mode}"
pipeline_submission=$(
  /usr/local/slurm/bin/sbatch \
    "${dependency_args[@]}" \
    --partition=q-h800 \
    --gres=gpu:h800:4 \
    --cpus-per-task=32 \
    --mem=384G \
    --time=02:00:00 \
    --mail-user="${notify_email}" \
    --mail-type=ALL \
    --export="${pipeline_export}" \
    "${pipeline_worker}"
)
pipeline_job_id=${pipeline_submission##* }
if [[ ! "${pipeline_job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse reserve pipeline job ID: ${pipeline_submission}" >&2
  exit 2
fi
printf '%s\n' "${pipeline_submission}"
printf 'reserve_export_job_id=%s reserve_pipeline_job_id=%s gpu_type=h800 gpu_count=4 code_revision=%s remaining_gpu_minutes_before_submit=%s\n' \
  "${export_job_id:-skipped}" "${pipeline_job_id}" "${code_revision}" "${gpu_remaining}"
