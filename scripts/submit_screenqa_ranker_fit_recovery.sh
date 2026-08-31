#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
fit_root="${repo_dir}/artifacts/screenqa-train-factorized-v1/ranker-training/low-capacity-oof-v1"
rollouts="${repo_dir}/artifacts/screenqa-train-factorized-v1/ranker-rollouts-v1/merged/rollouts.jsonl"
protocol="${repo_dir}/artifacts/screenqa-train-factorized-v1/ops/ranker-development-protocol-v1.md"
worker="${repo_dir}/scripts/slurm_screenqa_ranker_fit_recovery.sh"
context_model="${fit_root}/context-geometry-oof-v1/model.json"
context_report="${fit_root}/context-geometry-oof-v1/report.json"
input_audit="${fit_root}/ranker-rollouts.audit.json"
spatial_dir="${fit_root}/spatial-context-geometry-oof-v1"
candidate_dir="${fit_root}/candidate-v1"
recovery_audit="${fit_root}/ranker-fit-recovery.audit.json"
previous_job_id=196911
previous_job_state=TIMEOUT
expected_code_revision=1174023b6ff4e00046eceb3783299aec286691e4
rollouts_sha256=0437d2a499adccb1b4e19eb0160583789cee00edf244718ecae9e290108bb8c9
protocol_sha256=c6118d8a013a171c3eecad374a3271e3bf00dfd199864d3efaab27c7b44e36b7
context_model_sha256=069e1e69ed6c74fe3d3ec95a201e9a13cc43150ae8e3792b595922f81b6493e5
context_report_sha256=3f1c0edf36832304808a57bd6cc34a702b5283716bb92d51c7c27d949f08174e
input_audit_sha256=0651debaeb5e742f6823e7321e8bfe8184a398a468e42e36dd033f68af74563c

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
  echo "tracked worktree must be clean before ScreenQA recovery submission" >&2
  exit 2
fi
actual_code_revision=$(git -C "${repo_dir}" rev-parse HEAD)
if [[ "${actual_code_revision}" != "${expected_code_revision}" ]]; then
  echo "ScreenQA recovery must use the original ranker-fit code revision" >&2
  exit 2
fi
for required_file in \
  "${rollouts}" \
  "${protocol}" \
  "${worker}" \
  "${context_model}" \
  "${context_report}" \
  "${input_audit}"; do
  if [[ ! -s "${required_file}" ]]; then
    echo "missing ScreenQA recovery input: ${required_file}" >&2
    exit 2
  fi
done
if [[ -e "${spatial_dir}" || -e "${candidate_dir}" || -e "${recovery_audit}" ]]; then
  echo "ScreenQA recovery output already exists" >&2
  exit 2
fi

previous_job=$(/usr/local/slurm/bin/scontrol show job "${previous_job_id}")
if [[ "${previous_job}" != *"JobState=${previous_job_state}"* ]]; then
  echo "ScreenQA predecessor is not in the registered TIMEOUT state" >&2
  exit 2
fi

check_sha256() {
  local path=$1
  local expected=$2
  local actual
  actual=$(sha256sum "${path}")
  actual=${actual%% *}
  if [[ "${actual}" != "${expected}" ]]; then
    echo "ScreenQA recovery SHA-256 mismatch: ${path}" >&2
    exit 2
  fi
}

check_sha256 "${rollouts}" "${rollouts_sha256}"
check_sha256 "${protocol}" "${protocol_sha256}"
check_sha256 "${context_model}" "${context_model_sha256}"
check_sha256 "${context_report}" "${context_report_sha256}"
check_sha256 "${input_audit}" "${input_audit_sha256}"
worker_sha256=$(sha256sum "${worker}")
worker_sha256=${worker_sha256%% *}

export_args="ALL,BE_SCREENQA_RANKER_ROLLOUTS=${rollouts},BE_SCREENQA_RANKER_ROLLOUTS_SHA256=${rollouts_sha256},BE_SCREENQA_PROTOCOL=${protocol},BE_SCREENQA_PROTOCOL_SHA256=${protocol_sha256},BE_SCREENQA_FIT_ROOT=${fit_root},BE_SCREENQA_EXPECTED_CODE_REVISION=${expected_code_revision},BE_SCREENQA_CONTEXT_MODEL_SHA256=${context_model_sha256},BE_SCREENQA_CONTEXT_REPORT_SHA256=${context_report_sha256},BE_SCREENQA_INPUT_AUDIT_SHA256=${input_audit_sha256},BE_SCREENQA_RECOVERY_WORKER_SHA256=${worker_sha256},BE_SCREENQA_PREVIOUS_JOB_ID=${previous_job_id},BE_SCREENQA_PREVIOUS_JOB_STATE=${previous_job_state}"

submission=$(
  sbatch \
    --mail-user="${notify_email}" \
    --mail-type=ALL \
    --export="${export_args}" \
    "${worker}"
)
job_id=${submission##* }
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse ScreenQA ranker-recovery job ID: ${submission}" >&2
  exit 2
fi
printf '%s\n' "${submission}"
printf 'screenqa_ranker_recovery_job_id=%s predecessor=%s code_revision=%s worker_sha256=%s\n' \
  "${job_id}" "${previous_job_id}" "${expected_code_revision}" "${worker_sha256}"
