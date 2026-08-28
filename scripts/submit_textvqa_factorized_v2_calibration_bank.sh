#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
candidate="${repo_dir}/artifacts/textvqa-train-factorized-v2/frozen-candidate/model.json"
candidate_audit="${repo_dir}/artifacts/textvqa-train-factorized-v2/frozen-candidate/audit.json"
protocol="${repo_dir}/docs/textvqa_factorized_fixed_sequence_preregistration.md"
allocation="${repo_dir}/data/textvqa-train-factorized-v2/allocation.json"
allocation_audit="${repo_dir}/data/textvqa-train-factorized-v2/allocation.audit.json"
role_dir="${repo_dir}/data/textvqa-train-factorized-v2/risk-calibration"
manifest="${role_dir}/manifest.jsonl"
manifest_provenance="${role_dir}/manifest.provenance.json"
run_dir="${repo_dir}/artifacts/textvqa-train-factorized-v2/risk-calibration/qwen3b-c4-seed0"
feature_dir="${repo_dir}/artifacts/textvqa-train-factorized-v2/risk-calibration/attention-semantic-v1"
expected_states=4747
scientific_status="fresh factorized TextVQA fixed-sequence calibration sibling bank; outcomes may calibrate the sole frozen candidate only"

check_sha256() {
  local path=$1
  local expected=$2
  local actual
  actual=$(sha256sum "${path}")
  actual=${actual%% *}
  if [[ "${actual}" != "${expected}" ]]; then
    echo "SHA-256 mismatch: ${path}" >&2
    exit 2
  fi
}

if [[ ! -r "${mail_file}" ]]; then
  echo "Missing private Slurm email file: ${mail_file}" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != *@* ]]; then
  echo "Invalid notification email in ${mail_file}" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "Tracked worktree must be clean before fresh calibration submission" >&2
  exit 2
fi

check_sha256 "${candidate}" 9a6c9d032ebdbc271b7d3c829fbb3d6ff167cac01b54ce75adc8da86e3063342
check_sha256 "${candidate_audit}" 63d8040e25701a6dc4e2f2841d4e10c2b688ccb1a0f23e65f15ea6450eb5d294
check_sha256 "${protocol}" babf01d4090263d1cfcb28c42f86f7b13ae9de4bb6bab0ca10d6e4707f02e2ca
check_sha256 "${allocation}" bc0ecb4b6f49a5b0e92b90b4c30620f72246722370d59c8078753d5846f5e9b6
check_sha256 "${allocation_audit}" f01f853a7de7774466be55c012b7e174f57f4ac120ed58a0bf3984e71252b5c3
check_sha256 "${manifest}" 0db79580d7bb96794901703a6ec0bfc0ae14e31159ddde5664762aa0351b323a
check_sha256 "${manifest_provenance}" 3cf60f8474c10bc81b83b5cf47ef22224b010154b0933c2ffb00bec7225e0c45
if [[ "$(wc -l < "${manifest}")" -ne "${expected_states}" ]]; then
  echo "Fresh calibration manifest count mismatch" >&2
  exit 2
fi
if [[ -e "${run_dir}/rollouts.jsonl" || -e "${feature_dir}/features-question-region-attention-label-free.pt" ]]; then
  echo "Refusing to reuse existing fresh calibration model outcomes" >&2
  exit 2
fi

code_revision=$(git -C "${repo_dir}" rev-parse HEAD)
rollout_submission=$(
  sbatch \
    --job-name=be-tvqa-fv2-cal-rollout \
    --mail-user="${notify_email}" \
    --mail-type=ALL \
    --export="ALL,BE_SCALE_ROLE=factorized-v2-risk-calibration,BE_SCALE_MANIFEST=${manifest},BE_SCALE_MANIFEST_SHA256=0db79580d7bb96794901703a6ec0bfc0ae14e31159ddde5664762aa0351b323a,BE_SCALE_EXPECTED_STATES=${expected_states},BE_SCALE_RUN_DIR=${run_dir},BE_SCALE_SCIENTIFIC_STATUS=${scientific_status},BE_CODE_REVISION=${code_revision}" \
    "${repo_dir}/scripts/slurm_textvqa_train_scale_rollout.sh"
)
rollout_job_id=${rollout_submission##* }
if [[ ! "${rollout_job_id}" =~ ^[0-9]+$ ]]; then
  echo "Could not parse rollout job ID: ${rollout_submission}" >&2
  exit 2
fi
printf '%s\n' "${rollout_submission}"

feature_submission=$(
  sbatch \
    --dependency="afterok:${rollout_job_id}" \
    --job-name=be-tvqa-fv2-cal-features \
    --mail-user="${notify_email}" \
    --mail-type=ALL \
    --export="ALL,BE_SCALE_ROLE=factorized-v2-risk-calibration,BE_SCALE_MANIFEST=${manifest},BE_SCALE_MANIFEST_SHA256=0db79580d7bb96794901703a6ec0bfc0ae14e31159ddde5664762aa0351b323a,BE_SCALE_EXPECTED_STATES=${expected_states},BE_SCALE_ROLLOUTS=${run_dir}/rollouts.jsonl,BE_SCALE_SCIENTIFIC_STATUS=${scientific_status},BE_SCALE_FEATURE_DIR=${feature_dir}" \
    "${repo_dir}/scripts/slurm_textvqa_train_scale_features.sh"
)
feature_job_id=${feature_submission##* }
if [[ ! "${feature_job_id}" =~ ^[0-9]+$ ]]; then
  echo "Could not parse feature job ID: ${feature_submission}" >&2
  exit 2
fi
printf '%s\n' "${feature_submission}"
printf 'rollout_job_id=%s feature_job_id=%s\n' "${rollout_job_id}" "${feature_job_id}"
