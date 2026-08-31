#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
manifest_dir="${repo_dir}/artifacts/screenqa-train-factorized-v1/formal-manifest-v1"
candidate_dir="${repo_dir}/artifacts/screenqa-train-factorized-v1/ranker-training/low-capacity-oof-v1/candidate-v1"
calibration_dir="${repo_dir}/artifacts/screenqa-train-factorized-v1/calibration-v1"
run_root="${repo_dir}/artifacts/screenqa-train-factorized-v1/formal-rollouts-v1"
evaluation_dir="${repo_dir}/artifacts/screenqa-train-factorized-v1/formal-evaluation-v1"
paper_analysis_dir="${repo_dir}/artifacts/screenqa-train-factorized-v1/formal-paper-analysis-v1"
paper_protocol="${repo_dir}/artifacts/screenqa-train-factorized-v1/ops/formal-paper-analysis-protocol-v1.md"
paper_protocol_sha256=ddd75cb2a591001065e4c79217119d901620daa0fd73287948b3c710173a8e66
reserve_dir="${repo_dir}/artifacts/screenqa-train-factorized-v1/reserve-manifest-v1"
untouched_dir="${repo_dir}/artifacts/screenqa-train-factorized-v1/untouched-manifest-v1"
expected_states=14672
shard_count=4
python_bin=/userhome/cs3/yihangc/anaconda3/bin/python

if [[ ! -r "${mail_file}" ]]; then
  echo "missing private Slurm email file" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != "yihangc@connect.hku.hk" ]]; then
  echo "invalid ScreenQA notification email" >&2
  exit 2
fi
for required_file in \
  "${manifest_dir}/manifest.jsonl" \
  "${manifest_dir}/manifest.audit.json" \
  "${manifest_dir}/SHA256SUMS" \
  "${candidate_dir}/model.json" \
  "${candidate_dir}/candidate.audit.json" \
  "${candidate_dir}/SHA256SUMS" \
  "${calibration_dir}/calibration.json" \
  "${calibration_dir}/model.json" \
  "${calibration_dir}/calibration.audit.json" \
  "${calibration_dir}/SHA256SUMS"; do
  if [[ ! -s "${required_file}" ]]; then
    echo "ScreenQA formal-bank input is incomplete: ${required_file}" >&2
    exit 2
  fi
done
if [[ -e "${run_root}" ]]; then
  echo "refusing to reuse existing ScreenQA one-shot formal outcomes: ${run_root}" >&2
  exit 2
fi
if [[ -e "${evaluation_dir}" ]]; then
  echo "refusing to reuse existing ScreenQA one-shot formal evaluation: ${evaluation_dir}" >&2
  exit 2
fi
if [[ -e "${paper_analysis_dir}" ]]; then
  echo "refusing to reuse existing ScreenQA formal paper analysis: ${paper_analysis_dir}" >&2
  exit 2
fi
actual_paper_protocol_sha256=$(sha256sum "${paper_protocol}")
actual_paper_protocol_sha256=${actual_paper_protocol_sha256%% *}
if [[ "${actual_paper_protocol_sha256}" != "${paper_protocol_sha256}" ]]; then
  echo "ScreenQA formal paper-analysis protocol SHA-256 mismatch" >&2
  exit 2
fi
for sealed_dir in "${reserve_dir}" "${untouched_dir}"; do
  if [[ -e "${sealed_dir}" ]] && [[ ! -d "${sealed_dir}" || -n "$(find "${sealed_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "sealed ScreenQA role is already materialized: ${sealed_dir}" >&2
    exit 2
  fi
done
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before ScreenQA formal-bank submission" >&2
  exit 2
fi

manifest_sha256=$(sha256sum "${manifest_dir}/manifest.jsonl")
manifest_sha256=${manifest_sha256%% *}
manifest_audit_sha256=$(sha256sum "${manifest_dir}/manifest.audit.json")
manifest_audit_sha256=${manifest_audit_sha256%% *}
candidate_bundle_sha256=$(sha256sum "${candidate_dir}/SHA256SUMS")
candidate_bundle_sha256=${candidate_bundle_sha256%% *}
calibration_bundle_sha256=$(sha256sum "${calibration_dir}/SHA256SUMS")
calibration_bundle_sha256=${calibration_bundle_sha256%% *}
code_revision=$(git -C "${repo_dir}" rev-parse HEAD)

cd "${repo_dir}"
PYTHONPATH="${repo_dir}/src" "${python_bin}" -m \
  scripts.verify_screenqa_formal_manifest \
  --manifest-dir "${manifest_dir}" \
  --candidate-dir "${candidate_dir}" \
  --expected-candidate-bundle-sha256 "${candidate_bundle_sha256}" \
  --calibration-dir "${calibration_dir}" \
  --expected-calibration-bundle-sha256 "${calibration_bundle_sha256}" \
  --expected-manifest-sha256 "${manifest_sha256}" \
  --expected-audit-sha256 "${manifest_audit_sha256}"

export_args="ALL,BE_SCREENQA_FORMAL_MANIFEST_DIR=${manifest_dir},BE_SCREENQA_FORMAL_MANIFEST_SHA256=${manifest_sha256},BE_SCREENQA_FORMAL_MANIFEST_AUDIT_SHA256=${manifest_audit_sha256},BE_SCREENQA_CANDIDATE_DIR=${candidate_dir},BE_SCREENQA_CANDIDATE_BUNDLE_SHA256=${candidate_bundle_sha256},BE_SCREENQA_CALIBRATION_DIR=${calibration_dir},BE_SCREENQA_CALIBRATION_BUNDLE_SHA256=${calibration_bundle_sha256},BE_SCREENQA_FORMAL_EXPECTED_STATES=${expected_states},BE_SCREENQA_FORMAL_RUN_ROOT=${run_root},BE_SCREENQA_FORMAL_EVALUATION_DIR=${evaluation_dir},BE_SCREENQA_FORMAL_PAPER_ANALYSIS_DIR=${paper_analysis_dir},BE_SCREENQA_FORMAL_PAPER_PROTOCOL=${paper_protocol},BE_SCREENQA_FORMAL_PAPER_PROTOCOL_SHA256=${paper_protocol_sha256},BE_SCREENQA_EXPECTED_CODE_REVISION=${code_revision},BE_SCREENQA_SHARD_COUNT=${shard_count}"

array_submission=$(
  sbatch \
    --mail-user="${notify_email}" \
    --mail-type=ALL \
    --export="${export_args}" \
    "${repo_dir}/scripts/slurm_screenqa_formal_bank.sh"
)
array_job_id=${array_submission##* }
if [[ ! "${array_job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse ScreenQA formal array job ID: ${array_submission}" >&2
  exit 2
fi
printf '%s\n' "${array_submission}"
printf 'screenqa_formal_array_job_id=%s code_revision=%s calibration_bundle_sha256=%s\n' \
  "${array_job_id}" "${code_revision}" "${calibration_bundle_sha256}"

merge_submission=$(
  sbatch \
    --dependency="afterok:${array_job_id}" \
    --mail-user="${notify_email}" \
    --mail-type=ALL \
    --export="${export_args}" \
    "${repo_dir}/scripts/slurm_screenqa_formal_bank_merge.sh"
)
merge_job_id=${merge_submission##* }
if [[ ! "${merge_job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse ScreenQA formal merge job ID: ${merge_submission}" >&2
  exit 2
fi
printf '%s\n' "${merge_submission}"
printf 'screenqa_formal_merge_job_id=%s code_revision=%s manifest_sha256=%s\n' \
  "${merge_job_id}" "${code_revision}" "${manifest_sha256}"

evaluation_submission=$(
  sbatch \
    --dependency="afterok:${merge_job_id}" \
    --mail-user="${notify_email}" \
    --mail-type=ALL \
    --export="${export_args}" \
    "${repo_dir}/scripts/slurm_screenqa_formal_evaluate.sh"
)
evaluation_job_id=${evaluation_submission##* }
if [[ ! "${evaluation_job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse ScreenQA formal evaluation job ID: ${evaluation_submission}" >&2
  exit 2
fi
printf '%s\n' "${evaluation_submission}"
printf 'screenqa_formal_evaluation_job_id=%s code_revision=%s calibration_bundle_sha256=%s\n' \
  "${evaluation_job_id}" "${code_revision}" "${calibration_bundle_sha256}"

analysis_submission=$(
  sbatch \
    --dependency="afterok:${evaluation_job_id}" \
    --mail-user="${notify_email}" \
    --mail-type=ALL \
    --export="${export_args}" \
    "${repo_dir}/scripts/slurm_screenqa_formal_paper_analysis.sh"
)
analysis_job_id=${analysis_submission##* }
if [[ ! "${analysis_job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse ScreenQA formal paper-analysis job ID: ${analysis_submission}" >&2
  exit 2
fi
printf '%s\n' "${analysis_submission}"
printf 'screenqa_formal_paper_analysis_job_id=%s code_revision=%s\n' \
  "${analysis_job_id}" "${code_revision}"
