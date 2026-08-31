#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
candidate_dir="${repo_dir}/artifacts/screenqa-train-factorized-v1/ranker-training/low-capacity-oof-v1/candidate-v1"
manifest_dir="${repo_dir}/artifacts/screenqa-train-factorized-v1/calibration-manifest-v1"
bank_dir="${repo_dir}/artifacts/screenqa-train-factorized-v1/calibration-rollouts-v1/merged"
rollouts="${bank_dir}/rollouts.jsonl"
merge_audit="${bank_dir}/rollouts.merge.json"
output_dir="${repo_dir}/artifacts/screenqa-train-factorized-v1/calibration-v1"
formal_dir="${repo_dir}/artifacts/screenqa-train-factorized-v1/formal-manifest-v1"
reserve_dir="${repo_dir}/artifacts/screenqa-train-factorized-v1/reserve-manifest-v1"
untouched_dir="${repo_dir}/artifacts/screenqa-train-factorized-v1/untouched-manifest-v1"
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
  "${candidate_dir}/model.json" \
  "${candidate_dir}/candidate.audit.json" \
  "${candidate_dir}/SHA256SUMS" \
  "${manifest_dir}/manifest.jsonl" \
  "${manifest_dir}/manifest.audit.json" \
  "${manifest_dir}/SHA256SUMS" \
  "${rollouts}" \
  "${merge_audit}" \
  "${bank_dir}/SHA256SUMS"; do
  if [[ ! -s "${required_file}" ]]; then
    echo "ScreenQA risk-calibration input is incomplete: ${required_file}" >&2
    exit 2
  fi
done
if [[ -e "${output_dir}" ]]; then
  echo "refusing to reuse ScreenQA risk-calibration output: ${output_dir}" >&2
  exit 2
fi
for sealed_dir in "${formal_dir}" "${reserve_dir}" "${untouched_dir}"; do
  if [[ -e "${sealed_dir}" ]] && [[ ! -d "${sealed_dir}" || -n "$(find "${sealed_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "sealed ScreenQA role is already materialized: ${sealed_dir}" >&2
    exit 2
  fi
done
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before ScreenQA risk-calibration submission" >&2
  exit 2
fi

cd "${candidate_dir}"
sha256sum --check SHA256SUMS
cd "${manifest_dir}"
sha256sum --check SHA256SUMS
cd "${bank_dir}"
sha256sum --check SHA256SUMS
cd "${repo_dir}"

candidate_bundle_sha256=$(sha256sum "${candidate_dir}/SHA256SUMS")
candidate_bundle_sha256=${candidate_bundle_sha256%% *}
manifest_sha256=$(sha256sum "${manifest_dir}/manifest.jsonl")
manifest_sha256=${manifest_sha256%% *}
manifest_audit_sha256=$(sha256sum "${manifest_dir}/manifest.audit.json")
manifest_audit_sha256=${manifest_audit_sha256%% *}
rollouts_sha256=$(sha256sum "${rollouts}")
rollouts_sha256=${rollouts_sha256%% *}
merge_audit_sha256=$(sha256sum "${merge_audit}")
merge_audit_sha256=${merge_audit_sha256%% *}
bank_code_revision=$(
  "${python_bin}" - "${merge_audit}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload["invariant_provenance"]["code_revision"])
PY
)
code_revision=$(git -C "${repo_dir}" rev-parse HEAD)

PYTHONPATH="${repo_dir}/src" "${python_bin}" -m \
  scripts.verify_screenqa_calibration_manifest \
  --manifest-dir "${manifest_dir}" \
  --candidate-dir "${candidate_dir}" \
  --expected-candidate-bundle-sha256 "${candidate_bundle_sha256}" \
  --expected-manifest-sha256 "${manifest_sha256}" \
  --expected-audit-sha256 "${manifest_audit_sha256}"

export_args="ALL,BE_SCREENQA_CANDIDATE_DIR=${candidate_dir},BE_SCREENQA_CANDIDATE_BUNDLE_SHA256=${candidate_bundle_sha256},BE_SCREENQA_CALIBRATION_MANIFEST_DIR=${manifest_dir},BE_SCREENQA_CALIBRATION_MANIFEST_SHA256=${manifest_sha256},BE_SCREENQA_CALIBRATION_MANIFEST_AUDIT_SHA256=${manifest_audit_sha256},BE_SCREENQA_CALIBRATION_ROLLOUTS=${rollouts},BE_SCREENQA_CALIBRATION_ROLLOUTS_SHA256=${rollouts_sha256},BE_SCREENQA_CALIBRATION_MERGE_AUDIT=${merge_audit},BE_SCREENQA_CALIBRATION_MERGE_AUDIT_SHA256=${merge_audit_sha256},BE_SCREENQA_CALIBRATION_BANK_CODE_REVISION=${bank_code_revision},BE_SCREENQA_CALIBRATION_OUTPUT_DIR=${output_dir},BE_SCREENQA_FORMAL_OUTPUT_DIR=${formal_dir},BE_SCREENQA_RESERVE_OUTPUT_DIR=${reserve_dir},BE_SCREENQA_UNTOUCHED_OUTPUT_DIR=${untouched_dir},BE_SCREENQA_EXPECTED_CODE_REVISION=${code_revision}"

submission=$(
  sbatch \
    --mail-user="${notify_email}" \
    --mail-type=ALL \
    --export="${export_args}" \
    "${repo_dir}/scripts/slurm_screenqa_calibrate.sh"
)
job_id=${submission##* }
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse ScreenQA calibration job ID: ${submission}" >&2
  exit 2
fi
printf '%s\n' "${submission}"
printf 'screenqa_calibration_job_id=%s code_revision=%s bank_code_revision=%s\n' \
  "${job_id}" "${code_revision}" "${bank_code_revision}"
