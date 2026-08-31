#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
candidate_dir="${repo_dir}/artifacts/screenqa-train-factorized-v1/ranker-training/low-capacity-oof-v1/candidate-v1"
calibration_dir="${repo_dir}/artifacts/screenqa-train-factorized-v1/calibration-v1"
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
  "${calibration_dir}/calibration.json" \
  "${calibration_dir}/model.json" \
  "${calibration_dir}/calibration.audit.json" \
  "${calibration_dir}/SHA256SUMS"; do
  if [[ ! -s "${required_file}" ]]; then
    echo "ScreenQA formal-export input is incomplete: ${required_file}" >&2
    exit 2
  fi
done
if [[ -e "${formal_dir}" ]]; then
  echo "refusing to reuse ScreenQA formal manifest: ${formal_dir}" >&2
  exit 2
fi
for sealed_dir in "${reserve_dir}" "${untouched_dir}"; do
  if [[ -e "${sealed_dir}" ]] && [[ ! -d "${sealed_dir}" || -n "$(find "${sealed_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "sealed ScreenQA role is already materialized: ${sealed_dir}" >&2
    exit 2
  fi
done
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before ScreenQA formal submission" >&2
  exit 2
fi

cd "${candidate_dir}"
sha256sum --check SHA256SUMS
cd "${calibration_dir}"
sha256sum --check SHA256SUMS
cd "${repo_dir}"
PYTHONPATH="${repo_dir}/src" "${python_bin}" - "${candidate_dir}" "${calibration_dir}" <<'PY'
import json
import sys
from pathlib import Path

from scripts.export_screenqa_formal_manifest import verify_formal_gate

result = verify_formal_gate(Path(sys.argv[1]), Path(sys.argv[2]))
print(json.dumps(result, indent=2, sort_keys=True))
PY

candidate_bundle_sha256=$(sha256sum "${candidate_dir}/SHA256SUMS")
candidate_bundle_sha256=${candidate_bundle_sha256%% *}
calibration_bundle_sha256=$(sha256sum "${calibration_dir}/SHA256SUMS")
calibration_bundle_sha256=${calibration_bundle_sha256%% *}
code_revision=$(git -C "${repo_dir}" rev-parse HEAD)
export_args="ALL,BE_SCREENQA_CANDIDATE_DIR=${candidate_dir},BE_SCREENQA_CANDIDATE_BUNDLE_SHA256=${candidate_bundle_sha256},BE_SCREENQA_CALIBRATION_DIR=${calibration_dir},BE_SCREENQA_CALIBRATION_BUNDLE_SHA256=${calibration_bundle_sha256},BE_SCREENQA_FORMAL_MANIFEST_DIR=${formal_dir},BE_SCREENQA_RESERVE_MANIFEST_DIR=${reserve_dir},BE_SCREENQA_UNTOUCHED_MANIFEST_DIR=${untouched_dir},BE_SCREENQA_EXPECTED_CODE_REVISION=${code_revision}"

submission=$(
  sbatch \
    --mail-user="${notify_email}" \
    --mail-type=ALL \
    --export="${export_args}" \
    "${repo_dir}/scripts/slurm_screenqa_formal_manifest.sh"
)
job_id=${submission##* }
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse ScreenQA formal-export job ID: ${submission}" >&2
  exit 2
fi
printf '%s\n' "${submission}"
printf 'screenqa_formal_export_job_id=%s code_revision=%s calibration_bundle_sha256=%s\n' \
  "${job_id}" "${code_revision}" "${calibration_bundle_sha256}"
