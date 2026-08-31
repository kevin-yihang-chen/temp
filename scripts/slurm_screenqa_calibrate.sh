#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=be-screenqa-calibrate
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-screenqa-calibrate-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

: "${BE_SCREENQA_CANDIDATE_DIR:?missing BE_SCREENQA_CANDIDATE_DIR}"
: "${BE_SCREENQA_CANDIDATE_BUNDLE_SHA256:?missing BE_SCREENQA_CANDIDATE_BUNDLE_SHA256}"
: "${BE_SCREENQA_CALIBRATION_MANIFEST_DIR:?missing BE_SCREENQA_CALIBRATION_MANIFEST_DIR}"
: "${BE_SCREENQA_CALIBRATION_MANIFEST_SHA256:?missing BE_SCREENQA_CALIBRATION_MANIFEST_SHA256}"
: "${BE_SCREENQA_CALIBRATION_MANIFEST_AUDIT_SHA256:?missing BE_SCREENQA_CALIBRATION_MANIFEST_AUDIT_SHA256}"
: "${BE_SCREENQA_CALIBRATION_ROLLOUTS:?missing BE_SCREENQA_CALIBRATION_ROLLOUTS}"
: "${BE_SCREENQA_CALIBRATION_ROLLOUTS_SHA256:?missing BE_SCREENQA_CALIBRATION_ROLLOUTS_SHA256}"
: "${BE_SCREENQA_CALIBRATION_MERGE_AUDIT:?missing BE_SCREENQA_CALIBRATION_MERGE_AUDIT}"
: "${BE_SCREENQA_CALIBRATION_MERGE_AUDIT_SHA256:?missing BE_SCREENQA_CALIBRATION_MERGE_AUDIT_SHA256}"
: "${BE_SCREENQA_CALIBRATION_BANK_CODE_REVISION:?missing BE_SCREENQA_CALIBRATION_BANK_CODE_REVISION}"
: "${BE_SCREENQA_CALIBRATION_OUTPUT_DIR:?missing BE_SCREENQA_CALIBRATION_OUTPUT_DIR}"
: "${BE_SCREENQA_FORMAL_OUTPUT_DIR:?missing BE_SCREENQA_FORMAL_OUTPUT_DIR}"
: "${BE_SCREENQA_RESERVE_OUTPUT_DIR:?missing BE_SCREENQA_RESERVE_OUTPUT_DIR}"
: "${BE_SCREENQA_UNTOUCHED_OUTPUT_DIR:?missing BE_SCREENQA_UNTOUCHED_OUTPUT_DIR}"
: "${BE_SCREENQA_EXPECTED_CODE_REVISION:?missing BE_SCREENQA_EXPECTED_CODE_REVISION}"

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/bin/python

tracked_status=$(git -C "${repo_dir}" status --porcelain --untracked-files=no)
if [[ -n "${tracked_status}" ]]; then
  echo "Tracked worktree must be clean before ScreenQA risk calibration" >&2
  exit 2
fi
actual_code_revision=$(git -C "${repo_dir}" rev-parse HEAD)
if [[ "${actual_code_revision}" != "${BE_SCREENQA_EXPECTED_CODE_REVISION}" ]]; then
  echo "ScreenQA risk-calibration code revision mismatch" >&2
  exit 2
fi
actual_candidate_bundle_sha256=$(sha256sum "${BE_SCREENQA_CANDIDATE_DIR}/SHA256SUMS")
actual_candidate_bundle_sha256=${actual_candidate_bundle_sha256%% *}
if [[ "${actual_candidate_bundle_sha256}" != "${BE_SCREENQA_CANDIDATE_BUNDLE_SHA256}" ]]; then
  echo "ScreenQA risk-calibration candidate bundle mismatch" >&2
  exit 2
fi
if [[ -e "${BE_SCREENQA_CALIBRATION_OUTPUT_DIR}" ]]; then
  echo "ScreenQA risk-calibration output already exists" >&2
  exit 2
fi

export PYTHONPATH="${repo_dir}/src"
export BE_CODE_REVISION="${actual_code_revision}"
cd "${repo_dir}"
"${python_bin}" -m scripts.calibrate_screenqa_fixed_sequence \
  --candidate-dir "${BE_SCREENQA_CANDIDATE_DIR}" \
  --manifest-dir "${BE_SCREENQA_CALIBRATION_MANIFEST_DIR}" \
  --expected-manifest-sha256 "${BE_SCREENQA_CALIBRATION_MANIFEST_SHA256}" \
  --expected-manifest-audit-sha256 "${BE_SCREENQA_CALIBRATION_MANIFEST_AUDIT_SHA256}" \
  --rollouts "${BE_SCREENQA_CALIBRATION_ROLLOUTS}" \
  --expected-rollouts-sha256 "${BE_SCREENQA_CALIBRATION_ROLLOUTS_SHA256}" \
  --merge-audit "${BE_SCREENQA_CALIBRATION_MERGE_AUDIT}" \
  --expected-merge-audit-sha256 "${BE_SCREENQA_CALIBRATION_MERGE_AUDIT_SHA256}" \
  --expected-bank-code-revision "${BE_SCREENQA_CALIBRATION_BANK_CODE_REVISION}" \
  --formal-output-dir "${BE_SCREENQA_FORMAL_OUTPUT_DIR}" \
  --reserve-output-dir "${BE_SCREENQA_RESERVE_OUTPUT_DIR}" \
  --untouched-output-dir "${BE_SCREENQA_UNTOUCHED_OUTPUT_DIR}" \
  --output-dir "${BE_SCREENQA_CALIBRATION_OUTPUT_DIR}"

cd "${BE_SCREENQA_CALIBRATION_OUTPUT_DIR}"
sha256sum --check SHA256SUMS
cd "${repo_dir}"
"${python_bin}" -m scripts.verify_screenqa_calibration_result \
  --output-dir "${BE_SCREENQA_CALIBRATION_OUTPUT_DIR}" \
  --candidate-dir "${BE_SCREENQA_CANDIDATE_DIR}"
