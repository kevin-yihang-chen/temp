#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=be-screenqa-formal-export
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-screenqa-formal-export-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

: "${BE_SCREENQA_CANDIDATE_DIR:?missing BE_SCREENQA_CANDIDATE_DIR}"
: "${BE_SCREENQA_CANDIDATE_BUNDLE_SHA256:?missing BE_SCREENQA_CANDIDATE_BUNDLE_SHA256}"
: "${BE_SCREENQA_CALIBRATION_DIR:?missing BE_SCREENQA_CALIBRATION_DIR}"
: "${BE_SCREENQA_CALIBRATION_BUNDLE_SHA256:?missing BE_SCREENQA_CALIBRATION_BUNDLE_SHA256}"
: "${BE_SCREENQA_FORMAL_MANIFEST_DIR:?missing BE_SCREENQA_FORMAL_MANIFEST_DIR}"
: "${BE_SCREENQA_RESERVE_MANIFEST_DIR:?missing BE_SCREENQA_RESERVE_MANIFEST_DIR}"
: "${BE_SCREENQA_UNTOUCHED_MANIFEST_DIR:?missing BE_SCREENQA_UNTOUCHED_MANIFEST_DIR}"
: "${BE_SCREENQA_EXPECTED_CODE_REVISION:?missing BE_SCREENQA_EXPECTED_CODE_REVISION}"

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/bin/python
allocation_dir="${repo_dir}/artifacts/screenqa-train-factorized-v1/allocation-v1"
short_train=/userhome/cs3/yihangc/Data/screen_qa_annotations/short_answers/train.json
rico_images_dir=/userhome/cs3/yihangc/Data/rico-v0.1/extracted/combined

tracked_status=$(git -C "${repo_dir}" status --porcelain --untracked-files=no)
if [[ -n "${tracked_status}" ]]; then
  echo "Tracked worktree must be clean before ScreenQA formal export" >&2
  exit 2
fi
actual_code_revision=$(git -C "${repo_dir}" rev-parse HEAD)
if [[ "${actual_code_revision}" != "${BE_SCREENQA_EXPECTED_CODE_REVISION}" ]]; then
  echo "ScreenQA formal export code revision mismatch" >&2
  exit 2
fi
actual_candidate_bundle_sha256=$(sha256sum "${BE_SCREENQA_CANDIDATE_DIR}/SHA256SUMS")
actual_candidate_bundle_sha256=${actual_candidate_bundle_sha256%% *}
if [[ "${actual_candidate_bundle_sha256}" != "${BE_SCREENQA_CANDIDATE_BUNDLE_SHA256}" ]]; then
  echo "ScreenQA formal candidate bundle mismatch" >&2
  exit 2
fi
actual_calibration_bundle_sha256=$(sha256sum "${BE_SCREENQA_CALIBRATION_DIR}/SHA256SUMS")
actual_calibration_bundle_sha256=${actual_calibration_bundle_sha256%% *}
if [[ "${actual_calibration_bundle_sha256}" != "${BE_SCREENQA_CALIBRATION_BUNDLE_SHA256}" ]]; then
  echo "ScreenQA formal calibration bundle mismatch" >&2
  exit 2
fi
if [[ -e "${BE_SCREENQA_FORMAL_MANIFEST_DIR}" ]]; then
  echo "ScreenQA formal manifest output already exists" >&2
  exit 2
fi

export PYTHONPATH="${repo_dir}/src"
cd "${repo_dir}"
"${python_bin}" -m scripts.verify_screenqa_calibration_result \
  --output-dir "${BE_SCREENQA_CALIBRATION_DIR}" \
  --candidate-dir "${BE_SCREENQA_CANDIDATE_DIR}"
"${python_bin}" -m scripts.export_screenqa_formal_manifest \
  --allocation-dir "${allocation_dir}" \
  --short-train "${short_train}" \
  --rico-images-dir "${rico_images_dir}" \
  --candidate-dir "${BE_SCREENQA_CANDIDATE_DIR}" \
  --calibration-dir "${BE_SCREENQA_CALIBRATION_DIR}" \
  --reserve-output-dir "${BE_SCREENQA_RESERVE_MANIFEST_DIR}" \
  --untouched-output-dir "${BE_SCREENQA_UNTOUCHED_MANIFEST_DIR}" \
  --output-dir "${BE_SCREENQA_FORMAL_MANIFEST_DIR}"

cd "${BE_SCREENQA_FORMAL_MANIFEST_DIR}"
sha256sum --check SHA256SUMS
