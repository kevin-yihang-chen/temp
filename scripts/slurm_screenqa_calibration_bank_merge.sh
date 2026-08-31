#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=be-screenqa-cal-merge
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-screenqa-calibration-bank-merge-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

: "${BE_SCREENQA_CALIBRATION_MANIFEST_DIR:?missing BE_SCREENQA_CALIBRATION_MANIFEST_DIR}"
: "${BE_SCREENQA_CALIBRATION_MANIFEST_SHA256:?missing BE_SCREENQA_CALIBRATION_MANIFEST_SHA256}"
: "${BE_SCREENQA_CALIBRATION_MANIFEST_AUDIT_SHA256:?missing BE_SCREENQA_CALIBRATION_MANIFEST_AUDIT_SHA256}"
: "${BE_SCREENQA_CANDIDATE_DIR:?missing BE_SCREENQA_CANDIDATE_DIR}"
: "${BE_SCREENQA_CANDIDATE_BUNDLE_SHA256:?missing BE_SCREENQA_CANDIDATE_BUNDLE_SHA256}"
: "${BE_SCREENQA_CALIBRATION_EXPECTED_STATES:?missing BE_SCREENQA_CALIBRATION_EXPECTED_STATES}"
: "${BE_SCREENQA_CALIBRATION_RUN_ROOT:?missing BE_SCREENQA_CALIBRATION_RUN_ROOT}"
: "${BE_SCREENQA_EXPECTED_CODE_REVISION:?missing BE_SCREENQA_EXPECTED_CODE_REVISION}"
: "${BE_SCREENQA_SHARD_COUNT:?missing BE_SCREENQA_SHARD_COUNT}"

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
manifest="${BE_SCREENQA_CALIBRATION_MANIFEST_DIR}/manifest.jsonl"
output_dir="${BE_SCREENQA_CALIBRATION_RUN_ROOT}/merged"
output="${output_dir}/rollouts.jsonl"

tracked_status=$(git -C "${repo_dir}" status --porcelain --untracked-files=no)
if [[ -n "${tracked_status}" ]]; then
  echo "Tracked worktree must be clean before ScreenQA calibration merge" >&2
  exit 2
fi
actual_code_revision=$(git -C "${repo_dir}" rev-parse HEAD)
if [[ "${actual_code_revision}" != "${BE_SCREENQA_EXPECTED_CODE_REVISION}" ]]; then
  echo "ScreenQA calibration merge code revision mismatch" >&2
  exit 2
fi
if [[ "${BE_SCREENQA_CALIBRATION_EXPECTED_STATES}" -ne 9951 ]]; then
  echo "ScreenQA calibration merge is frozen to 9951 states" >&2
  exit 2
fi

export PYTHONPATH="${repo_dir}/src"
cd "${repo_dir}"
/userhome/cs3/yihangc/anaconda3/bin/python -m \
  scripts.verify_screenqa_calibration_manifest \
  --manifest-dir "${BE_SCREENQA_CALIBRATION_MANIFEST_DIR}" \
  --candidate-dir "${BE_SCREENQA_CANDIDATE_DIR}" \
  --expected-candidate-bundle-sha256 "${BE_SCREENQA_CANDIDATE_BUNDLE_SHA256}" \
  --expected-manifest-sha256 "${BE_SCREENQA_CALIBRATION_MANIFEST_SHA256}" \
  --expected-audit-sha256 "${BE_SCREENQA_CALIBRATION_MANIFEST_AUDIT_SHA256}"

/userhome/cs3/yihangc/anaconda3/bin/python \
  "${repo_dir}/scripts/merge_qwen_rollout_shards.py" \
  --manifest "${manifest}" \
  --expected-manifest-sha256 "${BE_SCREENQA_CALIBRATION_MANIFEST_SHA256}" \
  --run-root "${BE_SCREENQA_CALIBRATION_RUN_ROOT}" \
  --shard-count "${BE_SCREENQA_SHARD_COUNT}" \
  --expected-code-revision "${BE_SCREENQA_EXPECTED_CODE_REVISION}" \
  --expected-scorer screenqa \
  --require-resume-audit \
  --bootstrap-resamples 5000 \
  --bootstrap-seed 20260831 \
  --output "${output}"

cd "${output_dir}"
sha256sum rollouts.jsonl rollouts.diagnostic.json rollouts.merge.json > SHA256SUMS
sha256sum --check SHA256SUMS
