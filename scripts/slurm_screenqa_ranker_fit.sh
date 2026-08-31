#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=be-screenqa-fit
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-screenqa-ranker-fit-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

: "${BE_SCREENQA_MANIFEST_DIR:?missing BE_SCREENQA_MANIFEST_DIR}"
: "${BE_SCREENQA_MANIFEST_SHA256:?missing BE_SCREENQA_MANIFEST_SHA256}"
: "${BE_SCREENQA_MANIFEST_AUDIT_SHA256:?missing BE_SCREENQA_MANIFEST_AUDIT_SHA256}"
: "${BE_SCREENQA_RANKER_ROLLOUTS:?missing BE_SCREENQA_RANKER_ROLLOUTS}"
: "${BE_SCREENQA_RANKER_ROLLOUTS_SHA256:?missing BE_SCREENQA_RANKER_ROLLOUTS_SHA256}"
: "${BE_SCREENQA_MERGE_AUDIT:?missing BE_SCREENQA_MERGE_AUDIT}"
: "${BE_SCREENQA_MERGE_AUDIT_SHA256:?missing BE_SCREENQA_MERGE_AUDIT_SHA256}"
: "${BE_SCREENQA_BANK_CODE_REVISION:?missing BE_SCREENQA_BANK_CODE_REVISION}"
: "${BE_SCREENQA_PROTOCOL:?missing BE_SCREENQA_PROTOCOL}"
: "${BE_SCREENQA_PROTOCOL_SHA256:?missing BE_SCREENQA_PROTOCOL_SHA256}"
: "${BE_SCREENQA_FIT_ROOT:?missing BE_SCREENQA_FIT_ROOT}"
: "${BE_SCREENQA_EXPECTED_CODE_REVISION:?missing BE_SCREENQA_EXPECTED_CODE_REVISION}"

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
context_dir="${BE_SCREENQA_FIT_ROOT}/context-geometry-oof-v1"
spatial_dir="${BE_SCREENQA_FIT_ROOT}/spatial-context-geometry-oof-v1"
candidate_dir="${BE_SCREENQA_FIT_ROOT}/candidate-v1"
input_audit="${BE_SCREENQA_FIT_ROOT}/ranker-rollouts.audit.json"

tracked_status=$(git -C "${repo_dir}" status --porcelain --untracked-files=no)
if [[ -n "${tracked_status}" ]]; then
  echo "Tracked worktree must be clean before ScreenQA ranker fitting" >&2
  exit 2
fi
actual_code_revision=$(git -C "${repo_dir}" rev-parse HEAD)
if [[ "${actual_code_revision}" != "${BE_SCREENQA_EXPECTED_CODE_REVISION}" ]]; then
  echo "ScreenQA ranker fitting code revision mismatch" >&2
  exit 2
fi
if [[ -e "${BE_SCREENQA_FIT_ROOT}" ]]; then
  echo "ScreenQA fit root must not exist" >&2
  exit 2
fi

export BE_CODE_REVISION="${actual_code_revision}"
export PYTHONPATH="${repo_dir}/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${repo_dir}"
"${python_bin}" scripts/verify_screenqa_ranker_manifest.py \
  --manifest-dir "${BE_SCREENQA_MANIFEST_DIR}" \
  --expected-manifest-sha256 "${BE_SCREENQA_MANIFEST_SHA256}" \
  --expected-audit-sha256 "${BE_SCREENQA_MANIFEST_AUDIT_SHA256}" \
  --expected-states 14511
"${python_bin}" scripts/verify_screenqa_ranker_rollouts.py \
  --rollouts "${BE_SCREENQA_RANKER_ROLLOUTS}" \
  --expected-rollouts-sha256 "${BE_SCREENQA_RANKER_ROLLOUTS_SHA256}" \
  --merge-audit "${BE_SCREENQA_MERGE_AUDIT}" \
  --expected-merge-audit-sha256 "${BE_SCREENQA_MERGE_AUDIT_SHA256}" \
  --expected-manifest-sha256 "${BE_SCREENQA_MANIFEST_SHA256}" \
  --expected-bank-code-revision "${BE_SCREENQA_BANK_CODE_REVISION}" \
  --output "${input_audit}"

common_args=(
  --domain "screenqa=${BE_SCREENQA_RANKER_ROLLOUTS}"
  --model-family factorized-oof
  --oof-folds 5
  --bootstrap-resamples 2000
  --lambda-cost 0.05
  --alpha 0.1
  --alpha 1.0
  --alpha 10.0
  --alpha 100.0
  --alpha 1000.0
  --seed 20260831
)
"${python_bin}" scripts/fit_multidomain_action_value.py \
  "${common_args[@]}" \
  --feature-mode context-geometry \
  --output-dir "${context_dir}"
"${python_bin}" scripts/fit_multidomain_action_value.py \
  "${common_args[@]}" \
  --feature-mode spatial-context-geometry \
  --output-dir "${spatial_dir}"

"${python_bin}" scripts/select_screenqa_ranker_candidate.py \
  --context-report "${context_dir}/report.json" \
  --context-model "${context_dir}/model.json" \
  --spatial-report "${spatial_dir}/report.json" \
  --spatial-model "${spatial_dir}/model.json" \
  --protocol "${BE_SCREENQA_PROTOCOL}" \
  --expected-protocol-sha256 "${BE_SCREENQA_PROTOCOL_SHA256}" \
  --output-dir "${candidate_dir}"

cd "${candidate_dir}"
sha256sum --check SHA256SUMS
