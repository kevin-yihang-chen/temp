#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-docvqa-train-factorized-v2-fit-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

: "${BE_DOCVQA_RANKER_ROLLOUTS:?missing BE_DOCVQA_RANKER_ROLLOUTS}"
: "${BE_DOCVQA_RANKER_ROLLOUTS_SHA256:?missing BE_DOCVQA_RANKER_ROLLOUTS_SHA256}"
: "${BE_DOCVQA_RANKER_FEATURES:?missing BE_DOCVQA_RANKER_FEATURES}"
: "${BE_DOCVQA_RANKER_FEATURES_SHA256:?missing BE_DOCVQA_RANKER_FEATURES_SHA256}"
: "${BE_DOCVQA_RANKER_ROLLOUT_AUDIT:?missing BE_DOCVQA_RANKER_ROLLOUT_AUDIT}"
: "${BE_DOCVQA_RANKER_ROLLOUT_AUDIT_SHA256:?missing BE_DOCVQA_RANKER_ROLLOUT_AUDIT_SHA256}"
: "${BE_DOCVQA_RANKER_LABEL_FREE_AUDIT:?missing BE_DOCVQA_RANKER_LABEL_FREE_AUDIT}"
: "${BE_DOCVQA_RANKER_LABEL_FREE_AUDIT_SHA256:?missing BE_DOCVQA_RANKER_LABEL_FREE_AUDIT_SHA256}"
: "${BE_DOCVQA_EXPECTED_CODE_REVISION:?missing BE_DOCVQA_EXPECTED_CODE_REVISION}"
: "${BE_DOCVQA_FIT_OUTPUT_DIR:?missing BE_DOCVQA_FIT_OUTPUT_DIR}"

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python

tracked_status=$(git -C "${repo_dir}" status --porcelain --untracked-files=no)
if [[ -n "${tracked_status}" ]]; then
  echo "Tracked worktree must be clean before DocVQA candidate fitting" >&2
  exit 2
fi
actual_code_revision=$(git -C "${repo_dir}" rev-parse HEAD)
if [[ "${actual_code_revision}" != "${BE_DOCVQA_EXPECTED_CODE_REVISION}" ]]; then
  echo "DocVQA fitting code revision mismatch" >&2
  exit 2
fi
actual_rollouts_sha256=$(sha256sum "${BE_DOCVQA_RANKER_ROLLOUTS}")
actual_rollouts_sha256=${actual_rollouts_sha256%% *}
if [[ "${actual_rollouts_sha256}" != "${BE_DOCVQA_RANKER_ROLLOUTS_SHA256}" ]]; then
  echo "DocVQA ranker rollout SHA-256 mismatch" >&2
  exit 2
fi
actual_features_sha256=$(sha256sum "${BE_DOCVQA_RANKER_FEATURES}")
actual_features_sha256=${actual_features_sha256%% *}
if [[ "${actual_features_sha256}" != "${BE_DOCVQA_RANKER_FEATURES_SHA256}" ]]; then
  echo "DocVQA ranker feature SHA-256 mismatch" >&2
  exit 2
fi
test -s "${BE_DOCVQA_RANKER_ROLLOUT_AUDIT}"
test -s "${BE_DOCVQA_RANKER_LABEL_FREE_AUDIT}"
actual_rollout_audit_sha256=$(sha256sum "${BE_DOCVQA_RANKER_ROLLOUT_AUDIT}")
actual_rollout_audit_sha256=${actual_rollout_audit_sha256%% *}
if [[ "${actual_rollout_audit_sha256}" != "${BE_DOCVQA_RANKER_ROLLOUT_AUDIT_SHA256}" ]]; then
  echo "DocVQA ranker rollout audit SHA-256 mismatch" >&2
  exit 2
fi
actual_label_free_audit_sha256=$(sha256sum "${BE_DOCVQA_RANKER_LABEL_FREE_AUDIT}")
actual_label_free_audit_sha256=${actual_label_free_audit_sha256%% *}
if [[ "${actual_label_free_audit_sha256}" != "${BE_DOCVQA_RANKER_LABEL_FREE_AUDIT_SHA256}" ]]; then
  echo "DocVQA label-free audit SHA-256 mismatch" >&2
  exit 2
fi
if [[ -e "${BE_DOCVQA_FIT_OUTPUT_DIR}" ]]; then
  echo "DocVQA candidate output must not exist" >&2
  exit 2
fi

export BE_CODE_REVISION="${actual_code_revision}"
export PYTHONPATH="${repo_dir}/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${repo_dir}"
"${python_bin}" scripts/fit_multidomain_action_value.py \
  --domain "docvqa=${BE_DOCVQA_RANKER_ROLLOUTS}" \
  --features "docvqa=${BE_DOCVQA_RANKER_FEATURES}" \
  --output-dir "${BE_DOCVQA_FIT_OUTPUT_DIR}" \
  --feature-mode hybrid-context-semantic \
  --model-family factorized-oof \
  --oof-folds 5 \
  --bootstrap-resamples 2000 \
  --lambda-cost 0.05 \
  --alpha 1.0 \
  --seed 20260829
