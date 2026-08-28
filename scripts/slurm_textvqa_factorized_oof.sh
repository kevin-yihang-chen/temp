#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-textvqa-factorized-oof-%x-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

: "${BE_FACTOR_FEATURE_MODE:?missing BE_FACTOR_FEATURE_MODE}"
: "${BE_FACTOR_OUTPUT_DIR:?missing BE_FACTOR_OUTPUT_DIR}"

case "${BE_FACTOR_FEATURE_MODE}" in
  semantic-context|hybrid-context-semantic) ;;
  *)
    echo "Unsupported factorized feature mode: ${BE_FACTOR_FEATURE_MODE}" >&2
    exit 2
    ;;
esac

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
rollouts="${repo_dir}/artifacts/textvqa-train-scale-v1/ranker-training/qwen3b-c4-seed0/rollouts.jsonl"
features="${repo_dir}/artifacts/textvqa-train-scale-v1/ranker-training/attention-semantic-v1/features-question-region-attention-label-free.pt"
feature_audit="${repo_dir}/artifacts/textvqa-train-scale-v1/ranker-training/attention-semantic-v1/label-free-audit.json"
expected_rollouts_sha256=1c1d5b67010b5ddfbdabe47072291336b34dcc54928e5db7a12727daa4f14c8e
expected_features_sha256=93cdfa91b570fcc67f16bdd4e39d59489fa160e26c2797abf16d684f2f44a504

export PYTHONPATH="${repo_dir}/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${repo_dir}"
tracked_status=$(git status --porcelain --untracked-files=no)
if [[ -n "${tracked_status}" ]]; then
  echo "Tracked worktree must be clean before factorized OOF fitting" >&2
  exit 2
fi
export BE_CODE_REVISION
BE_CODE_REVISION=$(git rev-parse HEAD)

test -s "${feature_audit}"
actual_rollouts_sha256=$(sha256sum "${rollouts}")
actual_rollouts_sha256=${actual_rollouts_sha256%% *}
actual_features_sha256=$(sha256sum "${features}")
actual_features_sha256=${actual_features_sha256%% *}
if [[ "${actual_rollouts_sha256}" != "${expected_rollouts_sha256}" ]]; then
  echo "Ranker rollout SHA-256 mismatch" >&2
  exit 2
fi
if [[ "${actual_features_sha256}" != "${expected_features_sha256}" ]]; then
  echo "Ranker feature SHA-256 mismatch" >&2
  exit 2
fi

"${python_bin}" scripts/fit_multidomain_action_value.py \
  --domain "textvqa=${rollouts}" \
  --features "textvqa=${features}" \
  --output-dir "${BE_FACTOR_OUTPUT_DIR}" \
  --feature-mode "${BE_FACTOR_FEATURE_MODE}" \
  --model-family factorized-oof \
  --oof-folds 5 \
  --bootstrap-resamples 2000 \
  --lambda-cost 0.05
