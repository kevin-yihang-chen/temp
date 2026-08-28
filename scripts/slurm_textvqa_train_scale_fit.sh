#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-textvqa-train-scale-fit-%j.out
#SBATCH --mail-type=ALL

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
rollouts="${repo_dir}/artifacts/textvqa-train-scale-v1/ranker-training/qwen3b-c4-seed0/rollouts.jsonl"
feature_dir="${repo_dir}/artifacts/textvqa-train-scale-v1/ranker-training/attention-semantic-v1"
features="${feature_dir}/features-question-region-attention-label-free.pt"
output_dir="${repo_dir}/artifacts/textvqa-train-scale-v1/pairwise-attention-primary-v1/ranker-development"

export PYTHONPATH="${repo_dir}/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${repo_dir}"
tracked_status=$(git -C "${repo_dir}" status --porcelain --untracked-files=no)
if [[ -n "${tracked_status}" ]]; then
  echo "Tracked worktree must be clean before scaled fitting" >&2
  exit 2
fi
export BE_CODE_REVISION
BE_CODE_REVISION=$(git -C "${repo_dir}" rev-parse HEAD)
test -s "${feature_dir}/rollouts.audit.json"
test -s "${feature_dir}/label-free-audit.json"
rollouts_sha256=$(sha256sum "${rollouts}")
rollouts_sha256=${rollouts_sha256%% *}
features_sha256=$(sha256sum "${features}")
features_sha256=${features_sha256%% *}
"${python_bin}" scripts/fit_scaled_textvqa_action_value.py \
  --rollouts "${rollouts}" \
  --expected-rollouts-sha256 "${rollouts_sha256}" \
  --features "${features}" \
  --expected-features-sha256 "${features_sha256}" \
  --output-dir "${output_dir}" \
  --feature-mode semantic-context \
  --bootstrap-resamples 2000
