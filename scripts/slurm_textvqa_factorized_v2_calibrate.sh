#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-textvqa-factorized-v2-calibrate-%j.out
#SBATCH --mail-type=ALL

set -euo pipefail

: "${BE_FV2_ROLLOUTS_SHA256:?missing BE_FV2_ROLLOUTS_SHA256}"
: "${BE_FV2_ROLLOUT_AUDIT_SHA256:?missing BE_FV2_ROLLOUT_AUDIT_SHA256}"
: "${BE_FV2_FEATURES_SHA256:?missing BE_FV2_FEATURES_SHA256}"
: "${BE_CODE_REVISION:?missing BE_CODE_REVISION}"

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
candidate="${repo_dir}/artifacts/textvqa-train-factorized-v2/frozen-candidate/model.json"
allocation="${repo_dir}/data/textvqa-train-factorized-v2/allocation.json"
allocation_audit="${repo_dir}/data/textvqa-train-factorized-v2/allocation.audit.json"
role_dir="${repo_dir}/data/textvqa-train-factorized-v2/risk-calibration"
manifest="${role_dir}/manifest.jsonl"
manifest_provenance="${role_dir}/manifest.provenance.json"
artifact_root="${repo_dir}/artifacts/textvqa-train-factorized-v2/risk-calibration"
rollouts="${artifact_root}/qwen3b-c4-seed0/rollouts.jsonl"
rollout_audit="${artifact_root}/attention-semantic-v1/rollouts.audit.json"
features="${artifact_root}/attention-semantic-v1/features-question-region-attention-label-free.pt"
protocol="${repo_dir}/docs/textvqa_factorized_fixed_sequence_preregistration.md"
output_dir="${repo_dir}/artifacts/textvqa-train-factorized-v2/fixed-sequence-calibrated"

if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "Tracked worktree must be clean before fixed-sequence calibration" >&2
  exit 2
fi
if [[ "$(git -C "${repo_dir}" rev-parse HEAD)" != "${BE_CODE_REVISION}" ]]; then
  echo "Calibration code revision differs from submitted revision" >&2
  exit 2
fi
if [[ -e "${output_dir}/calibration.json" || -e "${output_dir}/model.json" ]]; then
  echo "Refusing to overwrite fixed-sequence calibration output" >&2
  exit 2
fi

export PYTHONPATH="${repo_dir}/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${repo_dir}"
"${python_bin}" scripts/calibrate_factorized_textvqa_fixed_sequence.py \
  --candidate "${candidate}" \
  --allocation "${allocation}" \
  --allocation-audit "${allocation_audit}" \
  --manifest "${manifest}" \
  --manifest-provenance "${manifest_provenance}" \
  --rollouts "${rollouts}" \
  --expected-rollouts-sha256 "${BE_FV2_ROLLOUTS_SHA256}" \
  --rollout-audit "${rollout_audit}" \
  --expected-rollout-audit-sha256 "${BE_FV2_ROLLOUT_AUDIT_SHA256}" \
  --features "${features}" \
  --expected-features-sha256 "${BE_FV2_FEATURES_SHA256}" \
  --protocol "${protocol}" \
  --output-dir "${output_dir}"
