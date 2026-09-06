#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --job-name=be-phase-c-formal
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-phase-c-formal-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL
#SBATCH --no-requeue

set -euo pipefail

required=(
  BE_PHASE_C_FORMAL_PLAN BE_PHASE_C_FORMAL_PLAN_SHA256
  BE_PHASE_C_FORMAL_CODE_REVISION BE_PHASE_C_FORMAL_WORKER_SHA256
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "missing ${name}" >&2; exit 2; }
done

repo=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
worker="${repo}/scripts/slurm_factorized_phase_c_formal.sh"

digest() { sha256sum "$1" | cut -d ' ' -f 1; }
[[ "$(git -C "${repo}" rev-parse HEAD)" == "${BE_PHASE_C_FORMAL_CODE_REVISION}" ]] || {
  echo "formal code revision mismatch" >&2; exit 2;
}
[[ -z "$(git -C "${repo}" status --porcelain --untracked-files=no)" ]] || {
  echo "tracked worktree must be clean for formal access" >&2; exit 2;
}
[[ "$(digest "${worker}")" == "${BE_PHASE_C_FORMAL_WORKER_SHA256}" ]] || {
  echo "formal worker hash mismatch" >&2; exit 2;
}
[[ "$(digest "${BE_PHASE_C_FORMAL_PLAN}")" == "${BE_PHASE_C_FORMAL_PLAN_SHA256}" ]] || {
  echo "formal plan hash mismatch" >&2; exit 2;
}

cd "${repo}"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_CACHE=/userhome/cs3/yihangc/Data/hf_cache/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${repo}/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4

"${python_bin}" scripts/execute_factorized_phase_c_formal.py \
  --plan "${BE_PHASE_C_FORMAL_PLAN}" \
  --plan-sha256 "${BE_PHASE_C_FORMAL_PLAN_SHA256}"
