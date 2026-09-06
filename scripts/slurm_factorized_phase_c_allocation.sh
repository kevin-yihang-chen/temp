#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=be-phase-c-alloc
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-phase-c-allocation-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL
#SBATCH --no-requeue

set -euo pipefail

required=(
  BE_PHASE_C_CODE_REVISION BE_PHASE_C_WORKER_SHA256 BE_PHASE_C_ALLOCATOR_SHA256
  BE_PHASE_C_HELPERS_SHA256 BE_PHASE_C_REUSED_ALLOCATOR_SHA256
  BE_PHASE_C_MANIFEST_EXPORT_SHA256 BE_PHASE_C_CONFIG_SHA256
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "missing ${name}" >&2; exit 2; }
done

repo=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
worker="${repo}/scripts/slurm_factorized_phase_c_allocation.sh"
allocator="${repo}/scripts/freeze_factorized_phase_c_data.py"
helpers="${repo}/src/beyond_entropy/phase_c_allocation.py"
reused="${repo}/scripts/freeze_predictability_data.py"
manifest_export="${repo}/src/beyond_entropy/manifest_export.py"
config="${repo}/configs/factorized_phase_c_allocation_v1.json"

check_hash() {
  local actual
  actual=$(sha256sum "$1")
  actual=${actual%% *}
  [[ "${actual}" == "$2" ]] || { echo "Phase-C allocation $3 hash mismatch" >&2; exit 2; }
}
[[ "$(git -C "${repo}" rev-parse HEAD)" == "${BE_PHASE_C_CODE_REVISION}" ]] || {
  echo "Phase-C allocation code revision mismatch" >&2; exit 2;
}
[[ -z "$(git -C "${repo}" status --porcelain --untracked-files=no)" ]] || {
  echo "tracked worktree must be clean" >&2; exit 2;
}
check_hash "${worker}" "${BE_PHASE_C_WORKER_SHA256}" worker
check_hash "${allocator}" "${BE_PHASE_C_ALLOCATOR_SHA256}" allocator
check_hash "${helpers}" "${BE_PHASE_C_HELPERS_SHA256}" helpers
check_hash "${reused}" "${BE_PHASE_C_REUSED_ALLOCATOR_SHA256}" reused_allocator
check_hash "${manifest_export}" "${BE_PHASE_C_MANIFEST_EXPORT_SHA256}" manifest_export
check_hash "${config}" "${BE_PHASE_C_CONFIG_SHA256}" config

export PYTHONPATH="${repo}/src:${repo}/scripts"
export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export BE_CODE_REVISION="${BE_PHASE_C_CODE_REVISION}"

"${python_bin}" "${allocator}" --config "${config}" --repository-root "${repo}"
