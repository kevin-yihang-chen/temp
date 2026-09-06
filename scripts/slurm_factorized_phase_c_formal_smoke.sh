#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --job-name=be-phase-c-formal-smoke
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-phase-c-formal-smoke-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL
#SBATCH --no-requeue

set -euo pipefail
required=(
  BE_PHASE_C_SMOKE_CODE_REVISION BE_PHASE_C_SMOKE_WORKER_SHA256
  BE_PHASE_C_SMOKE_SCRIPT_SHA256 BE_PHASE_C_SMOKE_TRAIN_JOB
  BE_PHASE_C_SMOKE_SEED
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "missing ${name}" >&2; exit 2; }
done
repo=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
worker="${repo}/scripts/slurm_factorized_phase_c_formal_smoke.sh"
script="${repo}/scripts/smoke_factorized_phase_c_formal_runtime.py"
digest() { sha256sum "$1" | cut -d ' ' -f 1; }
[[ "$(git -C "${repo}" rev-parse HEAD)" == "${BE_PHASE_C_SMOKE_CODE_REVISION}" ]] || exit 2
[[ -z "$(git -C "${repo}" status --porcelain --untracked-files=no)" ]] || exit 2
[[ "$(digest "${worker}")" == "${BE_PHASE_C_SMOKE_WORKER_SHA256}" ]] || exit 2
[[ "$(digest "${script}")" == "${BE_PHASE_C_SMOKE_SCRIPT_SHA256}" ]] || exit 2
cd "${repo}"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_CACHE=/userhome/cs3/yihangc/Data/hf_cache/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${repo}/src:${repo}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
output="${repo}/artifacts/factorized-potential-outcomes-v1/phase-c-formal-smoke/job-${SLURM_JOB_ID}/report.json"
"${python_bin}" scripts/smoke_factorized_phase_c_formal_runtime.py \
  --matrix configs/factorized_phase_c_training_matrix_v1.json \
  --training-root artifacts/factorized-potential-outcomes-v1/phase-c-training \
  --job-id "${BE_PHASE_C_SMOKE_TRAIN_JOB}" \
  --seed "${BE_PHASE_C_SMOKE_SEED}" \
  --output "${output}"
