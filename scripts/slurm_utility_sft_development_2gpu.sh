#!/usr/bin/env bash
#SBATCH --partition=q-h800
#SBATCH --gres=gpu:h800:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --job-name=be-util-dev-2g
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-utility-dev-2gpu-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL
#SBATCH --no-requeue

set -euo pipefail
repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
for name in FORMAT_PLAN FORMAT_SHA256 BEST_PLAN BEST_SHA256 UTILITY_PLAN UTILITY_SHA256; do
  [[ -n "${!name:-}" ]] || { echo "missing ${name}" >&2; exit 2; }
done
cd "${repo_dir}"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_CACHE=/userhome/cs3/yihangc/Data/hf_cache/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${repo_dir}/src" CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1

# Format and Best-Action are independent.  Utility starts on GPU 0 only after
# Format releases it, keeping peak allocation to two H800s without model merge.
CUDA_VISIBLE_DEVICES=0 "${python_bin}" scripts/execute_utility_sft_development.py \
  --plan "${FORMAT_PLAN}" --sha256 "${FORMAT_SHA256}" &
format_pid=$!
CUDA_VISIBLE_DEVICES=1 "${python_bin}" scripts/execute_utility_sft_development.py \
  --plan "${BEST_PLAN}" --sha256 "${BEST_SHA256}" &
best_pid=$!

status=0
wait "${format_pid}" || status=$?
if [[ "${status}" -eq 0 ]]; then
  CUDA_VISIBLE_DEVICES=0 "${python_bin}" scripts/execute_utility_sft_development.py \
    --plan "${UTILITY_PLAN}" --sha256 "${UTILITY_SHA256}" &
  utility_pid=$!
else
  utility_pid=""
fi
wait "${best_pid}" || status=$?
if [[ -n "${utility_pid}" ]]; then
  wait "${utility_pid}" || status=$?
fi
exit "${status}"
