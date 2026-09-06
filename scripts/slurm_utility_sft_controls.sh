#!/usr/bin/env bash
#SBATCH --partition=q-hgpu-small
#SBATCH --gres=gpu:h800:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --job-name=be-utility-controls
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-utility-controls-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL
#SBATCH --no-requeue

set -euo pipefail
repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
: "${BE_UTILITY_FORMAT_PLAN:?missing format plan}"
: "${BE_UTILITY_FORMAT_SHA256:?missing format plan hash}"
: "${BE_UTILITY_BEST_PLAN:?missing best-action plan}"
: "${BE_UTILITY_BEST_SHA256:?missing best-action plan hash}"
cd "${repo_dir}"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_CACHE=/userhome/cs3/yihangc/Data/hf_cache/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${repo_dir}/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=0 "${python_bin}" scripts/execute_utility_sft_smoke.py \
  --plan "${BE_UTILITY_FORMAT_PLAN}" --sha256 "${BE_UTILITY_FORMAT_SHA256}" &
format_pid=$!
CUDA_VISIBLE_DEVICES=1 "${python_bin}" scripts/execute_utility_sft_smoke.py \
  --plan "${BE_UTILITY_BEST_PLAN}" --sha256 "${BE_UTILITY_BEST_SHA256}" &
best_pid=$!

exit_code=0
wait "${format_pid}" || exit_code=$?
wait "${best_pid}" || exit_code=$?
exit "${exit_code}"
