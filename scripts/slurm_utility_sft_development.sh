#!/usr/bin/env bash
#SBATCH --partition=q-h800
#SBATCH --gres=gpu:h800:3
#SBATCH --cpus-per-task=36
#SBATCH --mem=192G
#SBATCH --time=01:00:00
#SBATCH --job-name=be-utility-dev
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-utility-dev-%j.out
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

CUDA_VISIBLE_DEVICES=0 "${python_bin}" scripts/execute_utility_sft_development.py --plan "${FORMAT_PLAN}" --sha256 "${FORMAT_SHA256}" &
p0=$!
CUDA_VISIBLE_DEVICES=1 "${python_bin}" scripts/execute_utility_sft_development.py --plan "${BEST_PLAN}" --sha256 "${BEST_SHA256}" &
p1=$!
CUDA_VISIBLE_DEVICES=2 "${python_bin}" scripts/execute_utility_sft_development.py --plan "${UTILITY_PLAN}" --sha256 "${UTILITY_SHA256}" &
p2=$!
status=0
wait "${p0}" || status=$?
wait "${p1}" || status=$?
wait "${p2}" || status=$?
exit "${status}"
