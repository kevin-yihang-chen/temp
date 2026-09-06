#!/usr/bin/env bash
#SBATCH --partition=q-hgpu-small
#SBATCH --gres=gpu:h800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=be-utility-sft-smoke
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-utility-sft-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL
#SBATCH --no-requeue

set -euo pipefail
repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
: "${BE_UTILITY_PLAN:?missing frozen smoke plan}"
: "${BE_UTILITY_PLAN_SHA256:?missing plan hash}"
cd "${repo_dir}"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_CACHE=/userhome/cs3/yihangc/Data/hf_cache/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${repo_dir}/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1

# Standard-library-only verification before importing new training modules.
"${python_bin}" scripts/execute_utility_sft_smoke.py \
  --plan "${BE_UTILITY_PLAN}" --sha256 "${BE_UTILITY_PLAN_SHA256}"
