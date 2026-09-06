#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=144G
#SBATCH --time=01:00:00
#SBATCH --job-name=be-factorized-po
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-factorized-po-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL
#SBATCH --no-requeue

set -euo pipefail
repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
[[ -n "${CV_METHOD_PLAN:-}" ]] || { echo "missing CV_METHOD_PLAN" >&2; exit 2; }
[[ -n "${CV_METHOD_PLAN_SHA256:-}" ]] || { echo "missing CV_METHOD_PLAN_SHA256" >&2; exit 2; }
cd "${repo_dir}"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_CACHE=/userhome/cs3/yihangc/Data/hf_cache/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${repo_dir}/src" CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
"${python_bin}" scripts/execute_cv_method_stage.py \
  --plan "${CV_METHOD_PLAN}" --sha256 "${CV_METHOD_PLAN_SHA256}"

