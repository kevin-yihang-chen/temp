#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-semantic-%x-%j.out
#SBATCH --mail-type=ALL

set -euo pipefail

: "${BE_SEMANTIC_ROLLOUTS:?missing BE_SEMANTIC_ROLLOUTS}"
: "${BE_SEMANTIC_ROLLOUTS_SHA256:?missing BE_SEMANTIC_ROLLOUTS_SHA256}"
: "${BE_SEMANTIC_RUN_DIR:?missing BE_SEMANTIC_RUN_DIR}"

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python

export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${repo_dir}/src"

cd "${repo_dir}"
export BE_CODE_REVISION
BE_CODE_REVISION=$(git rev-parse HEAD)
mkdir -p "${BE_SEMANTIC_RUN_DIR}"
"${python_bin}" -m beyond_entropy extract-qwen-features \
  --rollouts "${BE_SEMANTIC_ROLLOUTS}" \
  --expected-rollouts-sha256 "${BE_SEMANTIC_ROLLOUTS_SHA256}" \
  --output "${BE_SEMANTIC_RUN_DIR}/features.pt" \
  --resume \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --model-revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --min-pixels 200704 \
  --max-pixels 602112 \
  --attention-implementation sdpa

"${python_bin}" -m beyond_entropy fit-semantic \
  --features "${BE_SEMANTIC_RUN_DIR}/features.pt" \
  --rollouts "${BE_SEMANTIC_ROLLOUTS}" \
  --output-dir "${BE_SEMANTIC_RUN_DIR}/semantic-model" \
  --split-group image_id \
  --train-fraction 0.7 \
  --validation-fraction 0.2 \
  --lambda-costs 0 0.01 0.05 0.1 0.2 \
  --hidden-dim 64 \
  --dropout 0.2 \
  --learning-rate 0.001 \
  --weight-decay 0.001 \
  --rank-weight 1.0 \
  --max-epochs 500 \
  --patience 50 \
  --seed 17 \
  --device cuda
