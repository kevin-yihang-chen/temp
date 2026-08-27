#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-question-reembed-%j.out
#SBATCH --mail-type=ALL

set -euo pipefail

: "${BE_SEMANTIC_SOURCE_FEATURES:?missing BE_SEMANTIC_SOURCE_FEATURES}"
: "${BE_SEMANTIC_ROLLOUTS:?missing BE_SEMANTIC_ROLLOUTS}"
: "${BE_SEMANTIC_OUTPUT_FEATURES:?missing BE_SEMANTIC_OUTPUT_FEATURES}"

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
model_revision=66285546d2b821cf421d4f5eb2576359d3770cd3

export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${repo_dir}/src"

resume_args=()
if [[ -e "${BE_SEMANTIC_OUTPUT_FEATURES}" ]]; then
  resume_args=(--resume)
fi

cd "${repo_dir}"
export BE_CODE_REVISION
BE_CODE_REVISION=$(git rev-parse HEAD)
"${python_bin}" scripts/reembed_contextual_questions.py \
  --source-features "${BE_SEMANTIC_SOURCE_FEATURES}" \
  --rollouts "${BE_SEMANTIC_ROLLOUTS}" \
  --output "${BE_SEMANTIC_OUTPUT_FEATURES}" \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --revision "${model_revision}" \
  --device-map cuda:0 \
  --dtype bfloat16 \
  --attention-implementation sdpa \
  --batch-size 64 \
  --checkpoint-interval 512 \
  "${resume_args[@]}"
