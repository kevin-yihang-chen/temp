#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-semantic-multiseed-%j.out
#SBATCH --mail-type=ALL

set -euo pipefail

: "${BE_SEMANTIC_FEATURES:?missing BE_SEMANTIC_FEATURES}"
: "${BE_SEMANTIC_ROLLOUTS:?missing BE_SEMANTIC_ROLLOUTS}"
: "${BE_SEMANTIC_MULTISEED_DIR:?missing BE_SEMANTIC_MULTISEED_DIR}"

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
for experiment_seed in 3 11 17 29 47; do
  "${python_bin}" -m beyond_entropy fit-semantic \
    --features "${BE_SEMANTIC_FEATURES}" \
    --rollouts "${BE_SEMANTIC_ROLLOUTS}" \
    --output-dir "${BE_SEMANTIC_MULTISEED_DIR}/seed-${experiment_seed}" \
    --split-group image_id \
    --train-fraction 0.7 \
    --validation-fraction 0.2 \
    --lambda-costs 0.05 \
    --hidden-dim 64 \
    --dropout 0.2 \
    --learning-rate 0.001 \
    --weight-decay 0.001 \
    --rank-weight 1.0 \
    --nonzero-weight 8.0 \
    --transition-weight 8.0 \
    --similarity-cv-folds 5 \
    --bootstrap-resamples 500 \
    --bootstrap-seed 0 \
    --max-epochs 500 \
    --patience 50 \
    --seed "${experiment_seed}" \
    --device cuda
done
