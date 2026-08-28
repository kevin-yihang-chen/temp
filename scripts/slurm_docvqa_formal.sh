#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-docvqa-formal-%x-%j.out
#SBATCH --mail-type=ALL

set -euo pipefail

: "${BE_FORMAL_MANIFEST:?missing BE_FORMAL_MANIFEST}"
: "${BE_FORMAL_MANIFEST_SHA256:?missing BE_FORMAL_MANIFEST_SHA256}"
: "${BE_FORMAL_RUN_DIR:?missing BE_FORMAL_RUN_DIR}"
: "${BE_CODE_REVISION:?missing BE_CODE_REVISION}"

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
rollouts="${BE_FORMAL_RUN_DIR}/rollouts.jsonl"

export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${repo_dir}/src"

cd "${repo_dir}"
mkdir -p "${BE_FORMAL_RUN_DIR}"
"${python_bin}" -m beyond_entropy collect-qwen \
  --manifest "${BE_FORMAL_MANIFEST}" \
  --expected-manifest-sha256 "${BE_FORMAL_MANIFEST_SHA256}" \
  --output "${rollouts}" \
  --resume \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --model-revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --scorer docvqa \
  --candidate-count 4 \
  --proposer ug-grid \
  --visual-crop-ratio 2.0 \
  --visual-cost 1.0 \
  --generation-seeds 0 \
  --bootstrap-resamples 5000 \
  --bootstrap-seed 20260828 \
  --scientific-status "outcome-unseen DocVQA formal confirmation; frozen OOF action-value model" \
  --max-new-tokens 32 \
  --min-pixels 200704 \
  --max-pixels 602112 \
  --attention-implementation sdpa \
  --system-prompt "You are a helpful assistant."
