#!/usr/bin/env bash
#SBATCH --job-name=be-cqapro-formal
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-chartqapro-gate3-formal-%j.out
#SBATCH --mail-type=ALL

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
run_dir="${repo_dir}/artifacts/gate3-chartqapro-formal-1625/qwen3b-c4-direct-seed0"
: "${BE_CODE_REVISION:?BE_CODE_REVISION must be frozen at submission}"

export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${repo_dir}/src"

cd "${repo_dir}"
mkdir -p "${run_dir}"
"${python_bin}" -m beyond_entropy collect-qwen \
  --manifest "${repo_dir}/data/chartqapro-gate3-e27c287-v2/formal/manifest.jsonl" \
  --expected-manifest-sha256 5a3ddca2e6476196aac8ad4fa7bc00033f2ac9c39d2011fe21fa070e965b97d4 \
  --output "${run_dir}/rollouts.jsonl" \
  --resume \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --model-revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --scorer chartqapro \
  --candidate-count 4 \
  --generation-seeds 0 \
  --bootstrap-resamples 2000 \
  --bootstrap-seed 0 \
  --limit 1625 \
  --max-new-tokens 16 \
  --min-pixels 200704 \
  --max-pixels 602112 \
  --attention-implementation sdpa \
  --system-prompt "You are a helpful assistant."
