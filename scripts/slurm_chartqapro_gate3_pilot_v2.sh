#!/usr/bin/env bash
#SBATCH --job-name=be-cqapro-v2
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-chartqapro-gate3-pilot-v2-%j.out
#SBATCH --mail-type=ALL

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
run_dir="${repo_dir}/artifacts/gate3-chartqapro-pilot-309/qwen3b-c4-direct-seed0-gate-context-v2"
: "${BE_CODE_REVISION:?BE_CODE_REVISION must be frozen at submission}"

export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${repo_dir}/src"

cd "${repo_dir}"
mkdir -p "${run_dir}"
"${python_bin}" -m beyond_entropy collect-qwen \
  --manifest "${repo_dir}/data/chartqapro-gate3-e27c287-v2/pilot/manifest.jsonl" \
  --expected-manifest-sha256 b5a61ebc91e8ac94686af13af47ca8714df9b290bae239d820d699c510f7fe4d \
  --output "${run_dir}/rollouts.jsonl" \
  --resume \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --model-revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --scorer chartqapro \
  --candidate-count 4 \
  --generation-seeds 0 \
  --bootstrap-resamples 2000 \
  --bootstrap-seed 0 \
  --limit 309 \
  --max-new-tokens 16 \
  --min-pixels 200704 \
  --max-pixels 602112 \
  --attention-implementation sdpa \
  --system-prompt "You are a helpful assistant."
