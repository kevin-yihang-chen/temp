#!/usr/bin/env bash
#SBATCH --job-name=be-vstar64-c9
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-vstar-c9-%j.out
#SBATCH --mail-type=ALL

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
run_dir="${repo_dir}/artifacts/gate1-vstar-64/qwen3b-c9-seed0"

export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${repo_dir}/src"

cd "${repo_dir}"
export BE_CODE_REVISION
BE_CODE_REVISION=$(git rev-parse HEAD)
mkdir -p "${run_dir}"
"${python_bin}" -m beyond_entropy collect-qwen \
  --manifest "${repo_dir}/data/vstar-frozen-64/manifest.jsonl" \
  --expected-manifest-sha256 3c0b93e94cd4fd925d4b9cbadddd559ec497a375d045cef842af0a5299e99900 \
  --output "${run_dir}/rollouts.jsonl" \
  --resume \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --model-revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --scorer vstar \
  --candidate-count 9 \
  --generation-seeds 0 \
  --bootstrap-resamples 2000 \
  --bootstrap-seed 0 \
  --limit 64 \
  --max-new-tokens 16 \
  --min-pixels 200704 \
  --max-pixels 602112 \
  --attention-implementation sdpa
