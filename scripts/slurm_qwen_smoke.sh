#!/usr/bin/env bash
#SBATCH --job-name=be-qwen-smoke
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-%j.out
#SBATCH --mail-type=ALL

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
run_dir="${repo_dir}/artifacts/gate1-smoke/${SLURM_JOB_ID}"

export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${repo_dir}/src"

cd "${repo_dir}"
mkdir -p "${run_dir}"
"${python_bin}" scripts/make_smoke_fixture.py --output-dir "${run_dir}"
"${python_bin}" -m beyond_entropy collect-qwen \
  --manifest "${run_dir}/manifest.jsonl" \
  --output "${run_dir}/rollouts.jsonl" \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --model-revision cc594898137f460bfe9f0759e9844b3ce807cfb5 \
  --scorer vstar \
  --candidate-count 1 \
  --generation-seeds 0 \
  --max-new-tokens 4 \
  --max-pixels 602112 \
  --attention-implementation sdpa
