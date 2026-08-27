#!/usr/bin/env bash
#SBATCH --job-name=be-sem-smoke
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-semantic-smoke-%j.out
#SBATCH --mail-type=ALL

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
run_dir="${repo_dir}/artifacts/gate2-semantic-smoke/${SLURM_JOB_ID}"

export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${repo_dir}/src"

cd "${repo_dir}"
export BE_CODE_REVISION
BE_CODE_REVISION=$(git rev-parse HEAD)
mkdir -p "${run_dir}"
"${python_bin}" -m beyond_entropy extract-qwen-features \
  --rollouts "${repo_dir}/artifacts/gate1-smoke/189820/rollouts.jsonl" \
  --expected-rollouts-sha256 75dab17d9a22acef4db9f3da0e7a9f12e739ec1644e41c5c10f7311947309825 \
  --output "${run_dir}/features.pt" \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --model-revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --min-pixels 200704 \
  --max-pixels 602112 \
  --attention-implementation sdpa
