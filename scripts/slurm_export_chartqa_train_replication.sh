#!/usr/bin/env bash
#SBATCH --job-name=be-export-train-repl
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-export-train-replication-%j.out
#SBATCH --mail-type=ALL

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/bin/python
external_dir="${repo_dir}/data/external/chartqa-hfm4-b605b6e-train"

export PYTHONPATH="${repo_dir}/src"
cd "${repo_dir}"
export BE_CODE_REVISION
BE_CODE_REVISION=$(git rev-parse HEAD)
"${python_bin}" scripts/export_hfm4_chartqa_train_replication.py \
  --parquet \
    "${external_dir}/train-00000-of-00003-49492f364babfa44.parquet" \
    "${external_dir}/train-00001-of-00003-7302bae5e425bbc7.parquet" \
    "${external_dir}/train-00002-of-00003-194c9400785577a2.parquet" \
  --development-manifest "${repo_dir}/data/chartqa-frozen-2500/manifest.jsonl" \
  --validation-manifest "${repo_dir}/data/chartqa-val-confirmation-1918/manifest.jsonl" \
  --output-dir "${repo_dir}/data/chartqa-train-replication-4500" \
  --count 4500 \
  --seed 29
