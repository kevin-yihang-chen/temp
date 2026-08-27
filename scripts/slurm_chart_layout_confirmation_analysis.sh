#!/usr/bin/env bash
#SBATCH --job-name=be-chart-layout-analysis
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-chart-layout-analysis-%j.out
#SBATCH --mail-type=ALL

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
treatment_dir="${repo_dir}/artifacts/confirmation-chart-layout-2137/qwen3b-chart-layout-c4-concise-seed0"

export PYTHONPATH="${repo_dir}/src"
cd "${repo_dir}"
export BE_CODE_REVISION
BE_CODE_REVISION=$(git rev-parse HEAD)
"${python_bin}" scripts/analyze_chart_layout_confirmation.py \
  --baseline-rollouts "${repo_dir}/artifacts/gate1-chartqa-2500/qwen3b-c4-concise-seed0/rollouts.jsonl" \
  --treatment-rollouts "${treatment_dir}/rollouts.jsonl" \
  --treatment-provenance "${treatment_dir}/rollouts.provenance.json" \
  --target-manifest "${repo_dir}/data/chartqa-frozen-2500/manifest.chart-layout-confirm.jsonl" \
  --output-dir "${repo_dir}/artifacts/confirmation-chart-layout-2137/matched-comparison-v1" \
  --bootstrap-resamples 5000
