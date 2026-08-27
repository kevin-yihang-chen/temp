#!/usr/bin/env bash
#SBATCH --job-name=be-train-repl-analysis
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-train-replication-analysis-%j.out
#SBATCH --mail-type=ALL

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
target_dir="${repo_dir}/artifacts/replication-chartqa-train-4500/qwen3b-c4-concise-seed0"

export PYTHONPATH="${repo_dir}/src"
cd "${repo_dir}"
export BE_CODE_REVISION
BE_CODE_REVISION=$(git rev-parse HEAD)
"${python_bin}" scripts/analyze_frozen_confirmation.py \
  --target-rollouts "${target_dir}/rollouts.jsonl" \
  --target-provenance "${target_dir}/rollouts.provenance.json" \
  --target-manifest "${repo_dir}/data/chartqa-train-replication-4500/manifest.jsonl" \
  --frozen-model "${repo_dir}/artifacts/gate2-transfer-chartqa-vstar/factorized-context-v2-quantile/model.json" \
  --source-report "${repo_dir}/artifacts/gate2-transfer-chartqa-vstar/factorized-context-v2-quantile/report.json" \
  --secondary-action-model "${repo_dir}/artifacts/gate2-transfer-chartqa-val/context-quadrant-secondary-v1/model.json" \
  --secondary-source-report "${repo_dir}/artifacts/gate2-transfer-chartqa-val/context-quadrant-secondary-v1/report.json" \
  --secondary-text-model "${repo_dir}/artifacts/gate2-transfer-chartqa-val/factorized-text-secondary-v1/model.json" \
  --secondary-text-report "${repo_dir}/artifacts/gate2-transfer-chartqa-val/factorized-text-secondary-v1/report.json" \
  --output-dir "${repo_dir}/artifacts/replication-chartqa-train-4500/frozen-factorized-context-v1" \
  --expected-model-sha256 5d5c0f781a7141726e786d6ad87b861a6395c489bcf9ad8567a2e9ca825c3330 \
  --expected-source-report-sha256 1f05ddeef52fa9abced549479cdb8fa386578d12600fb874a964a12a4d927462 \
  --expected-target-manifest-sha256 72db6feaa4bc042e98741a48dd55421c5246c1b48c84b1fd75740d1d072ca621 \
  --expected-secondary-action-model-sha256 5989974482785b31868473e7a925708d15f6f1fbac3095906ded7a88def53bbd \
  --expected-secondary-source-report-sha256 c0510901bc351ea9bac799497775ff53f7bb42b23ad574b555a7995c9922f35c \
  --expected-secondary-text-model-sha256 175c044ceca6b755b8cc16b3f106604cfc1b54396b695bf9c685a81ffd162fa5 \
  --expected-secondary-text-report-sha256 ce1caa6bb08054d4ab30dea7af09e61f95d08eb91598e710275fe441adf43fc4 \
  --expected-rollout-code-revision cdc425ed892a7a4c6c8365be622cbc113544a2dc \
  --expected-examples 4500 \
  --report-title "ChartQA train high-power replication" \
  --require-image-ci \
  --lambda-cost 0.05 \
  --bootstrap-resamples 5000 \
  --bootstrap-seed 0
