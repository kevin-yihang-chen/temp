#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-frozen-confirmation-%j.out
#SBATCH --mail-type=ALL

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
target_rollouts="${repo_dir}/artifacts/confirmation-chartqa-val-1918/qwen3b-c4-concise-seed0/rollouts.jsonl"
target_provenance="${repo_dir}/artifacts/confirmation-chartqa-val-1918/qwen3b-c4-concise-seed0/rollouts.provenance.json"
target_manifest="${repo_dir}/data/chartqa-val-confirmation-1918/manifest.jsonl"
frozen_model="${repo_dir}/artifacts/gate2-transfer-chartqa-vstar/factorized-context-v2-quantile/model.json"
source_report="${repo_dir}/artifacts/gate2-transfer-chartqa-vstar/factorized-context-v2-quantile/report.json"
secondary_action_model="${repo_dir}/artifacts/gate2-transfer-chartqa-val/context-quadrant-secondary-v1/model.json"
secondary_source_report="${repo_dir}/artifacts/gate2-transfer-chartqa-val/context-quadrant-secondary-v1/report.json"
secondary_text_model="${repo_dir}/artifacts/gate2-transfer-chartqa-val/factorized-text-secondary-v1/model.json"
secondary_text_report="${repo_dir}/artifacts/gate2-transfer-chartqa-val/factorized-text-secondary-v1/report.json"
output_dir="${repo_dir}/artifacts/confirmation-chartqa-val-1918/frozen-factorized-context-v1"

export PYTHONPATH="${repo_dir}/src"

cd "${repo_dir}"
export BE_CODE_REVISION
BE_CODE_REVISION=$(git rev-parse HEAD)
"${python_bin}" scripts/analyze_frozen_confirmation.py \
  --target-rollouts "${target_rollouts}" \
  --target-provenance "${target_provenance}" \
  --target-manifest "${target_manifest}" \
  --frozen-model "${frozen_model}" \
  --source-report "${source_report}" \
  --secondary-action-model "${secondary_action_model}" \
  --secondary-source-report "${secondary_source_report}" \
  --secondary-text-model "${secondary_text_model}" \
  --secondary-text-report "${secondary_text_report}" \
  --output-dir "${output_dir}" \
  --expected-model-sha256 5d5c0f781a7141726e786d6ad87b861a6395c489bcf9ad8567a2e9ca825c3330 \
  --expected-source-report-sha256 1f05ddeef52fa9abced549479cdb8fa386578d12600fb874a964a12a4d927462 \
  --expected-target-manifest-sha256 d3178218853b10447228963e839716f0eac768b51bdc0f5b4a83268d3819b58b \
  --expected-secondary-action-model-sha256 5989974482785b31868473e7a925708d15f6f1fbac3095906ded7a88def53bbd \
  --expected-secondary-source-report-sha256 c0510901bc351ea9bac799497775ff53f7bb42b23ad574b555a7995c9922f35c \
  --expected-secondary-text-model-sha256 175c044ceca6b755b8cc16b3f106604cfc1b54396b695bf9c685a81ffd162fa5 \
  --expected-secondary-text-report-sha256 ce1caa6bb08054d4ab30dea7af09e61f95d08eb91598e710275fe441adf43fc4 \
  --lambda-cost 0.05 \
  --bootstrap-resamples 5000 \
  --bootstrap-seed 0
