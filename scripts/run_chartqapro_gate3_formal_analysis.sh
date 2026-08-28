#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/bin/python
rollouts="${repo_dir}/artifacts/gate3-chartqapro-formal-1625/qwen3b-c4-direct-seed0/rollouts.jsonl"

if [[ ! -r "${rollouts}" || ! -r "${rollouts%.jsonl}.provenance.json" ]]; then
  echo "Formal rollout or provenance is incomplete" >&2
  exit 2
fi
if [[ "$(wc -l < "${rollouts}")" -ne 8125 ]]; then
  echo "Expected 8,125 complete formal sibling records" >&2
  exit 2
fi

cd "${repo_dir}"
export PYTHONPATH="${repo_dir}/src"
"${python_bin}" scripts/analyze_chartqapro_formal.py \
  --rollouts "${rollouts}" \
  --manifest "${repo_dir}/data/chartqapro-gate3-e27c287-v2/formal/manifest.jsonl" \
  --frozen-model "${repo_dir}/artifacts/gate2-transfer-chartqa-vstar/factorized-context-v1/model.json" \
  --source-report "${repo_dir}/artifacts/gate2-transfer-chartqa-vstar/factorized-context-v2-quantile/report.json" \
  --pilot-report "${repo_dir}/artifacts/gate3-chartqapro-pilot-309/analysis-v2-final/report.json" \
  --replay-audit "${repo_dir}/artifacts/gate3-chartqapro-pilot-309/replay-audit-v1-v2.json" \
  --output-dir "${repo_dir}/artifacts/gate3-chartqapro-formal-1625/analysis-v1" \
  --bootstrap-resamples 5000 \
  --bootstrap-seed 0
